import numpy as _np
from scipy.linalg import eigh_tridiagonal
from copy import deepcopy
from numba import njit

__all__ = ["lanczos_full", "lanczos_iter", "lin_comb_Q_T"]


@njit
def _axpy(x, y, a):
    for i in range(x.size):
        y[i] += a * x[i]


def _lanczos_vec_iter_core(A, v0, a, b):
    dtype = _np.result_type(A.dtype, v0.dtype)

    q = v0.astype(dtype, copy=True)

    q_norm = _np.linalg.norm(q)
    if _np.abs(q_norm - 1.0) > _np.finfo(dtype).eps:
        _np.divide(q, q_norm, out=q)

    q_view = q[:]
    q_view.setflags(write=0, uic=0)

    m = a.size
    n = q.size

    v = _np.zeros_like(v0, dtype=dtype)
    r = _np.zeros_like(v0, dtype=dtype)

    try:
        A.dot(q, out=r)
        use_out = True
    except TypeError:
        r[:] = A.dot(q)
        use_out = False

    _axpy(q, r, -a[0])

    yield q_view  # return non-writable array

    for i in range(1, m, 1):
        v[:] = q[:]

        _np.divide(r, b[i - 1], out=q)

        if use_out:
            A.dot(q, out=r)
        else:
            r[:] = A.dot(q)

        _axpy(v, r, -b[i - 1])
        _axpy(q, r, -a[i])

        yield q_view  # return non-writable array


class _lanczos_vec_iter(object):
    def __init__(self, A, v0, a, b):
        self._A = A
        self._v0 = v0
        self._a = a
        self._b = b

    def __iter__(self):
        return _lanczos_vec_iter_core(self._A, self._v0, self._a, self._b)

    def __del__(self):
        del self._A
        del self._v0
        del self._b
        del self._a


def lanczos_full(A, v0, m, full_ortho=False, out=None, eps=None):
    """Creates Lanczos basis; diagonalizes Krylov subspace in Lanczos basis.

    Given a hermitian matrix `A` of size :math:`n\\times n` and an integer `m`, the Lanczos algorithm computes

    * an :math:`n\\times m` matrix  :math:`Q`, and
    * a real symmetric tridiagonal matrix :math:`T=Q^\\dagger A Q` of size :math:`m\\times m`. The matrix :math:`T` can be represented via its eigendecomposition `(E,V)`: :math:`T=V\\mathrm{diag}(E)V^T`.

    This function computes the triple :math:`(E,V,Q^T)`.

    :red:`NOTE:` This function returns :math:`Q^T;\\,Q^T` is (in general) different from :math:`Q^\\dagger`.


    Notes
    -----

    * performs classical lanczos algorithm for hermitian matrices and cannot handle degeneracies when calculating eigenvalues.
    * the function allows for full orthogonalization, see `full_ortho`. The resulting :math:`T` will not neccesarily be tridiagonal.
    * `V` is always real-valued, since :math:`T` is real and symmetric.
    * `A` must have a 'dot' method to perform calculation,
    * The 'out' argument to pass back the results of the matrix-vector product will be used if the 'dot' function supports this argument.

    Parameters
    ----------
    A : LinearOperator, hamiltonian, numpy.ndarray, or object with a 'dot' method and a 'dtype' method.
            Python object representing a linear map to compute the Lanczos approximation to the largest eigenvalues/vectors of. Must contain a dot-product method, used as `A.dot(v)` and a dtype method, used as `A.dtype`, e.g. `hamiltonian`, `quantum_operator`, `quantum_LinearOperator`, sparse or dense matrix.
    v0 : array_like, (n,)
            initial vector to start the Lanczos algorithm from.
    m : int
            Number of Lanczos vectors (size of the Krylov subspace)
    full_ortho : bool, optional
            perform a QR decomposition on Q_T generated from the standard lanczos iteration to remove any loss of orthogonality due to numerical precision.
    out : numpy.ndarray, optional
            Array to store the Lanczos vectors in (e.g. `Q`). in memory efficient way.
    eps : float, optional
            Used to cutoff lanczos iteration when off diagonal matrix elements of `T` drops below this value.

    Returns
    -------
    tuple(E,V,Q_T)
            * E : (m,) numpy.ndarray: eigenvalues of Krylov subspace tridiagonal matrix :math:`T`.
            * V : (m,m) numpy.ndarray: eigenvectors of Krylov subspace tridiagonal matrix :math:`T`.
            * Q_T : (m,n) numpy.ndarray: matrix containing the `m` Lanczos vectors. This is :math:`Q^T` (not :math:`Q^\\dagger`)!

    Examples
    --------

    >>> E, V, Q_T = lanczos_full(H,v0,20)



    """

    v0 = _np.asanyarray(v0)
    n = v0.size
    dtype = _np.result_type(A.dtype, v0.dtype)

    if v0.ndim != 1:
        raise ValueError("expecting array with ndim=1 for initial Lanczos vector.")

    if m >= n:
        raise ValueError(
            "Requested size of Lanczos basis must be smaller then size of original space (e.g. m < n)."
        )

    if out is not None:
        if out.shape != (m, n):
            raise ValueError("argument 'out' must have shape (m,n), see documentation.")
        if out.dtype != dtype:
            raise ValueError(
                "argument 'out' has dtype {}, expecting dtype {}".format(
                    out.dtype, dtype
                )
            )
        if not out.flags["CARRAY"]:
            raise ValueError("argument 'out' must be C-contiguous and writable.")

        Q = out
    else:
        Q = _np.zeros((m, n), dtype=dtype)

    Q[0, :] = v0[:]
    v = _np.zeros_like(v0, dtype=dtype)
    r = _np.zeros_like(v0, dtype=dtype)

    b = _np.zeros((m,), dtype=v.real.dtype)
    a = _np.zeros((m,), dtype=v.real.dtype)

    if eps is None:
        eps = _np.finfo(dtype).eps

    q_norm = _np.linalg.norm(Q[0, :])

    if _np.abs(q_norm - 1.0) > eps:
        _np.divide(Q[0, :], q_norm, out=Q[0, :])

    try:
        A.dot(Q[0, :], out=r)  # call if operator supports 'out' argument
        use_out = True
    except TypeError:
        r[:] = A.dot(Q[0, :])
        use_out = False

    a[0] = _np.vdot(Q[0, :], r).real

    _axpy(Q[0, :], r, -a[0])
    b[0] = _np.linalg.norm(r)

    i = 0
    for i in range(1, m, 1):
        v[:] = Q[i - 1, :]

        _np.divide(r, b[i - 1], out=Q[i, :])

        if use_out:
            A.dot(Q[i, :], out=r)  # call if operator supports 'out' argument
        else:
            r[:] = A.dot(Q[i, :])

        _axpy(v, r, -b[i - 1])

        a[i] = _np.vdot(Q[i, :], r).real
        _axpy(Q[i, :], r, -a[i])

        b[i] = _np.linalg.norm(r)
        if b[i] < eps:
            m = i
            break

    if full_ortho:
        q, _ = _np.linalg.qr(Q[:m].T)

        Q[:m, :] = q.T[...]

        h = _np.zeros((m, m), dtype=a.dtype)

        for i in range(m):
            if use_out:
                A.dot(Q[i, :], out=r)  # call if operator supports 'out' argument
            else:
                r[:] = A.dot(Q[i, :])

            _np.conj(r, out=r)
            h[i, i:] = _np.dot(Q[i:m, :], r).real

        E, V = _np.linalg.eigh(h, UPLO="U")

    else:
        E, V = eigh_tridiagonal(a[:m], b[: m - 1])

    return E, V, Q[:m]


def lanczos_iter(A, v0, m, return_vec_iter=True, copy_v0=True, copy_A=False, eps=None):
    """Creates generator for Lanczos basis; diagonalizes Krylov subspace in Lanczos basis.

    Given a hermitian matrix `A` of size :math:`n\\times n` and an integer `m`, the Lanczos algorithm computes

    * an :math:`n\\times m` matrix  :math:`Q`, and
    * a real symmetric tridiagonal matrix :math:`T=Q^\\dagger A Q` of size :math:`m\\times m`. The matrix :math:`T` can be represented via its eigendecomposition `(E,V)`: :math:`T=V\\mathrm{diag}(E)V^T`.

    This function computes the triple :math:`(E,V,Q^T)`.

    :red:`NOTE:` This function returns :math:`Q^T;\\,Q^T` is (in general) different from :math:`Q^\\dagger`.


    Parameters
    ----------
    A : LinearOperator, hamiltonian, numpy.ndarray, etc. with a 'dot' method and a 'dtype' method.
            Python object representing a linear map to compute the Lanczos approximation to the largest eigenvalues/vectors of. Must contain a dot-product method, used as `A.dot(v)` and a dtype method, used as `A.dtype`, e.g. `hamiltonian`, `quantum_operator`, `quantum_LinearOperator`, sparse or dense matrix.
    v0 : array_like, (n,)
            initial vector to start the Lanczos algorithm from.
    m : int
            Number of Lanczos vectors (size of the Krylov subspace)
    return_vec_iter : bool, optional
            Toggles whether or not to return the Lanczos basis iterator.
    copy_v0 : bool, optional
            Whether or not to produce of copy of initial vector `v0`.
    copy_A : bool, optional
            Whether or not to produce of copy of linear operator `A`.
    eps : float, optional
            Used to cutoff lanczos iteration when off diagonal matrix elements of `T` drops below this value.

    Returns
    -------
    tuple(E,V,Q_T)
            * E : (m,) numpy.ndarray: eigenvalues of Krylov subspace tridiagonal matrix :math:`T`.
            * V : (m,m) numpy.ndarray: eigenvectors of Krylov subspace tridiagonal matrix :math:`T`.
            * Q_T : generator that yields the `m` lanczos basis vectors on the fly, produces the same result as: :code:`iter(Q_T[:])` where `Q_T` is the array generated by `lanczos_full`

    Notes
    -----
    * this function is useful to minimize any memory requirements in the calculation of the Lanczos basis.
    * the generator of the lanczos basis performs the calculation 'on the fly'. This means that the lanczos iteration is repeated every time this generator is looped over.
    * this generator `Q_T` can be reused as many times as needed, this relies on the data in both `v0` and `A` remaining unchanged during runtime. If this cannot be guaranteed then it is safer to set both `copy_v0` and `copy_A` to be true.
    * `V` is always real-valued, since :math:`T` is real and symmetric.


    Examples
    --------

    >>> E, V, Q_T_iterator = lanczos_iter(H,v0,20)

    """

    v0 = _np.asanyarray(v0)
    n = v0.size
    dtype = _np.result_type(A.dtype, v0.dtype)

    if copy_v0 and return_vec_iter:
        v0 = v0.copy()

    if copy_A and return_vec_iter:
        A = deepcopy(A)

    if v0.ndim != 1:
        raise ValueError("expecting array with ndim=1 for initial Lanczos vector.")

    if m >= n:
        raise ValueError(
            "Requested size of Lanczos basis must be smaller then size of original space (e.g. m < n)."
        )

    q = v0.astype(dtype, copy=True)
    v = _np.zeros_like(v0, dtype=dtype)
    r = _np.zeros_like(v0, dtype=dtype)

    b = _np.zeros((m,), dtype=q.real.dtype)
    a = _np.zeros((m,), dtype=q.real.dtype)

    if eps is None:
        eps = _np.finfo(dtype).eps

    q_norm = _np.linalg.norm(q)

    if _np.abs(q_norm - 1.0) > eps:
        _np.divide(q, q_norm, out=q)

    try:
        A.dot(q, out=r)  # call if operator supports 'out' argument
        use_out = True
    except TypeError:
        r[:] = A.dot(q)
        use_out = False

    a[0] = _np.vdot(q, r).real
    _axpy(q, r, -a[0])
    b[0] = _np.linalg.norm(r)

    i = 0
    for i in range(1, m, 1):
        v[:] = q[:]

        _np.divide(r, b[i - 1], out=q)

        if use_out:
            A.dot(q, out=r)  # call if operator supports 'out' argument
        else:
            r[:] = A.dot(q)

        _axpy(v, r, -b[i - 1])
        a[i] = _np.vdot(q, r).real
        _axpy(q, r, -a[i])

        b[i] = _np.linalg.norm(r)
        if b[i] < eps:
            m = i
            break

    E, V = eigh_tridiagonal(a[:m], b[: m - 1])

    if return_vec_iter:
        return E, V, _lanczos_vec_iter(A, v0, a[:m], b[: m - 1])
    else:
        return E, V


def _get_first_lv_iter(r, Q_iter):
    yield r
    for Q in Q_iter:
        yield Q


def _get_first_lv(Q_iter):
    r = next(Q_iter)
    return r, _get_first_lv_iter(r, Q_iter)


# I suggest the name `lv_average()` or `lv_linearcomb` or `linear_combine_Q()` instead of `lin_comb_Q()`
def lin_comb_Q_T(coeff, Q_T, out=None):
    """Computes a linear combination of the Lanczos basis vectors:

    .. math::
            v_j = \\sum_{i=1}^{m} c_i \\left(Q^T\\right)_{ij}


    Parameters
    ----------
    coeff : (m,) array_like
            list of coefficients to compute the linear combination of Lanczos basis vectors with.
    Q_T : (m,n) numpy.ndarray, generator
            Lanczos basis vectors or a generator for the Lanczos basis.
    out : (n,) numpy.ndarray, optional
            Array to store the result in.

    Returns
    -------
    (n,) numpy.ndarray
            Linear combination :math:`v` of Lanczos basis vectors.

    Examples
    --------

    >>> v = lin_comb_Q(coeff,Q_T)

    """

    coeff = _np.asanyarray(coeff)

    if isinstance(Q_T, _np.ndarray):
        Q_iter = iter(Q_T[:])
    else:
        Q_iter = iter(Q_T)

    q = next(Q_iter)

    dtype = _np.result_type(q.dtype, coeff.dtype)

    if out is not None:
        if out.shape != q.shape:
            raise ValueError("'out' must have same shape as a Lanczos vector.")
        if out.dtype != dtype:
            raise ValueError(
                "argument 'out' has dtype {}, expecting dtype {}".format(
                    out.dtype, dtype
                )
            )
        if not out.flags["CARRAY"]:
            raise ValueError("argument 'out' must be C-contiguous and writable.")
    else:
        out = _np.zeros(q.shape, dtype=dtype)

    n = q.size

    _np.multiply(q, coeff[0], out=out)
    for weight, q in zip(coeff[1:], Q_iter):
        _axpy(q, out, weight)

    return out
