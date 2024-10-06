from quspin.operators import hamiltonian_core, quantum_operator_core

# need linear algebra packages
import scipy
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

from scipy.sparse.linalg import expm_multiply as _expm_multiply

from copy import deepcopy as _deepcopy  # recursively copies all data into new object
from copy import copy as _shallowcopy

from six import iteritems

__all__ = ["exp_op", "isexp_op"]


class exp_op(object):
    """Constructs matrix exponentials of quantum operators.

    The `exp_op` class does not calculate the actual matrix exponential but instead computes the action of the
    matrix exponential through its Taylor series. This is slower but for sparse arrays it is more memory efficient.
    All of the functions make use of the `expm_multiply` function in Scipy's sparse library.

    This class also allows the option to specify a grid of points on a line in the complex plane via the
    optional arguments. If this is specified, then an array `grid` is created via the function
    `numpy.linspace`, and the exponential is evaluated for all points on te grid: `exp(a*grid[i]*O)`.

    Notes
    -----
    To calculate the matrix exponential itself, use the function method `exp_op.get_mat()`.

    For a faster computations, look up the `tools.expm_multiply_parallel` function.

    Examples
    --------

    The Example below shows how to compute the time-evolvution of a state under a constant Hamiltonian.
    This is done using the matrix exponential to define the evolution operator and then applying it directly
    onto the initial state.

    .. literalinclude:: ../../doc_examples/exp_op-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(
        self, O, a=1.0, start=None, stop=None, num=None, endpoint=None, iterate=False
    ):
        """Initialises the `exp_op` object (matrix exponential of the operator `O`).

        Parameters
        ----------
        O : obj
                `numpy.ndarray`, `scipy.spmatrix`, `hamiltonian`, `quantum_operator` object: the operator to compute the matrix exponential of.
        a : `numpy.dtype`, optional
                Prefactor to go in front of the operator in the exponential: `exp(a*O)`. Can be a complex number.
                Default is `a = 1.0`.
        start : scalar, optional
                Specifies the starting point for a grid of points to evaluate the matrix exponential at.
        stop : scalar, optional
                Specifies the end point of for a grid of points to evaluate the matrix exponential at.
        num : int, optional
                Specifies the number of grid points between start and stop. Default is `num = 50`.
        endpoint : bool, optional
                Wehether or not the value `stop` is included in the set of grid points. Note that this changes
                the grid step size.
        iterate : bool, optional
                If set to `True` class methods return generators which will iterate over the `grid` points.

                If set to `False`, a list of all the evaluated points is produced. This is more memory efficient
                but at the sacrifice of speed.

                Default is `False`.

        """
        if _np.array(a).ndim > 0:
            raise TypeError("expecting scalar argument for a")

        self._a = a

        self._start = start
        self._stop = stop
        self._num = num
        self._endpoint = endpoint
        self._iterate = iterate
        if self._iterate:
            if self._start is None and self._stop is None:
                raise ValueError(
                    "'iterate' can only be True with time discretization. must specify 'start' and 'stop' points."
                )

            if num is not None:
                if type(num) is not int:
                    raise ValueError("expecting integer for 'num'.")
            else:
                num = 50
                self._num = num

            if endpoint is not None:
                if type(endpoint) is not bool:
                    raise ValueError("expecting bool for 'endpoint'.")
            else:
                endpoint = True
                self._endpoint = endpoint

            self._grid, self._step = _np.linspace(
                start, stop, num=num, endpoint=endpoint, retstep=True
            )
        else:
            if self._start is None and self._stop is None:
                if self._num != None:
                    raise ValueError("unexpected argument 'num'.")

                if self._endpoint != None:
                    raise ValueError("unexpected argument 'endpoint'.")

                self._grid = None
                self._step = None
            else:

                if not (_np.isscalar(start) and _np.isscalar(stop)):
                    raise ValueError("expecting scalar values for 'start' and 'stop'")

                if not (_np.isreal(start) and _np.isreal(stop)):
                    raise ValueError("expecting real values for 'start' and 'stop'")

                if num is not None:
                    if type(num) is not int:
                        raise ValueError("expecting integer for 'num'.")
                else:
                    num = 50
                    self._num = num

                if endpoint is not None:
                    if type(endpoint) is not bool:
                        raise ValueError("expecting bool for 'endpoint'.")
                else:
                    endpoint = True
                    self._endpoint = endpoint

                self._grid, self._step = _np.linspace(
                    start, stop, num=num, endpoint=endpoint, retstep=True
                )

        if hamiltonian_core.ishamiltonian(O):
            self._O = O
        elif quantum_operator_core.isquantum_operator(O):
            self._O = O
        else:
            if _sp.issparse(O) or O.__class__ in [_np.ndarray, _np.matrix]:
                self._O = hamiltonian_core.hamiltonian([O], [], dtype=O.dtype)
            else:
                O = _np.asanyarray(O)
                self._O = hamiltonian_core.hamiltonian([O], [], dtype=O.dtype)

        self._ndim = 2

    @property
    def ndim(self):
        """int: number of dimensions, always equal to 2."""
        return self._ndim

    @property
    def H(self):
        """:obj:`exp_op`: transposes and conjugates the matrix exponential."""
        return self.getH(copy=False)

    @property
    def T(self):
        """:obj:`exp_op`: transposes the matrix exponential."""
        return self.transpose(copy=False)

    @property
    def O(self):
        """obj: Returns the operator to be exponentiated."""
        return self._O

    @property
    def a(self):
        """`numpy.dtype`: constant (c-number) multiplying the operator to be exponentiated, `exp(a*O)`."""
        return self._a

    @property
    def get_shape(self):
        """tuple: shape of the `hamiltonian` object, always equal to `(Ns,Ns)`."""
        return self.O.get_shape

    @property
    def Ns(self):
        """int: number of states in the (symmetry-reduced) Hilbert space spanned by `basis`."""
        return self.O.Ns

    @property
    def grid(self):
        """numpy.array: grid containing equidistant points to evaluate the matrix exponential at."""
        return self._grid

    @property
    def step(self):
        """float: step size between equidistant grid points."""
        return self._step

    @property
    def iterate(self):
        """bool: shows if iterate option is on/off."""
        return self._iterate

    def transpose(self, copy=False):
        """Transposes `exp_op` operator.

        Notes
        -----
        This function does NOT conjugate the exponentiated operator.

        Returns
        -------
        :obj:`exp_op`
                :math:`\\exp(a\\mathcal{O})_{ij}\\mapsto \\exp(a\\mathcal{O})_{ji}`

        Examples
        --------

        >>> expO_tran = expO.transpose()

        """
        if copy:
            return self.copy().transpose(copy=False)
        else:
            self._O = self._O.transpose()
            return self

    def conj(self):
        """Conjugates `exp_op` operator.

        Notes
        -----
        This function does NOT transpose the exponentiated operator.

        Returns
        -------
        :obj:`exo_op`
                :math:`\\left[\\exp(a\\mathcal{O})_{ij}\\right]\\mapsto \\left[\\exp(a\\mathcal{O})_{ij}\\right]^*`

        Examples
        --------

        >>> expO_conj = expO.conj()

        """
        self._O = self._O.conj()
        self._a = self._a.conjugate()
        return self

    def getH(self, copy=False):
        """Calculates hermitian conjugate of `exp_op` operator.

        Parameters
        ----------
        copy : bool, optional
                Whether to return a deep copy of the original object. Default is `copy = False`.

        Returns
        -------
        :obj:`exp_op`
                :math:`\\exp(a\\mathcal{O})_{ij}\\mapsto \\exp(a\\mathcal{O})_{ij}^*`

        Examples
        --------

        >>> expO_herm = expO.getH()

        """
        if copy:
            return self.copy().getH(copy=False)
        else:
            self._O = self._O.getH(copy=False)
            self._a = self._a.conjugate()
            return self

    def copy(self):
        """Returns a deep copy of `exp_op` object."""
        return _deepcopy(self)

    def set_a(self, new_a):
        """Resets attribute `a` to multiply the operator in `exp(a*O)`.

        Parameters
        ----------
        new_a : `numpy.dtype`
                New value for `a`.

        Examples
        --------
        >>> expO = exp_op(O,a=1.0)
        >>> print(expO.a)
        >>> expO.set_a(2.0)
        >>> print(expO.a)

        """
        if not _np.isscalar(new_a):
            raise ValueError("'a' must be set to scalar value.")
        self._a = _np.complex128(new_a)

    def set_grid(self, start, stop, num=None, endpoint=None):
        """Resets attribute `grid` to evaluate the operator for every `i` in `exp(a*O*grid[i])`.

        Parameters
        ----------
        start : scalar, optional
                Specifies the new starting point for a grid of points to evaluate the matrix exponential at.
        stop : scalar, optional
                Specifies the new end point of for a grid of points to evaluate the matrix exponential at.
        num : int, optional
                Specifies the new number of grid points between start and stop. Default is `num = 50`.
        endpoint : bool, optional
                Wehether or not the value `stop` is included in the set of grid points. Note that this changes
                the grid step size.

        Examples
        --------
        >>> expO = exp_op(O,start=0.0,stop=6.0,num=601,endpoint=True)
        >>> print(expO.grid)
        >>> expO.set_grid(start=2.0,stop=4.0,num=200,endpoint=False)
        >>> print(expO.grid)

        """

        if not (_np.isscalar(start) and _np.isscalar(stop)):
            raise ValueError("expecting scalar values for 'start' and 'stop'")

        if not (_np.isreal(start) and _np.isreal(stop)):
            raise ValueError("expecting real values for 'start' and 'stop'")

        if type(num) is not None:
            if type(num) is not int:
                raise ValueError("expecting integer for 'num'.")

        if type(endpoint) is not None:
            if type(endpoint) is not bool:
                raise ValueError("expecting bool for 'endpoint'.")

        self._start = start
        self._stop = stop
        self._num = num
        self._endpoint = endpoint
        self._grid, self._step = _np.linspace(
            start, stop, num=num, endpoint=endpoint, retstep=True
        )

    def unset_grid(self):
        """Resets grid parameters to their default values.

        Examples
        --------
        >>> expO = exp_op(O,start=0.0,stop=6.0,num=601,endpoint=True)
        >>> print(expO.grid)
        >>> expO.unset_grid()
        >>> print(expO.grid)

        """
        self._iterate = False
        self._start = None
        self._stop = None
        self._num = None
        self._endpoint = None
        self._grid, self._step = None, None

    def set_iterate(self, Value):
        """Resets `iterate` attribute.

        Parameters
        ----------
        Value : bool
                New value for `iterate` attribute.

        Examples
        --------
        >>> expO = exp_op(O,iterate=True)
        >>> print(expO.iterate)
        >>> expO.set_a(False)
        >>> print(expO.iterate)

        """

        if type(Value) is not bool:
            raise ValueError("iterate option must be true or false.")

        if Value:
            if (self._grid, self._step) == (None, None):
                raise ValueError("grid must be set in order to set iterate to be True.")

        self._iterate = Value

    def get_mat(self, dense=False, **call_kwargs):
        """Calculates matrix corresponding to matrix exponential object: `exp(a*O)`.

        Parameters
        ----------
        dense : bool
                Whether or not to return a dense or a sparse array. Detault is `dense = False`.
        call_kwargs : obj, optional
                extra keyword arguments which include:
                        **time** (*scalar*) - if the operator `O` to be exponentiated is a `hamiltonian` object.
                        **pars** (*dict*) - if the operator `O` to be exponentiated is a `quantum_operator` object.

        Returns
        -------
        obj
                Can be either one of

                * `numpy.ndarray`: dense array if `dense = True`.
                * `scipy.sparse.csc`: sparse array if `dense = False`.

        Examples
        --------
        >>> expO = exp_op(O)
        >>> print(expO.get_mat(time=0.0))
        >>> print(expO.get_mat(time=0.0,dense=True))

        """
        if self.O.is_dense or dense:
            return _la.expm(self._a * self.O.toarray(**call_kwargs))
        else:
            return _sp.linalg.expm(self._a * self.O.tocsc(**call_kwargs))

    def dot(self, other, shift=None, **call_kwargs):
        """Left-multiply operator by matrix exponential.

        Let the matrix exponential object be :math:`\\exp(\\mathcal{O})` and let the operator be :math:`A`.
        Then this funcion implements:

        .. math::
                \\exp(\\mathcal{O}) A

        Parameters
        ----------
        other : obj
                The operator :math:`A` which multiplies from the right the matrix exponential :math:`\\exp(\\mathcal{O})`.
        shift : scalar
                Shifts operator to be exponentiated by a constant `shift` times te identity matrix: :math:`\\exp(\\mathcal{O} - \\mathrm{shift}\\times\\mathrm{Id})`.
        call_kwargs : obj, optional
                extra keyword arguments which include:
                        **time** (*scalar*) - if the operator `O` to be exponentiated is a `hamiltonian` object.
                        **pars** (*dict*) - if the operator `O` to be exponentiated is a `quantum_operator` object.

        Returns
        -------
        obj
                matrix exponential multiplied by `other` from the right.

        Examples
        --------
        >>> expO = exp_op(O)
        >>> A = exp_op(O,a=2j).get_mat()
        >>> print(expO.dot(A))

        """

        is_sp = False
        is_ham = False

        if hamiltonian_core.ishamiltonian(other):
            shape = other._shape
            is_ham = True
        elif _sp.issparse(other):
            shape = other.shape
            is_sp = True
        elif other.__class__ in [_np.matrix, _np.ndarray]:
            shape = other.shape
        else:
            other = _np.asanyarray(other)
            shape = other.shape

        if other.ndim not in [1, 2]:
            raise ValueError("Expecting a 1 or 2 dimensional array for 'other'")

        if shape[0] != self.get_shape[1]:
            raise ValueError(
                "Dimension mismatch between expO: {0} and other: {1}".format(
                    self._O.get_shape, other.shape
                )
            )

        if shift is not None:
            M = self._a * (
                self.O(**call_kwargs)
                + shift * _sp.identity(self.Ns, dtype=self.O.dtype)
            )
        else:
            M = self._a * self.O(**call_kwargs)

        if self._iterate:
            if is_ham:
                return _hamiltonian_iter_dot(M, other, self._step, self._grid)
            else:
                return _iter_dot(M, other, self._step, self._grid)

        else:
            if self._grid is None and self._step is None:
                if is_ham:
                    return _hamiltonian_dot(M, other)
                else:
                    return _expm_multiply(M, other)
            else:
                if is_sp:
                    mats = _iter_dot(M, other, self._step, self._grid)
                    return _np.array([mat for mat in mats])
                elif is_ham:
                    mats = _hamiltonian_iter_dot(M, other, self._step, self._grid)
                    return _np.array([mat for mat in mats])
                else:
                    ver = [int(v) for v in scipy.__version__.split(".")]

                    if (
                        _np.iscomplexobj(_np.float32(1.0).astype(M.dtype))
                        and ver[1] < 19
                    ):
                        mats = _iter_dot(M, other, self._step, self._grid)
                        return _np.array([mat for mat in mats]).T
                    else:
                        return _expm_multiply(
                            M,
                            other,
                            start=self._start,
                            stop=self._stop,
                            num=self._num,
                            endpoint=self._endpoint,
                        ).T

    def rdot(self, other, shift=None, **call_kwargs):
        """Right-multiply an operator by matrix exponential.

        Let the matrix exponential object be :math:`\\exp(\\mathcal{O})` and let the operator be :math:`A`.
        Then this funcion implements:

        .. math::
                A \\exp(\\mathcal{O})

        Notes
        -----
        For `hamiltonian` objects `A`, this function is the same as `A.dot(expO)`.

        Parameters
        ----------
        other : obj
                The operator :math:`A` which multiplies from the left the matrix exponential :math:`\\exp(\\mathcal{O})`.
        shift : scalar
                Shifts operator to be exponentiated by a constant `shift` times the identity matrix: :math:`\\exp(\\mathcal{O} - \\mathrm{shift}\\times\\mathrm{Id})`.
        call_kwargs : obj, optional
                extra keyword arguments which include:
                        **time** (*scalar*) - if the operator `O` to be exponentiated is a `hamiltonian` object.
                        **pars** (*dict*) - if the operator `O` to be exponentiated is a `quantum_operator` object.

        Returns
        -------
        obj
                matrix exponential multiplied by `other` from the left.

        Examples
        --------
        >>> expO = exp_op(O)
        >>> A = exp_op(O,a=2j).get_mat()
        >>> print(expO.rdot(A))
        >>> print(A.dot(expO))

        """

        is_sp = False
        is_ham = False

        if hamiltonian_core.ishamiltonian(other):
            shape = other._shape
            is_ham = True
        elif _sp.issparse(other):
            shape = other.shape
            is_sp = True
        elif other.__class__ in [_np.matrix, _np.ndarray]:
            shape = other.shape
        else:
            other = _np.asanyarray(other)
            shape = other.shape

        if other.ndim not in [1, 2]:
            raise ValueError("Expecting a 1 or 2 dimensional array for 'other'")

        if other.ndim == 2:
            if shape[1] != self.get_shape[0]:
                raise ValueError(
                    "Dimension mismatch between expO: {0} and other: {1}".format(
                        self._O.get_shape, other.shape
                    )
                )
        elif shape[0] != self.get_shape[0]:
            raise ValueError(
                "Dimension mismatch between expO: {0} and other: {1}".format(
                    self._O.get_shape, other.shape
                )
            )

        if shift is not None:
            M = (
                self._a
                * (
                    self.O(**call_kwargs)
                    + shift * _sp.identity(self.Ns, dtype=self.O.dtype)
                )
            ).T
        else:
            M = (self._a * self.O(**call_kwargs)).T

        if self._iterate:
            if is_ham:
                return _hamiltonian_iter_rdot(M, other.T, self._step, self._grid)
            else:
                return _iter_rdot(M, other.T, self._step, self._grid)
        else:
            if self._grid is None and self._step is None:

                if is_ham:
                    return _hamiltonian_rdot(M, other.T).T
                else:
                    return _expm_multiply(M, other.T).T
            else:
                if is_sp:
                    mats = _iter_rdot(M, other.T, self._step, self._grid)
                    return _np.array([mat for mat in mats])
                elif is_ham:
                    mats = _hamiltonian_iter_rdot(M, other.T, self._step, self._grid)
                    return _np.array([mat for mat in mats])
                else:
                    ver = [int(v) for v in scipy.__version__.split(".")]
                    if (
                        _np.iscomplexobj(_np.float32(1.0).astype(M.dtype))
                        and ver[1] < 19
                    ):
                        mats = _iter_rdot(M, other.T, self._step, self._grid)
                        return _np.array([mat for mat in mats])
                    else:
                        if other.ndim > 1:
                            return _expm_multiply(
                                M,
                                other.T,
                                start=self._start,
                                stop=self._stop,
                                num=self._num,
                                endpoint=self._endpoint,
                            ).transpose(0, 2, 1)
                        else:
                            return _expm_multiply(
                                M,
                                other.T,
                                start=self._start,
                                stop=self._stop,
                                num=self._num,
                                endpoint=self._endpoint,
                            )

    def sandwich(self, other, shift=None, **call_kwargs):
        """Sandwich operator between matrix exponentials.

        Let the matrix exponential object be :math:`\\exp(\\mathcal{O})` and let the operator to be sandwiched be
        :math:`C`. Then this funcion implements:

        .. math::
                \\exp(\\mathcal{O})^\\dagger C \\exp(\\mathcal{O})

        Notes
        -----
        The matrix exponential to multiply :math:`C` from the left is hermitian conjugated.

        Parameters
        ----------
        other : obj
                The operator :math:`C` to be sandwiched by the matrix exponentials :math:`\\exp(\\mathcal{O})^\\dagger`
                and :math:`\\exp(\\mathcal{O})`.
        shift : scalar
                Shifts operator to be exponentiated by a constant `shift` times the identity matrix: :math:`\\exp(\\mathcal{O} - \\mathrm{shift}\\times\\mathrm{Id})`.
        call_kwargs : obj, optional
                extra keyword arguments which include:
                        **time** (*scalar*) - if the operator `O` to be exponentiated is a `hamiltonian` object.
                        **pars** (*dict*) - if the operator `O` to be exponentiated is a `quantum_operator` object.

        Returns
        -------
        obj
                operator `other` sandwiched between matrix exponential `exp_op` and its hermitian conjugate.

        Examples
        --------
        >>> expO = exp_op(O,a=1j)
        >>> A = exp_op(O.T.conj())
        >>> print(expO.sandwich(A))

        """
        is_ham = False
        if hamiltonian_core.ishamiltonian(other):
            shape = other._shape
            is_ham = True
        elif _sp.issparse(other):
            shape = other.shape
        elif other.__class__ in [_np.matrix, _np.ndarray]:
            shape = other.shape
        else:
            other = _np.asanyarray(other)
            shape = other.shape

        if other.ndim != 2:
            raise ValueError("Expecting a 2 dimensional array for 'other'")

        if shape[0] != shape[1]:
            raise ValueError("Expecting square array for 'other'")

        if shape[0] != self.get_shape[0]:
            raise ValueError(
                "Dimension mismatch between expO: {0} and other: {1}".format(
                    self.get_shape, other.shape
                )
            )

        if shift is not None:
            M = self._a.conjugate() * (
                self.O.T.conj()(**call_kwargs)
                + shift * _sp.identity(self.Ns, dtype=self.O.dtype)
            )
        else:
            M = self._a.conjugate() * self.O.T.conj()(**call_kwargs)

        if self._iterate:

            if is_ham:
                mat_iter = _hamiltonian_iter_sandwich(M, other, self._step, self._grid)
            else:
                mat_iter = _iter_sandwich(M, other, self._step, self._grid)

            return mat_iter
        else:
            if self._grid is None and self._step is None:

                other = self.dot(other, **call_kwargs)
                other = self.T.conj().rdot(other, **call_kwargs)
                return other

            else:
                if is_ham:
                    mat_iter = _hamiltonian_iter_sandwich(
                        M, other, self._step, self._grid
                    )
                    return _np.asarray([mat for mat in mat_iter])
                else:
                    mat_iter = _iter_sandwich(M, other, self._step, self._grid)
                    return _np.asarray([mat for mat in mat_iter]).transpose((1, 2, 0))


### helper functions
def _iter_dot(M, other, step, grid):
    if grid[0] != 0:
        M *= grid[0]
        other = _expm_multiply(M, other)
        M /= grid[0]

    yield other.copy()

    M *= step
    for t in grid[1:]:
        other = _expm_multiply(M, other)
        yield other.copy()


def _iter_rdot(M, other, step, grid):
    if grid[0] != 0:
        M *= grid[0]
        other = _expm_multiply(M, other)
        M /= grid[0]

    yield other.T.copy()

    M *= step
    for t in grid[1:]:
        other = _expm_multiply(M, other)
        yield other.T.copy()


def _iter_sandwich(M, other, step, grid):
    if grid[0] != 0:
        M *= grid[0]
        other = _expm_multiply(M, other)
        other = _expm_multiply(M, other.T.conj()).T.conj()
        M /= grid[0]

    yield other.copy()

    M *= step
    for t in grid[1:]:
        other = _expm_multiply(M, other)
        other = _expm_multiply(M, other.T.conj()).T.conj()

        yield other.copy()


def _hamiltonian_dot(M, other):
    new = other.copy()  # deep=False: not implememnted in hamiltonian_core.copy()
    new._dtype = _np.result_type(M.dtype, new._dtype)
    new._static = _expm_multiply(M, other.static)
    new._dynamic = {
        func: _expm_multiply(M, Hd) for func, Hd in iteritems(other._dynamic)
    }

    return new


def _hamiltonian_iter_dot(M, other, grid, step):
    if grid[0] != 0:
        M *= grid[0]
        other = _hamiltonian_dot(M, other)
        M /= grid[0]

    yield other

    M *= step
    for t in grid[1:]:
        other = _hamiltonian_dot(M, other)
        yield other


def _hamiltonian_rdot(M, other):
    new = other.copy()  # deep=False: not implememnted in hamiltonian_core.copy()
    new._dtype = _np.result_type(M.dtype, new._dtype)
    new._static = _expm_multiply(M, other.static)
    new._dynamic = {
        func: _expm_multiply(M, Hd) for func, Hd in iteritems(other._dynamic)
    }

    return new


def _hamiltonian_iter_rdot(M, other, grid, step):
    if grid[0] != 0:
        M *= grid[0]
        other = _hamiltonian_rdot(M, other)
        M /= grid[0]

    yield other.transpose(copy=True)

    M *= step
    for t in grid[1:]:
        other = _hamiltonian_rdot(M, other)
        yield other.transpose(copy=True)


def _hamiltonian_iter_sandwich(M, other, step, grid):
    if grid[0] != 0:
        M *= grid[0]
        other = _hamiltonian_dot(M, other)
        other = _hamiltonian_dot(M, other.T.conj()).T.conj()
        M /= grid[0]

    yield other.copy()

    M *= step
    for t in grid[1:]:
        other = _hamiltonian_dot(M, other)
        other = _hamiltonian_dot(M, other.T.conj()).T.conj()
        yield other.copy()


def isexp_op(obj):
    """Checks if instance is object of `exp_op` class.

    Parameters
    ----------
    obj :
            Arbitraty python object.

    Returns
    -------
    bool
            Can be either of the following:

            * `True`: `obj` is an instance of `exp_op` class.
            * `False`: `obj` is NOT an instance of`exp_op` class.

    """
    return isinstance(obj, exp_op)
