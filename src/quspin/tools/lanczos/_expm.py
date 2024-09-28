import numpy as _np
from quspin.tools.lanczos._lanczos_utils import lin_comb_Q_T

__all__ = ["expm_lanczos"]


def expm_lanczos(E, V, Q_T, a=1.0, out=None):
    """Calculates action of matrix exponential on vector using Lanczos algorithm.

    The Lanczos decomposition `(E,V,Q)` with initial state `v0` of a hermitian matrix `A` can be used to compute the matrix exponential
    :math:`\\mathrm{exp}(aA)|v_0\\rangle` applied to the quantum state :math:`|v_0\\rangle`, without actually computing the exact matrix exponential:

    Let :math:`A \\approx Q T Q^\\dagger` with :math:`T=V \\mathrm{diag}(E) V^T`. Then, we can compute an approximation to the matrix exponential, applied to a state :math:`|\\psi\\rangle` as follows:

    .. math::
            \\exp(a A)|v_0\\rangle \\approx Q \\exp(a T) Q^\\dagger |v_0\\rangle = Q V \\mathrm{diag}(e^{a E}) V^T Q^\\dagger |v_0\\rangle.

    If we use :math:`|v_0\\rangle` as the (nondegenerate) initial state for the Lanczos algorithm, then :math:`\\sum_{j,k}V^T_{ij}Q^\\dagger_{jk}v_{0,k} = \\sum_{j}V_{ji}\\delta_{0,j} = V_{i,0}` [by construction, :math:`|v_{0}\\rangle` is the zero-th row of :math:`Q` and all the rows are orthonormal], and the expression simplifies further.

    Notes
    -----

    * uses precomputed Lanczos data `(E,V,Q_T)`, see e.g., `lanczos_full` and `lanczos_iter` functions.
    * the initial state `v0` used in `lanczos_full` and `lanczos_iter` is the state the matrix exponential is evaluated on.

    Parameters
    ----------
    E : (m,) np.ndarray
            eigenvalues of Krylov subspace tridiagonal matrix :math:`T`.
    V : (m,m) np.ndarray
            eigenvectors of Krylov subspace tridiagonal matrix :math:`T`.
    Q_T : (m,n) np.ndarray, generator
            Matrix containing the `m` Lanczos vectors in the rows.
    a : scalar, optional
            Scale factor `a` for the generator of the matrix exponential :math:`\\mathrm{exp}(aA)`.
    out : (n,) np.ndarray()
            Array to store the result in.

    Returns
    -------
    (n,) np.ndarray
            Matrix exponential applied to a state, evaluated using the Lanczos method.

    Examples
    --------

    >>> E, V, Q_T = lanczos_iter(H,v0,20)
    >>> expH_v0 = expm_lanczos(E,V,Q_T,a=-1j)

    """
    c = V.dot(_np.exp(a * E) * V[0, :])
    return lin_comb_Q_T(c, Q_T, out=out)
