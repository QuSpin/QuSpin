from scipy.linalg import eigh_tridiagonal
from six import iteritems
import numpy as _np


__all__ = ["LTLM_static_iteration"]


def LTLM_static_iteration(O_dict, E, V, Q_T, beta=0):
    """Calculate iteration for low-temperature Lanczos method.

    Here we give a brief overview of this method based on `arXiv:1111.5931 <https://arxiv.org/abs/1111.5931>`_.

    One would naively think that it would require full diagonalization to calculate thermodynamic expectation values
    for a quantum system as one has to fully diagonalize the Hamiltonian to evaluate:

    .. math::
            \\langle O\\rangle_\\beta = \\frac{1}{Z}Tr\\left(e^{-\\beta H}O\\right)

    with the partition function defined as: :math:`Z=Tr\\left(e^{-\\beta H}\\right)`. The idea behind the
    Low-Temperature Lanczos Method (LTLM) is to use quantum typicality as well as Krylov subspaces to
    simplify this calculation. Typicality states that the trace of an operator can be approximated as an average
    of that same operator with random vectors in the Hilbert-space sampled with the Harr measure. As a corollary, it
    is known that the fluctuations of this average for any finite sample set will converge to 0 as the size of
    the Hilbert space increases. Mathematically this is expressed as:

    .. math::
            \\frac{1}{\\dim\\mathcal{H}}Tr\\left(e^{-\\beta H}O\\right)\\approx \\frac{1}{N_r}\\sum_r\\langle r| e^{-\\beta H}O |r\\rangle

    where :math:`|r\\rangle` is a random state from the Harr measure of hilbert space :math:`\\mathcal{H}` if the
    Hamiltonian. An issue can occur when the temperature goes to zero as the overlap :math:`\\langle r| e^{-\\beta H}O |r\\rangle` will
    be quite small for most states :math:`|r\\rangle`. Hence, this will require more random realizations to converge.
    Fortunately the trace is cyclical and therefore we can make the expression more symmetric:

    .. math::
            \\frac{1}{\\dim\\mathcal{H}}Tr\\left(e^{-\\beta H}O\\right)=\\frac{1}{\\dim\\mathcal{H}}Tr\\left(e^{-\\beta H/2}O e^{-\\beta H/2}\\right)\\approx \\frac{1}{N_r}\\sum_r\\langle r|e^{-\\beta H/2}O e^{-\\beta H/2}|r\\rangle

    Such that the expecation value is exact as :math:`\\beta\\rightarrow\\infty` at the cost of having to calculate
    two matrix exponentials. Next we can approximate the matrix exponential using the Lanczos basis. The idea
    is that the eigenstates from the lanczos basis can effectively be inserted as an identity operator:

    .. math::
            \\frac{1}{\\dim\\mathcal{H}}Tr\\left(e^{-\\beta H/2}O e^{-\\beta H/2}\\right)\\approx \\frac{1}{N_r}\\sum_r\\langle r|e^{-\\beta H/2}O e^{-\\beta H/2}|r\\rangle\\approx \\frac{1}{N_r}\\sum_r\\sum_{i,j=1}^m e^{-\\beta(\\epsilon^{(r)}_i+\\epsilon^{(r)}_j)/2}\\langle r|\\psi^{(r)}_i\\rangle\\langle\\psi^{(r)}_i|O|\\psi^{(r)}_j\\rangle\\langle\\psi^{(r)}_j|r\\rangle = \\frac{1}{N_r}\\sum_r \\langle O\\rangle_r \\equiv \\overline{\\langle O\\rangle_r}

    Now going back to the thermal expecation value, we can use the expression above to calculate :math:`\\frac{1}{Z}Tr\\left(e^{-\\beta H}O\\right)`
    by noting that the partition function is simply the expecation value of the identity operator: :math:`Z=Tr\\left(e^{-\\beta H}I\\right)` and hence
    the thermal expecation value is approximated by:

    .. math::
            \\langle O\\rangle_\\beta \\approx \\frac{\\overline{\\langle O\\rangle_r}}{\\overline{\\langle I\\rangle_r}}


    The idea behind this function is to generate the the expecation value :math:`\\langle O\\rangle_r` and :math:`\\langle I\\rangle_r`
    for a lanczos basis generated from an initial state :math:`|r\\rangle`. Therefore if the user would like to calculate the thermal expecation value all one
    has to do call this function for each lanczos basis generated from a random state :math:`|r\\rangle`.

    Notes
    -----
    * The amount of memory used by this function scales like: :math:`nN_{op}` with :math:`n` being the size of the full Hilbert space and :math:`N_{op}` is the number of input operators.
    * LTLM converges equally well for low and high temperatures however it is more expensive compared to the FTLM and hence we recomend that one should use that method when dealing with high temperatures.
    * One has to be careful as typicality only applies to the trace operation over the entire Hilbert space. Using symmetries is possible, however it requires the user to keep track of the weights in the different sectors.

    Parameters
    ----------
    O_dict : dictionary of Python Objects
            These Objects must have a 'dot' method that calculates a matrix vector product on a numpy.ndarray[:], the effective shape of these objects should be (n,n).
    E : array_like, (m,)
            Eigenvalues for the Krylov projection of some operator.
    V : array_like, (m,m)
            Eigenvectors for the Krylov projection of some operator.
    Q_T : iterator over rows of Q_T
            generator or ndarray that contains the lanczos basis associated with E, and V.
    beta : scalar/array_like, any shape
            Inverse temperature values to evaluate.

    Returns
    -------
    Result_dict: dictionary
            A dictionary storying the results for a single iteration of the LTLM. The results are stored in numpy.ndarrays
            that have the same shape as `beta`. The keys of `Result_dict` are the same as the keys in `O_dict` and the values
            associated with the given key in `Result_dict` are the expectation values for the operator in `O_dict` with the same key.
    I_expt: numpy.ndarray, same shape as `beta`
            The expecation value of the identity operator for each beta.

    Examples
    --------

    >>> beta = numpy.linspace(0,10,101)
    >>> E, V, Q_T = lanczos_full(H,v0,20)
    >>> Res,Id = FTLM_static_iteration(Obs_dict,E,V,Q_T,beta=beta)

    """
    nv = E.size

    beta = _np.atleast_1d(beta)
    p = _np.exp(-E * beta.min()) * _np.abs(V[0, :]).max()
    mask = p < _np.finfo(p.dtype).eps

    if _np.any(mask):
        nv = _np.argmax(mask)

    Ome_dict = {}

    lv_row = iter(Q_T)
    for i, lv_r in enumerate(lv_row):
        if i >= nv:
            break

        lv_col = iter(Q_T)
        Ar_dict = {key: A.dot(lv_r) for key, A in iteritems(O_dict)}

        for j, lv_c in enumerate(lv_col):
            if j >= nv:
                break
            for key, A in iteritems(O_dict):
                if key not in Ome_dict:
                    dtype = _np.result_type(lv_r.dtype, A.dtype)
                    Ome_dict[key] = _np.zeros((nv, nv), dtype=dtype)

                me = _np.vdot(lv_c, Ar_dict[key])
                Ome_dict[key][i, j] = me

        del lv_col
    del lv_row

    p = _np.exp(-_np.outer(E[:nv], _np.atleast_1d(beta) / 2.0))
    V = V[:nv, :nv].copy()

    c = _np.einsum("j,j...->j...", V[0, :], p)

    results_dict = {}
    for key, Ame in iteritems(Ome_dict):
        A_diag = V.T.dot(Ame.dot(V))
        results_dict[key] = _np.squeeze(_np.einsum("j...,l...,jl->...", c, c, A_diag))

    p = _np.exp(-_np.outer(E[:nv], _np.atleast_1d(beta)))
    Id = _np.einsum("j,j...->...", V[0, :] ** 2, p)

    return results_dict, _np.squeeze(Id)
