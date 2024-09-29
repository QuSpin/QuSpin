# -*- coding: utf-8 -*-


# need linear algebra packages
import scipy.sparse as _sp
import numpy as _np


from parallel_sparse_tools.matvec.matvec_core import matvec, get_matvec_function

from quspin.basis import get_basis_type

import warnings

__all__ = [
    "project_op",
    "KL_div",
    "mean_level_spacing",
    "matvec",
    "get_matvec_function",
    "ints_to_array",
    "array_to_ints",
]


def project_op(Obs, proj, dtype=_np.complex128):
    """Projects observable onto symmetry-reduced subspace.

    This function takes an observable `Obs` and a reduced basis or a projector `proj`, and projects `Obs`
    onto that reduced basis.

    Examples
    --------

    The following example shows how to project an operator :math:`H_1=\\sum_j hS^x_j + g S^z_j` from the
    symmetry-reduced basis to the full basis.

    .. literalinclude:: ../../doc_examples/project_op-example.py
            :linenos:
            :language: python
            :lines: 7-

    Parameters
    ----------
    Obs : :obj:
            Operator to be projected, either a `numpy.ndarray` or a `hamiltonian` object.
    proj : :obj:
            Either one of the following:

            * `basis` object with the basis of the Hilbert space after the projection.
            * numpy.ndarray: a matrix which contains the projector.

            Projectors can be calculated conveniently using the function method `basis.get_proj()`.
    dtype : type, optional
            Data type of output. Default is `numpy.complex128`.

    Returns
    -------
    dict
            Dictionary with keys

            * "Proj_Obs": projected observable `Obs`.

    """

    # needed for isinstance only
    from quspin.operators import ishamiltonian
    from quspin.basis import isbasis

    variables = ["Proj_Obs"]

    if isbasis(proj):
        proj = proj.get_proj(dtype)
    elif (proj.__class__ not in [_np.ndarray, _np.matrix]) and (not _sp.issparse(proj)):
        raise ValueError(
            "Expecting either matrix/array or basis object for proj argument."
        )

    if ishamiltonian(Obs):

        if Obs.Ns != proj.shape[0]:
            if Obs.Ns != proj.shape[1]:
                raise ValueError(
                    "Dimension mismatch Obs:{0} proj{1}".format(
                        Obs.get_shape, proj.shape
                    )
                )
            else:
                # projecting from a smaller to larger H-space
                proj_down = False
        else:
            # projecting from larger to smaller H-space
            proj_down = True

        if proj_down:
            Proj_Obs = Obs.project_to(proj)
        else:
            Proj_Obs = Obs.project_to(proj.T.conj())

    else:

        if Obs.ndim != 2:
            raise ValueError("Expecting Obs to be a 2 dimensional array.")

        if Obs.shape[0] != Obs.shape[1]:
            raise ValueError("Expecting Obs to be a square array.")

        if Obs.shape[1] != proj.shape[0]:
            if Obs.shape[0] != proj.shape[1]:
                raise ValueError(
                    "Dimension mismatch Obs:{0} proj{1}".format(Obs.shape, proj.shape)
                )
            else:
                proj_down = False
        else:
            proj_down = True

        if proj_down:
            Proj_Obs = proj.T.conj().dot(Obs.dot(proj))
        else:
            Proj_Obs = proj.dot(Obs.dot(proj.T.conj()))

    # define dictionary with outputs
    return_dict = {}
    for i in range(len(variables)):
        return_dict[variables[i]] = locals()[variables[i]]

    return return_dict


def KL_div(p1, p2):
    """Calculates Kullback-Leibler divergence of two discrete probability distributions.

    .. math::
            \\mathrm{KL}(p_1||p_2) = \\sum_n p_1(n)\\log\\frac{p_1(n)}{p_2(n)}

    Parameters
    ----------
    p1 : numpy.ndarray
            Dscrete probability distribution.
    p2 : numpy.ndarray
            Discrete probability distribution.

    Returns
    -------
    numpy.ndarray
            Kullback-Leibler divergence of `p1` and `p2`.

    """
    p1 = _np.asarray(p1)
    p2 = _np.asarray(p2)

    if len(p1) != len(p2):
        raise TypeError(
            "Expecting the probability distributions 'p1' and 'p2' to have same size!"
        )
    if p1.ndim != 1 or p2.ndim != 1:
        raise TypeError(
            "Expecting the probability distributions 'p1' and 'p2' to have linear dimension!"
        )

    if _np.any(p1 <= 0.0) or _np.any(p2 <= 0.0):
        raise TypeError(
            "Expecting all entries of the probability distributions 'p1' and 'p2' to be non-negative!"
        )

    if abs(sum(p1) - 1.0) > 1e-13:
        raise ValueError("Expecting 'p1' to be normalised!")

    if abs(sum(p2) - 1.0) > 1e-13:
        raise ValueError("Expecting 'p2' to be normalised!")

    if _np.any(p1 == 0.0):

        inds = _np.where(p1 == 0)

        p1 = _np.delete(p1, inds)
        p2 = _np.delete(p2, inds)

    return _np.multiply(p1, _np.log(_np.divide(p1, p2))).sum()


def mean_level_spacing(E, verbose=True):
    """Calculates the mean-level spacing of an energy spectrum.

    See mean level spacing, :math:`\\langle\\tilde r_\mathrm{W}\\rangle`, in
    `arXiv:1212.5611 <https://arxiv.org/pdf/1212.5611.pdf>`_ for more details.

    For Wigner-Dyson statistics, we have :math:`\\langle\\tilde r_\mathrm{W}\\rangle\\approx 0.53`, while
    for Poisson statistics: :math:`\\langle\\tilde r_\mathrm{W}\\rangle\\approx 0.38`.

    Examples
    --------

    The following example shows how to calculate the mean level spacing :math:`\\langle\\tilde r_\mathrm{W}\\rangle` for the
    spectrum of the ergodic Hamiltonian :math:`H_1=\\sum_jJ S^z_{j+1}S^z + hS^x_j + g S^z_j`.

    .. literalinclude:: ../../doc_examples/mean_level_spacing-example.py
            :linenos:
            :language: python
            :lines: 7-

    Parameters
    ----------
    E : numpy.ndarray
                    Ordered list of ascending, NONdegenerate energies. If `E` contains a repeating value, the function returns `nan`.
    verbose : bool, optional
            Toggles warning message about degeneracies of the spectrum `E`.

    Returns
    -------
    float
            mean-level spacing.
    nan
            if spectrum `E` has degeneracies.

    """

    if not isinstance(E, _np.ndarray):
        E = _np.asarray(E)

    if _np.any(_np.sort(E) != E):
        raise TypeError(
            "Expecting a sorted list of ascending, nondegenerate eigenenergies 'E'."
        )

    # check for degeneracies
    if len(_np.unique(E)) != len(E):
        if verbose:
            warnings.warn("Degeneracies found in spectrum 'E'!")
        return _np.nan
    else:
        # compute consecutive E-differences
        sn = _np.diff(E)

        # calculate the ratios of consecutive spacings
        aux = _np.zeros((len(E) - 1, 2), dtype=_np.float64)

        aux[:, 0] = sn
        aux[:, 1] = _np.roll(sn, -1)

        return _np.mean(_np.divide(aux.min(1), aux.max(1))[0:-1])


def ints_to_array(basis_ints, N=None):
    """Converts QuSpin basis type integers to a state array with binary elements.

        This function takes an array of batched QuSpin basis type integers and converts it
        to a batched state array with 0/1 elements representing spin-down/up or 0/1 occupation.

        Notes
        -----
        Conversion to higher spins or larger occupation numbers is not yet implemented.

        Examples
        --------

        .. literalinclude:: ../../doc_examples/array_ints_conversion-example.py
                :linenos:
                :language: python
                :lines: 7-

        Parameters
        ----------
        basis_ints: np.ndarray(int)
        batched integers to be converted

        N: int, optional
                number of sites (doubled for spinful fermions), default to be the biggest size
        represented by `basis_ints.dtype`.

        Returns
        -------
        state_array: np.ndarray(np.uint8)
        batched state array with binary entries
    """

    basis_ints = _np.asarray(basis_ints, order="C").reshape(-1, 1)

    if not basis_ints.dtype.isbuiltin:
        basis_ints = basis_ints.view(_np.uint64)
        valid = basis_ints[:, -2:-1]
        basis_ints = _np.ascontiguousarray(basis_ints[:, :-2])
        idx = _np.arange(basis_ints.shape[1], dtype=_np.uint8)
        idx = _np.broadcast_to(idx, basis_ints.shape)
        basis_ints[idx >= valid] = 0
    basis_ints = basis_ints.view(_np.uint8)
    state_array = _np.unpackbits(basis_ints, axis=1, count=N, bitorder="little")
    return state_array[:, ::-1]


def array_to_ints(state_array, dtype=None):
    """Converts a state array with binary elements to QuSpin basis type integers.

        This function takes a batched state array with 0/1 elements and converts it
        to an array of batched QuSpin basis type integers.
        
        Notes
        -----
        Conversion of higher spins or larger occupation numbers is not yet implemented.

        Examples
        --------

        .. literalinclude:: ../../doc_examples/array_ints_conversion-example.py
                :linenos:
                :language: python
                :lines: 7-

        Parameters
        ----------
        state_array: np.ndarray(np.uint8)
        batched state array to be converted

        dtype: dtype, optional
                data type used for the basis integers, default to be the type expressing the
        state with smallest size

        Returns
        -------
        basis_ints: np.ndarray(int)
        batched basis integers
    """

    state_array = _np.atleast_2d(state_array)
    nbits = state_array.shape[1]
    if dtype is None:
        dtype = get_basis_type(nbits, None, 2)
    dtype = _np.dtype(dtype)
    nfull = dtype.itemsize * 8

    pads = _np.zeros((state_array.shape[0], nfull - nbits), state_array.dtype)
    state_array = _np.concatenate([pads, state_array], axis=1)
    state_array = _np.ascontiguousarray(state_array[:, ::-1], dtype=_np.uint8)
    basis_ints = _np.packbits(state_array, axis=1, bitorder="little")
    if not dtype.isbuiltin:
        valid = _np.array([(nbits - 1) // 64 + 1], _np.uint64).view(_np.uint8)
        basis_ints[:, -16:-8] = valid
    basis_ints = basis_ints.view(dtype)
    return basis_ints
