from quspin.basis.base import basis, MAXPRINT
from quspin.basis.tensor import tensor_basis

import numpy as _np
from scipy import sparse as _sp

from scipy.special import hyp2f1, binom

import warnings

_dtypes = {"f": _np.float32, "d": _np.float64, "F": _np.complex64, "D": _np.complex128}

__all__ = ["photon_basis", "ho_basis", "coherent_state", "photon_Hspace_dim"]


def coherent_state(a, n, dtype=_np.float64):
    """Creates a harmonic oscillator (HO) coherent state.

    Parameters
    ----------
    a: float
            Expectation value of annihilation operator :math:`\\langle a\\rangle` or, equivalently
            square root of the mean particle number.
    n: int
            Cut-off on the number of HO eigenstates kept in the definition of the coherent state.
    dtype: 'type'
            Data type (e.g. numpy.float64) to construct the coherent state with. Default is `np.float64`.

    Returns
    -------
    numpy.ndarray
            Harmonic oscilaltor coherent state.

    Examples
    --------

    >>> coherent_state(a,n,dtype=np.float64)

    """

    s1 = _np.full((n,), -_np.abs(a) ** 2 / 2.0, dtype=dtype)
    s2 = _np.arange(n, dtype=_np.float64)
    s3 = _np.array(s2)
    s3[0] = 1
    _np.log(s3, out=s3)
    s3[1:] = 0.5 * _np.cumsum(s3[1:])
    state = s1 + _np.log(a) * s2 - s3
    return _np.exp(state)


def photon_Hspace_dim(N, Ntot, Nph):
    """Calculates the dimension of the total spin-photon Hilbert space.

    Parameters
    ----------
    N: int
            Number of lattice particles.
    Ntot: int
            Total number of lattice particles and photons.
    Nph: int
            Number of photons.

    Returns
    -------
    int
            Dimension of the total spin-photon Hilbert space.

    Examples
    --------

    >>> Ns = photon_Hspace_dim(N,Ntot,Nph)

    """
    if Ntot is None and Nph is not None:  # no total particle # conservation
        return 2**N * (Nph + 1)
    elif Ntot is not None:
        return 2**N - binom(N, Ntot + 1) * hyp2f1(1, 1 - N + Ntot, 2 + Ntot, -1)
    else:
        raise TypeError("Either 'Ntot' or 'Nph' must be defined!")


class photon_basis(tensor_basis):
    """Constructs basis in photon-particle Hilbert space.

    The `photon_basis` is child class of `tensor_basis` which allows the user to define a basis which couples a lattice particle to a single photon mode.

    There are two types of `photon_basis` objects that one can create:
            * conserving basis: lattice particle basis and photon number conserved separately. This object is constructed using the `tensor_basis` class.
            * non-conserving basis: only total sum of lattice particles and photons conserved.

    The operator strings for the photon and lattice sectors are the same as for the lattice bases, respectively.
    The `photon_basis` operator string uses the pipe character '|' to distinguish between lattice operators (left) and photon operators (right).

    .. math::
       \\begin{array}{cccc}
          \\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}   &   \\texttt{"x"}   &   \\texttt{"y"}  \\newline
          \\texttt{spin_basis_*} &   \\hat{1}        &   \\hat S^+(\\hat\\sigma^+)       &   \\hat S^-(\\hat\\sigma^-)      &         -         &   \\hat S^z(\\hat\\sigma^z)       &   \\hat S^x(\\hat\\sigma^x)     &   \\hat S^y(\\hat\\sigma^y)  \\  \\newline
          \\texttt{boson_basis_*}&   \\hat{1}        &   \\hat b^\\dagger      &       \\hat b          & \\hat b^\\dagger \\hat b     &  \\hat b^\\dagger\\hat b - \\frac{\\mathrm{sps}-1}{2}       &   -       &   -  \\newline
          \\texttt{*_fermion_basis_*}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger \\hat c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}       &   \\hat c + \\hat c^\\dagger       &   -i\\left( \\hat c - \\hat c^\\dagger\\right)  \\newline
       \\end{array}

    Examples
    --------
    For the conserving basis, one can specify the total number of quanta using the the `Ntot` keyword argument:

    >>> p_basis = photon_basis(basis_class,*basis_args,Ntot=...,**symmetry_blocks)

    For the non-conserving basis, one must specify the total number of photon (a.k.a. harmonic oscillator)
    states using the `Nph` argument:

    >>> p_basis = photon_basis(basis_class,*basis_args,Nph=...,**symmetry_blocks)

    The code snippet below shows how to use the `photon_basis` class to construct the Jaynes-Cummings Hamiltonian. As an initial state, we choose a coherent state in the photon sector and the ground state of the two-level system (atom).

    .. literalinclude:: ../../doc_examples/photon_basis-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(self, basis_constructor, *constructor_args, **blocks):
        """Initialises the `photon_basis` object.

        Parameters
        ----------
        basis_constructor: :obj:`basis`
                `basis` constructor for the lattice part of the `photon_basis`.
        constructor_args: obj
                Required arguments required by the specific `basis` constructor.
        blocks: obj
                Optional keyword arguments for `basis_constructor` which include (but are not limited to):

                        **Nph** (*int*) - specify the dimension of photon (HO) Hilbert space.

                        **Ntot** (*int*) - specify total number of particles (photons + lattice).

                        **anyblock** (*int*) - specify any lattice symmetry blocks

        """

        Ntot = blocks.get("Ntot")
        Nph = blocks.get("Nph")
        self._Nph = Nph
        self._Ntot = Ntot
        if Ntot is not None:
            blocks.pop("Ntot")
        if Nph is not None:
            blocks.pop("Nph")

        if Ntot is None:
            if Nph is None:
                raise TypeError(
                    "If Ntot not specified, Nph must specify the cutoff on the number of photon states."
                )
            if type(Nph) is not int:
                raise TypeError("Nph must be integer")
            if Nph < 0:
                raise ValueError("Nph must be an integer >= 0.")

            self._check_pcon = False
            b1 = basis_constructor(*constructor_args, _Np=-1, **blocks)
            b2 = ho_basis(Nph)
            tensor_basis.__init__(self, b1, b2)
        else:
            if type(Ntot) is not int:
                raise TypeError("Ntot must be integer")
            if Ntot < 0:
                raise ValueError("Ntot must be an integer >= 0.")

            self._check_pcon = True
            self._basis_left = basis_constructor(*constructor_args, _Np=Ntot, **blocks)
            if isinstance(self._basis_left, tensor_basis):
                raise TypeError(
                    "Can only create photon basis with non-tensor type basis"
                )
            if not isinstance(self._basis_left, basis):
                raise TypeError("Can only create photon basis with basis type")
            self._basis_right = ho_basis(Ntot)
            self._n = self._basis_left._Np_list
            self._n -= Ntot
            self._n *= -1
            self._blocks = self._basis_left._blocks
            self._Ns = self._basis_left._Ns
            self._unique_me = self._basis_left._unique_me
            self._operators = (
                self._basis_left._operators + "\n" + self._basis_right._operators
            )

        self._sps = self._basis_left.sps



    @property
    def basis_particle(self):
        """:obj:`basis`: lattice basis."""
        return self._basis_left
    
    @property
    def basis_photon(self):
        """:obj:`basis`: photon basis."""
        return self._basis_right
    
    @property
    def particle_Ns(self):
        """int: number of states in the lattice Hilbert space only."""
        return self._basis_left.Ns

    @property
    def particle_N(self):
        """int: number of sites on lattice."""
        return self._basis_left.N

    @property
    def particle_sps(self):
        """int: number of lattice states per site."""
        return self._basis_left.sps

    @property
    def photon_Ns(self):
        """int: number of states in the photon Hilbert space only."""
        return self._basis_right.Ns

    @property
    def photon_sps(self):
        """int: number of photon states per site."""
        return self._basis_right.sps
    
    @property
    def chain_Ns(self):
        """int: number of states in the photon Hilbert space only."""
        return self._basis_left.Ns
    
    @property
    def chain_N(self):
        """int: number of sites on lattice."""
        return self._basis_left.N

    

    def Op(self, opstr, indx, J, dtype):
        """Constructs operator from a site-coupling list and anoperator string in the photon basis.

        Parameters
        ----------
        opstr: str
                Operator string in the photon basis format. Photon operators stand on the right. For instance:
                >>> opstr = "x|n"
        indx: list(int)
                List of integers to designate the sites the photon basis operator is defined on. The `photon_basis`
                assumes that the single photon couples globally to the lattice, hence the `indx` requires only
                the lattice site position. For instance:
                >>> indx = [5]
        J: scalar
                Coupling strength.
        dtype: 'type'
                Data type (e.g. numpy.float64) to construct the operator with.

        Returns
        -------
        tuple
                `(ME,row,col)`, where
                        * numpy.ndarray(scalar): `ME`: matrix elements of type `dtype`.
                        * numpy.ndarray(int): `row`: row indices of matrix representing the operator in the photon basis,
                                such that `row[i]` is the row index of `ME[i]`.
                        * numpy.ndarray(int): `col`: column index of matrix representing the operator in the photon basis,
                                such that `col[i]` is the column index of `ME[i]`.

        Examples
        --------

        >>> J = 1.41
        >>> indx = [5]
        >>> opstr = "x|n"
        >>> dtype = np.float64
        >>> ME, row, col = Op(opstr,indx,J,dtype)

        """
        if self._Ns <= 0:
            return [], [], []

        opstr1, opstr2 = opstr.split("|")

        if len(opstr1) != len(indx):
            raise ValueError(
                "The length of indx must be the same length as particle operators in {0},{1}".format(
                    opstr, indx
                )
            )

        if not self._check_pcon:
            n = len(opstr.replace("|", "")) - len(indx)
            indx = list(indx)
            indx.extend([0 for i in range(n)])

            return tensor_basis.Op(self, opstr, indx, J, dtype)
        else:
            # read off spin and photon operators
            n = len(opstr.replace("|", "")) - len(indx)
            indx.extend([0 for i in range(n)])

            if opstr.count("|") > 1:
                raise ValueError(
                    "only one '|' charactor allowed in opstr {0}".format(opstr)
                )
            if len(opstr) - 1 != len(indx):
                raise ValueError(
                    "not enough indices for opstr in: {0}, {1}".format(opstr, indx)
                )

            i = opstr.index("|")
            indx1 = indx[:i]
            indx2 = indx[i:]

            opstr1, opstr2 = opstr.split("|")

            # calculates matrix elements of spin and photon basis
            # the coupling 1.0 in self._basis_right.Op is used in order not to square the coupling J
            ME_ph, row_ph, col_ph = self._basis_right.Op(opstr2, indx2, 1.0, dtype)
            ME, row, col = self._basis_left.Op(opstr1, indx1, J, dtype)

            # calculate total matrix element
            ME *= ME_ph[self._n[col]]

            mask = ME != dtype(0.0)
            row = row[mask]
            col = col[mask]
            ME = ME[mask]

            del ME_ph, row_ph, col_ph

            return ME, row, col

    def get_vec(self, v0, sparse=True, Nph=None, full_part=True):
        """DEPRECATED (see `project_from`). Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        This function is :red:`deprecated`. Use `project_from()` instead (the inverse function, `project_to()`, is currently available in the `basis_general` classes only).

        """

        return self.project_from(v0, sparse=sparse, Nph=Nph, full_part=full_part)

    def project_from(self, v0, sparse=True, Nph=None, full_part=True):
        """Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        Particularly useful when a given operation canot be carried away in the symmetry-reduced basis in a straightforward manner.

        Supports parallelisation to multiple states listed in the columns.

        Parameters
        ----------
        v0: numpy.ndarray
                Contains in its columns the states in the symmetry-reduced basis.
        sparse: bool, optional
                Whether or not the output should be in sparse format. Default is `True`.
        Nph: int, optional
                Nuber of photons. Required for conserving `photon` basis.
        full_part: bool, optional
                Whether or not to transform the state to the full state in the lattice `basis` in the case
                of a conserving `photon_basis`. Default is `True`.

        Returns
        -------
        numpy.ndarray
                Array containing the state `v0` in the full basis.

        Examples
        --------

        >>> v_full = project_from(v0)
        >>> print(v_full.shape, v0.shape)

        """
        if not self._check_pcon:
            return tensor_basis.project_from(
                self, v0, sparse=sparse, full_left=full_part
            )
        else:
            if not hasattr(v0, "shape"):
                v0 = _np.asanyarray(v0)

            if Nph is None:
                Nph = self._Ntot

            if not type(Nph) is int:
                raise TypeError("Nph must be integer")

            if Nph < self._Ntot:
                raise ValueError(
                    "Nph must be larger or equal to {0}".format(self._Ntot)
                )

            if v0.ndim == 1:
                if v0.shape[0] != self._Ns:
                    raise ValueError("v0 has incompatible dimensions with basis")
                v0 = v0.reshape((-1, 1))
                if sparse:
                    return _conserved_project_from(self, v0, sparse, Nph, full_part)
                else:
                    return _conserved_project_from(
                        self, v0, sparse, Nph, full_part
                    ).reshape((-1,))

            elif v0.ndim == 2:
                if v0.shape[0] != self._Ns:
                    raise ValueError("v0 has incompatible dimensions with basis")

                if _sp.issparse(v0):
                    return self.get_proj(v0.dtype, Nph=Nph, full_part=full_part).dot(v0)

                return _conserved_project_from(self, v0, sparse, Nph, full_part)
            else:
                raise ValueError("excpecting v0 to have ndim at most 2")

    def get_proj(self, dtype, Nph=None, full_part=True):
        """Calculates transformation/projector from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
        in a straightforward manner.

        Parameters
        ----------
        dtype: 'type'
                Data type (e.g. numpy.float64) to construct the projector with.
        Nph: int, optional
                Nuber of photons. Required for conserving `photon` basis.
        full_part: bool, optional
                Whether or not to transform the state to the full state in the lattice `basis` in the case
                of a conserving `photon_basis`. Default is `True`.

        Returns
        -------
        numpy.ndarray
                Transformation/projector between the symmetry-reduced and the full basis.

        Examples
        --------

        >>> P = get_proj(np.float64)
        >>> print(P.shape)

        """
        if not self._check_pcon:
            return tensor_basis.get_proj(self, dtype, full_left=full_part)
        else:
            if Nph is None:
                Nph = self._Ntot

            if not type(Nph) is int:
                raise TypeError("Nph must be integer")

            if Nph < self._Ntot:
                raise ValueError(
                    "Nph must be larger or equal to {0}".format(self._Ntot)
                )

            return _conserved_get_proj(self, dtype, Nph, full_part)

    def partial_trace(
        self,
        state,
        sub_sys_A="particles",
        return_rdm=None,
        enforce_pure=False,
        sparse=False,
    ):
        """Calculates reduced density matrix, through a partial trace of a quantum state in `photon_basis`.

        Parameters
        ----------
        state: obj
                State of the quantum system. Can be either one of:

                        * numpy.ndarray [shape (Ns,)]: pure state (default).
                        * numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
        sub_sys_A: str, optional
                Subsystem to calculate the density matrix of. Can be either one of:

                        * "particles": refers to lattice subsystem (Default).
                        * "photons": refers to photon subsystem.
        return_rdm: str, optional
                Toggles returning the reduced DM. Can be tierh one of:

                        * "A": returns reduced DM of subsystem A.
                        * "B": returns reduced DM of subsystem B.
                        * "both": returns reduced DM of both A and B subsystems.
        enforce_pure: bool, optional
                Whether or not to assume `state` is a colelction of pure states or a mixed density matrix, if
                it is a square array. Default is `False`.
        sparse: bool, optional
                Whether or not to return a sparse DM. Default is `False`.

        Returns
        -------
        numpy.ndarray
                Density matrix associated with `state`. Depends on optional arguments.


        Examples
        --------

        >>> partial_trace(state,sub_sys_A=None,return_rdm="A",enforce_pure=False,sparse=False)

        """

        if sub_sys_A is None:
            sub_sys_A = "particles"

        tensor_dict = {
            "particles": "left",
            "photons": "right",
            "both": "both",
            "left": "left",
            "right": "right",
            None: None,
        }

        if sub_sys_A not in tensor_dict:
            raise ValueError("sub_sys_A '{}' not recognized".format(sub_sys_A))

        if not hasattr(state, "shape"):
            state = _np.asanyarray(state)

        if state.shape[0] != self.Ns:
            raise ValueError(
                "state shape {0} not compatible with Ns={1}".format(
                    state.shape, self._Ns
                )
            )

        if self._check_pcon:  # project to full photon basis
            if _sp.issparse(state) or sparse:
                proj_state = self.project_from(state, sparse=True, full_part=False)
            else:
                if state.ndim == 1:
                    # calculate full H-space representation of state
                    proj_state = self.project_from(state, sparse=False, full_part=False)

                elif state.ndim == 2:
                    if state.shape[0] != state.shape[1] or enforce_pure:
                        # calculate full H-space representation of state
                        proj_state = self.project_from(
                            state, sparse=False, full_part=False
                        )

                    else:
                        proj = self.get_proj(_dtypes[state.dtype.char], full_part=False)
                        proj_state = proj * state * proj.T.conj()

                        shape0 = proj_state.shape
                        proj_state = proj_state.reshape(shape0 + (1,))

                elif state.ndim == 3:  # 3D DM
                    proj = self.get_proj(_dtypes[state.dtype.char])
                    state = state.transpose((2, 0, 1))

                    Ns_full = proj.shape[0]
                    n_states = state.shape[0]

                    gen = (proj * s * proj.T.conj() for s in state[:])

                    proj_state = _np.zeros(
                        (Ns_full, Ns_full, n_states), dtype=_dtypes[state.dtype.char]
                    )

                    for i, s in enumerate(gen):
                        proj_state[..., i] += s[...]
                else:
                    raise ValueError("state must have ndim < 4")
        else:
            proj_state = state

        return tensor_basis.partial_trace(
            self,
            proj_state,
            sub_sys_A=tensor_dict[sub_sys_A],
            return_rdm=return_rdm,
            enforce_pure=enforce_pure,
            sparse=sparse,
        )

    def ent_entropy(
        self,
        state,
        sub_sys_A="particles",
        return_rdm=None,
        enforce_pure=False,
        return_rdm_EVs=False,
        sparse=False,
        alpha=1.0,
        sparse_diag=True,
        maxiter=None,
    ):
        """Calculates entanglement entropy of subsystem A and the corresponding reduced density matrix

        .. math::
                S_\\mathrm{ent}(\\alpha) =  \\frac{1}{1-\\alpha}\\log \\mathrm{tr}_{A} \\left( \\mathrm{tr}_{A^c} \\vert\\psi\\rangle\\langle\\psi\\vert \\right)^\\alpha

        **Note:** The logarithm used is the natural logarithm (base e).

        Notes
        -----
        Algorithm is based on both partial tracing and sigular value decomposition (SVD), optimised for speed.

        Parameters
        ----------
        state: obj
                State of the quantum system. Can be either one of:

                        * numpy.ndarray [shape (Ns,)]: pure state (default).
                        * numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
        sub_sys_A: str, optional
                Subsystem to calculate the density matrix of. Can be either one of:

                        * "particles": refers to lattice subsystem (Default).
                        * "photons": refers to photon subsystem.
        return_rdm: str, optional
                Toggles returning the reduced DM. Can be tierh one of:

                        * "A": returns reduced DM of subsystem A.
                        * "B": returns reduced DM of subsystem B.
                        * "both": returns reduced DM of both A and B subsystems.
        enforce_pure: bool, optional
                Whether or not to assume `state` is a colelction of pure states or a mixed density matrix, if
                it is a square array. Default is `False`.
        sparse: bool, optional
                Whether or not to return a sparse DM. Default is `False`.
        return_rdm_EVs: bool, optional
                Whether or not to return the eigenvalues of rthe educed DM. If `return_rdm` is specified,
                the eigenvalues of the corresponding DM are returned. If `return_rdm` is NOT specified,
                the spectrum of `rdm_A` is returned by default. Default is `False`.
        alpha: float, optional
                Renyi :math:`\\alpha` parameter for the entanglement entropy. Default is :math:`\\alpha=1`.
        sparse_diag: bool, optional
                When `sparse=True`, this flag enforces the use of
                `scipy.sparse.linalg.eigsh() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_
                to calculate the eigenvaues of the reduced DM.
        maxiter: int, optional
                Specifies the number of iterations for Lanczos diagonalisation. Look up documentation for
                `scipy.sparse.linalg.eigsh() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_.


        Returns
        -------
        dict
                Dictionary with following keys, depending on input parameters:
                        * "Sent_A": entanglement entropy of subsystem A (default).
                        * "Sent_B": entanglement entropy of subsystem B.
                        * "p_A": singular values of reduced DM of subsystem A (default).
                        * "p_B": singular values of reduced DM of subsystem B.
                        * "rdm_A": reduced DM of subsystem A.
                        * "rdm_B": reduced DM of subsystem B.

        Examples
        --------

        >>> ent_entropy(state,sub_sys_A="photons",return_rdm="A",enforce_pure=False,return_rdm_EVs=False,
        >>>				sparse=False,alpha=1.0,sparse_diag=True)

        """

        if self._check_pcon:  # project to full photon basis
            if _sp.issparse(state) or sparse:
                proj_state = self.project_from(state, sparse=True, full_part=False)
            else:
                if state.ndim == 1:
                    # calculate full H-space representation of state
                    proj_state = self.project_from(state, sparse=False, full_part=False)
                elif state.ndim == 2:
                    if state.shape[0] != state.shape[1] or enforce_pure:
                        # calculate full H-space representation of state
                        proj_state = self.project_from(
                            state, sparse=False, full_part=False
                        )
                    else:
                        proj = self.get_proj(_dtypes[state.dtype.char], full_part=False)
                        proj_state = proj * state * proj.T.conj()

                elif state.ndim == 3:  # 3D DM
                    proj = self.get_proj(_dtypes[state.dtype.char], full_part=False)

                    Ns_full = proj.shape[0]
                    n_states = state.shape[-1]

                    gen = (proj * state[:, :, i] * proj.T.conj() for i in range(n_states))
                    proj_state = _np.zeros(
                        (Ns_full, Ns_full, n_states), dtype=_dtypes[state.dtype.char]
                    )

                    for i, s in enumerate(gen):
                        proj_state[..., i] += s[...]
                else:
                    raise ValueError("state must have ndim < 4")
        else:
            proj_state = state

        tensor_dict = {
            "particles": "left",
            "photons": "right",
            "both": "both",
            "left": "left",
            "right": "right",
            None: None,
        }
        if sub_sys_A in tensor_dict:
            return tensor_basis.ent_entropy(
                self,
                proj_state,
                sub_sys_A=tensor_dict[sub_sys_A],
                return_rdm=return_rdm,
                alpha=alpha,
                sparse=sparse,
                sparse_diag=sparse_diag,
                maxiter=maxiter,
            )
        else:
            raise ValueError("sub_sys_A '{}' not recognized".format(return_rdm))

    ##### private methods of the photon class

    def __name__(self):
        return "<type 'quspin.basis.photon_basis'>"

    def _get__str__(self):
        if not self._check_pcon:
            return tensor_basis._get__str__(self)
        else:
            if not hasattr(self._basis_left, "_get__str__"):
                warnings.warn(
                    "basis class {0} missing _get__str__ function, can not print out basis representatives.".format(
                        type(self._basis_left)
                    ),
                    UserWarning,
                    stacklevel=3,
                )
                return "reference states: \n\t not availible"

            n_digits = len(str(self.Ns)) + 1
            n_space = len(str(self._Ntot))
            str_list_1 = self._basis_left._get__str__()
            temp = "\t{0:" + str(n_digits) + "d}.  "
            str_list = []
            for b1 in str_list_1:
                b1, s1 = b1.split(".  ")
                i1 = int(b1)
                s2 = ("|{:" + str(n_space) + "d}>").format(self._n[i1])
                str_list.append((temp.format(i1)) + "\t" + s1 + s2)

            if self._Ns > MAXPRINT:
                half = MAXPRINT // 2
                str_list_1 = str_list[:half]
                str_list_2 = str_list[-half:]

                str_list = str_list_1
                str_list.extend(str_list_2)

            return str_list

    def _check_symm(self, static, dynamic):
        # pick out operators which have charactors to the left of the '|' charactor.
        # otherwise this is operator must be
        new_static = []
        for opstr, bonds in static:
            if opstr.count("|") == 0:
                raise ValueError("missing '|' character in: {0}".format(opstr))

            opstr1, opstr2 = opstr.split("|")

            if opstr1:
                new_static.append([opstr, bonds])

        new_dynamic = []
        for opstr, bonds, f, f_args in dynamic:
            if opstr.count("|") == 0:
                raise ValueError("missing '|' character in: {0}".format(opstr))

            opstr1, opstr2 = opstr.split("|")

            if opstr1:
                new_dynamic.append([opstr, bonds, f, f_args])

        return self._basis_left._check_symm(new_static, new_dynamic, photon_basis=self)

    def _sort_opstr(self, op):
        op = list(op)
        opstr = op[0]
        indx = op[1]

        if opstr.count("|") == 0:
            raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr, indx))

        if opstr.count("|") > 1:
            raise ValueError(
                "only one '|' charactor allowed in: {0}, {1}".format(opstr, indx)
            )

        if len(opstr) - 1 != len(indx):
            raise ValueError(
                "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
            )

        i = opstr.index("|")
        indx1 = indx[:i]
        indx2 = indx[i:]

        opstr1, opstr2 = opstr.split("|")

        op1 = list(op)
        op1[0] = opstr1
        op1[1] = tuple(indx1)

        if indx1:
            ind_min = min(indx1)
        else:
            ind_min = 0

        op2 = list(op)
        op2[0] = opstr2
        op2[1] = tuple([ind_min for i in opstr2])

        op1 = self._basis_left._sort_opstr(op1)
        op2 = self._basis_right._sort_opstr(op2)

        op[0] = "|".join((op1[0], op2[0]))
        op[1] = op1[1] + op2[1]

        return tuple(op)

    def _get_local_lists(
        self, static, dynamic
    ):  # overwrite the default get_local_lists from base.
        static_list = []
        for opstr, bonds in static:
            if opstr.count("|") == 0:
                raise ValueError("missing '|' character in: {0}".format(opstr))

            opstr1, opstr2 = opstr.split("|")

            for bond in bonds:
                indx = list(bond[1:])

                if len(opstr1) != len(indx):
                    raise ValueError(
                        "The length of indx must be the same length as particle operators in {0},{1}".format(
                            opstr, indx
                        )
                    )

                # extend the operators such that the photon ops get an index.
                # choose that the index is equal to the smallest indx of the spin operators
                n = len(opstr.replace("|", "")) - len(indx)

                if opstr1:
                    indx.extend([min(indx) for i in range(n)])
                else:
                    indx.extend([0 for i in range(n)])

                J = complex(bond[0])
                static_list.append((opstr, tuple(indx), J))

        dynamic_list = []
        for opstr, bonds, f, f_args in dynamic:
            if opstr.count("|") == 0:
                raise ValueError("missing '|' character in: {0}".format(opstr))

            opstr1, opstr2 = opstr.split("|")

            for bond in bonds:
                indx = list(bond[1:])

                if len(opstr1) != len(indx):
                    raise ValueError(
                        "The length of indx must be the same length as particle operators in {0},{1}".format(
                            opstr, indx
                        )
                    )

                # extend the operators such that the photon ops get an index.
                # choose that the index is equal to the smallest indx of the spin operators
                n = len(opstr.replace("|", "")) - len(indx)

                if opstr1:
                    indx.extend([min(indx) for i in range(n)])
                else:
                    indx.extend([0 for i in range(n)])

                J = complex(bond[0])
                dynamic_list.append((opstr, tuple(indx), J, f, f_args))

        return self._sort_local_list(static_list), self._sort_local_list(dynamic_list)

    


def _conserved_project_from(p_basis, v0, sparse, Nph, full_part):
    v0_mask = _np.zeros_like(v0)
    np_min = p_basis._n.min()
    np_max = p_basis._n.max()
    v_ph = _np.zeros((Nph + 1, 1), dtype=_np.int8)

    v_ph[np_min] = 1
    mask = p_basis._n == np_min
    v0_mask[mask] = v0[mask]

    if full_part:
        v0_full = p_basis._basis_left.project_from(v0_mask, sparse=sparse)
    else:
        v0_full = v0_mask

    if sparse:
        v0_full = _sp.kron(v0_full, v_ph, format="csr")
    else:
        v0_full = _np.kron(v0_full, v_ph)

    v_ph[np_min] = 0
    v0_mask[mask] = 0.0

    for np in range(np_min + 1, np_max + 1, 1):
        v_ph[np] = 1
        mask = p_basis._n == np
        v0_mask[mask] = v0[mask]

        if full_part:
            v0_full_1 = p_basis._basis_left.project_from(v0_mask, sparse=sparse)
        else:
            v0_full_1 = v0_mask

        if sparse:
            v0_full = v0_full + _sp.kron(v0_full_1, v_ph, format="csr")
            v0_full.sum_duplicates()
            v0_full.eliminate_zeros()
        else:
            v0_full += _np.kron(v0_full_1, v_ph)

        v_ph[np] = 0
        v0_mask[mask] = 0.0

    return v0_full


def _conserved_get_proj(p_basis, dtype, Nph, full_part):
    np_min = p_basis._n.min()
    np_max = p_basis._n.max()
    v_ph = _np.zeros((Nph + 1, 1), dtype=_np.int8)

    if full_part:
        proj_1 = p_basis._basis_left.get_proj(dtype)
    else:
        proj_1 = _sp.identity(p_basis.Ns, dtype=dtype, format="csr")

    proj_1_mask = _sp.lil_matrix(proj_1.shape, dtype=dtype)

    v_ph[np_min] = 1
    mask = p_basis._n == np_min
    proj_1_mask[:, mask] = proj_1[:, mask]

    proj_1_full = _sp.kron(proj_1_mask, v_ph, format="csr")

    proj_1_mask[:, :] = 0.0
    v_ph[np_min] = 0

    for np in range(np_min + 1, np_max + 1, 1):
        v_ph[np] = 1
        mask = p_basis._n == np
        proj_1_mask[:, mask] = proj_1[:, mask]

        proj_1_full = proj_1_full + _sp.kron(proj_1_mask, v_ph, format="csr")

        proj_1_mask[:, :] = 0.0
        v_ph[np] = 0

    return proj_1_full


# helper class which calcualates ho matrix elements
class ho_basis(basis):
    def __init__(self, Np):
        if type(Np) is not int:
            raise ValueError("expecting integer for Np")

        self._Np = Np
        self._Ns = Np + 1
        self._N = 1
        self._dtype = _np.min_scalar_type(-self.Ns)
        self._basis = _np.arange(self.Ns, dtype=_np.min_scalar_type(self.Ns))
        self._operators = (
            "availible operators for ho_basis:"
            + "\n\tI: identity "
            + "\n\t+: raising operator"
            + "\n\t-: lowering operator"
            + "\n\tn: number operator"
        )

        self._blocks = {}
        self._unique_me = True

    @property
    def Np(self):
        return self._Np

    @property
    def N(self):
        return 1

    @property
    def sps(self):
        return self._Np + 1

    def get_vec(self, v0, sparse=True):
        return self.project_from(v0, sparse=sparse)

    def project_from(self, v0, sparse=True):
        if self._Ns <= 0:
            return _np.array([])
        if v0.ndim == 1:
            if v0.shape[0] != self.Ns:
                raise ValueError("v0 has incompatible dimensions with basis")
            v0 = v0.reshape((-1, 1))
        elif v0.ndim == 2:
            if v0.shape[0] != self.Ns:
                raise ValueError("v0 has incompatible dimensions with basis")
        else:
            raise ValueError("excpecting v0 to have ndim at most 2")

        if sparse:
            return _sp.csr_matrix(v0)
        else:
            return v0

    def __getitem__(self, key):
        return self._basis.__getitem__(key)

    def index(self, s):
        return _np.searchsorted(self._basis, s)

    def __iter__(self):
        return self._basis.__iter__()

    def _sort_opstr(self, op):
        return tuple(op)

    def _get__str__(self):
        def get_state(b):
            return ("|{0:" + str(len(str(self.Ns))) + "}>").format(b)

        temp1 = "     {0:" + str(len(str(self.Ns))) + "d}.  "
        if self._Ns > MAXPRINT:
            half = MAXPRINT // 2
            str_list = [
                (temp1.format(i)) + get_state(b)
                for i, b in zip(range(half), self._basis[:half])
            ]
            str_list.extend(
                [
                    (temp1.format(i)) + get_state(b)
                    for i, b in zip(
                        range(self._Ns - half, self._Ns, 1), self._basis[-half:]
                    )
                ]
            )
        else:
            str_list = [
                (temp1.format(i)) + get_state(b) for i, b in enumerate(self._basis)
            ]

        return tuple(str_list)

    def _hc_opstr(self, op):
        op = list(op)
        op[0] = list(op[0].replace("+", "%").replace("-", "+").replace("%", "-"))
        op[0].reverse()
        op[0] = "".join(op[0])

        op[1] = list(op[1])
        op[1].reverse()
        op[1] = tuple(op[1])

        op[2] = op[2].conjugate()
        return self._sort_opstr(op)

    def _non_zero(self, op):
        m = op[0].count("-") > self._Np
        p = op[0].count("+") > self._Np
        return p or m

    def _expand_opstr(self, op, num):
        op = list(op)
        op.append(num)
        op = tuple(op)
        return tuple([op])

    def get_proj(self, dtype):
        return _sp.identity(self.Ns, dtype=dtype)

    def Op(self, opstr, indx, J, dtype, *args):

        row = _np.array(self._basis, dtype=self._dtype)
        col = _np.array(self._basis, dtype=self._dtype)
        ME = _np.ones((self._Ns,), dtype=dtype)

        if len(opstr) != len(indx):
            raise ValueError("length of opstr does not match length of indx")
        if not _np.can_cast(_np.min_scalar_type(J), _np.dtype(dtype)):
            raise TypeError("can't cast J to proper dtype")

        for o in opstr[::-1]:
            if o == "I":
                continue
            elif o == "n":
                ME *= dtype(_np.abs(row))
            elif o == "+":
                row += 1
                ME *= _np.sqrt(dtype(_np.abs(row)))
            elif o == "-":
                ME *= _np.sqrt(dtype(_np.abs(row)))
                row -= 1
            else:
                raise Exception("operator symbol {0} not recognized".format(o))

        mask = row < 0
        mask += row >= (self._Ns)
        row[mask] = col[mask]
        ME[mask] = 0.0

        if J != 1.0:
            ME *= J

        return ME, row, col
