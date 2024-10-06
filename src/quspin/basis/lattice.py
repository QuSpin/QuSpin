from quspin.basis.base import basis, MAXPRINT
from quspin.basis._reshape_subsys import (
    _lattice_partial_trace_pure,
    _lattice_reshape_pure,
)
from quspin.basis._reshape_subsys import (
    _lattice_partial_trace_mixed,
    _lattice_reshape_mixed,
)
from quspin.basis._reshape_subsys import (
    _lattice_partial_trace_sparse_pure,
    _lattice_reshape_sparse_pure,
)
import numpy as _np
import scipy.sparse as _sp
from numpy.linalg import norm, eigvalsh
from scipy.sparse.linalg import eigsh
import warnings
from quspin_extensions.basis._basis_utils import fermion_ptrace_sign, anyon_ptrace_phase
from itertools import chain

_dtypes = {"f": _np.float32, "d": _np.float64, "F": _np.complex64, "D": _np.complex128}


class lattice_basis(basis):
    def __init__(self):
        basis.__init__(self)
        self._Ns = 0
        self._basis = _np.asarray([])
        self._operators = "no operators for base."
        self._noncommuting_bits = []

        if self.__class__.__name__ == "lattice_basis":
            raise ValueError(
                "This class is not intended" " to be instantiated directly."
            )

    @property
    def noncommuting_bits(self):
        """list: list of bits that represent sites that do not commute along with the phase required from commuting sites"""
        return self._noncommuting_bits

    def __getitem__(self, key):
        return self._basis.__getitem__(key)

    def __iter__(self):
        return self._basis.__iter__()

    @property
    def dtype(self):
        """numpy.dtype: data type of basis state integers."""
        return self._basis.dtype

    @property
    def states(self):
        """numpy.ndarray(int): basis states stored in their integer representation."""
        basis_view = self._basis[:]
        basis_view.setflags(write=0, uic=0)
        return basis_view

    def _int_to_state(self, *args, **kwargs):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_int_to_state' required for printing a child of lattice basis!".format(
                self.__class__
            )
        )

    def _state_to_int(self, *args, **kwargs):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_state_to_int' required for for printing a child of lattice basis".format(
                self.__class__
            )
        )

    def _index(self, *args, **kwargs):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_index' required for searching for index of representative!".format(
                self.__class__
            )
        )

    def int_to_state(self, state, bracket_notation=True):
        """Finds string representation of a state defined in integer representation.

        Notes
        -----
        This function is the inverse of `state_to_int`.

        Parameters
        ----------
        state : int
                Defines the Fock state in integer representation in underlying lattice `basis`.
        bracket_notation : bool, optional
                Toggles whether to return the state in `|str>` notation.

        Returns
        -------
        str
                String corresponding to the Fock `state` in the lattice basis.

        Examples
        --------

        >>> s = basis[0] # pick state from basis set
        >>> s_str = basis.int_to_state(s)
        >>> print(s_str)

        """
        return self._int_to_state(state, bracket_notation=bracket_notation)

    def state_to_int(self, state):
        """Finds integer representation of a state defined in string format.

        Notes
        -----
        This function is the einverse of `int_to_state`.

        Parameters
        ----------
        state : str
                Defines the Fock state with number of particles (spins) per site in underlying lattice `basis`.

        Returns
        -------
        int
                Integer corresponding to the Fock `state` in the lattice basis.

        Examples
        --------

        >>> s_str = "111000" # pick state from basis set
        >>> s = basis.state_to_int(s_str)
        >>> print(s)

        """

        return self._state_to_int(state)

    def index(self, s):
        """Finds the index of user-defined Fock state in any lattice basis.

        Notes
        -----
        Particularly useful for defining initial Fock states through a unit vector in the direction specified
        by `index()`.

        Parameters
        ----------
        s : {str, int}
                Defines the Fock state with number of particles (spins) per site in underlying lattice `basis`.

        Returns
        -------
        int
                Position of the Fock state in the lattice basis.

        Examples
        --------

        >>> i0 = index("111000") # pick state from basis set
        >>> print(basis)
        >>> print(i0)
        >>> psi = np.zeros(basis.Ns,dtype=np.float64)
        >>> psi[i0] = 1.0 # define state corresponding to the string "111000"

        """
        return self._index(s)

    def _ptrace_signs(self, sub_sys_A):
        """
        Helper function to compute the signs for the partial trace for noncontiguous femrionic subsystems.
        """
        # compute subsystem B and construct the entire system
        sub_sys_B = list(set(range(self.N)) - set(sub_sys_A))
        sub_sys_B.sort()
        system = _np.concatenate((sub_sys_A, sub_sys_B))
        # compute the map sich that system[pmap]=[0,1,...,N-1]
        pmap = _np.argsort(system).astype(_np.int32)
        # if mixing species of particles calculate signs
        # just for each set of sites

        phase_dtype = _np.int8

        for bits, phase in self.noncommuting_bits:
            phase_dtype = _np.result_type(phase_dtype, phase.dtype)

        phase_array = _np.ones(2**self.N, dtype=phase_dtype)

        for bits, phase in self.noncommuting_bits:
            m = _np.array(sum(1 << b for b in bits), dtype=_np.uint64)
            if phase.dtype == _np.int8:
                fermion_ptrace_sign(pmap, m, phase_array)
            elif phase.dtype == _np.complex128:
                anyon_ptrace_phase(pmap, m, phase_array)

        # for i,sign in enumerate(phase_array):
        # 	s = _np.array(2**self.N-i-1,dtype=_np.uint64)
        # 	print(sign,self.int_to_state(s))
        return phase_array

    def _partial_trace(
        self,
        state,
        sub_sys_A=None,
        subsys_ordering=True,
        return_rdm="A",
        enforce_pure=False,
        sparse=False,
    ):
        """Calculates reduced density matrix, through a partial trace of a quantum state in a lattice `basis`."""

        if sub_sys_A is None:
            sub_sys_A = tuple(range(self.N // 2))
        elif len(sub_sys_A) == self.N:
            raise ValueError(
                "Size of subsystem must be strictly smaller than total system size N!"
            )

        N_A = len(sub_sys_A)
        N_B = self.N - N_A

        if sub_sys_A is None:
            sub_sys_A = tuple(range(self.N // 2))

        sub_sys_A = tuple(sub_sys_A)

        if any(not _np.issubdtype(type(s), _np.integer) for s in sub_sys_A):
            raise ValueError(
                "sub_sys_A must iterable of integers with values in {0,...,N-1}!"
            )

        if any(s < 0 or s > self.N for s in sub_sys_A):
            raise ValueError(
                "sub_sys_A must iterable of integers with values in {0,...,N-1}"
            )

        doubles = tuple(s for s in sub_sys_A if sub_sys_A.count(s) > 1)
        if len(doubles) > 0:
            raise ValueError("sub_sys_A contains repeated values: {}".format(doubles))

        if return_rdm not in set(["A", "B", "both"]):
            raise ValueError("return_rdm must be: 'A','B','both' or None")

        if subsys_ordering:
            sub_sys_A = sorted(sub_sys_A)

        if not hasattr(state, "shape"):
            state = _np.asanyarray(state)
            state = state.squeeze()  # avoids artificial higher-dim reps of ndarray

        if state.shape[0] != self.Ns:
            raise ValueError(
                "state shape {0} not compatible with Ns={1}".format(
                    state.shape, self._Ns
                )
            )

        sps = self.sps
        N = self.N

        # compute signs for fermion bases
        compute_signs = len(self.noncommuting_bits) != 0

        if compute_signs:
            sign_array = self._ptrace_signs(sub_sys_A)

        if _sp.issparse(state) or sparse:
            state = self.project_from(state, sparse=True).T

            if compute_signs:  # imprint fermion signs
                state = state.multiply(sign_array)

            if state.shape[0] == 1:
                # sparse_pure partial trace
                rdm_A, rdm_B = _lattice_partial_trace_sparse_pure(
                    state, sub_sys_A, N, sps, return_rdm=return_rdm
                )
            else:
                if state.shape[0] != state.shape[1] or enforce_pure:
                    # vectorize sparse_pure partial trace
                    state = state.tocsr()
                    try:
                        state_gen = [
                            _lattice_partial_trace_sparse_pure(
                                state.getrow(i),
                                sub_sys_A,
                                N,
                                sps,
                                return_rdm=return_rdm,
                            )
                            for i in xrange(state.shape[0])
                        ]
                    except NameError:
                        state_gen = [
                            _lattice_partial_trace_sparse_pure(
                                state.getrow(i),
                                sub_sys_A,
                                N,
                                sps,
                                return_rdm=return_rdm,
                            )
                            for i in range(state.shape[0])
                        ]

                    left, right = zip(*state_gen)

                    rdm_A, rdm_B = _np.stack(left), _np.stack(right)

                    if any(rdm is None for rdm in rdm_A):
                        rdm_A = None

                    if any(rdm is None for rdm in rdm_B):
                        rdm_B = None
                else:
                    raise ValueError("Expecting a dense array for mixed states.")

        else:
            if state.ndim == 1:
                # calculate full H-space representation of state
                state = self.project_from(state, sparse=False)

                if compute_signs:  # imprint fermion signs
                    state *= sign_array

                rdm_A, rdm_B = _lattice_partial_trace_pure(
                    state.T, sub_sys_A, N, sps, return_rdm=return_rdm
                )

            elif state.ndim == 2:
                if state.shape[0] != state.shape[1] or enforce_pure:
                    # calculate full H-space representation of state
                    state = self.project_from(state, sparse=False)

                    if compute_signs:  # imprint fermion signs
                        state = (state.T * sign_array).T

                    rdm_A, rdm_B = _lattice_partial_trace_pure(
                        state.T, sub_sys_A, N, sps, return_rdm=return_rdm
                    )

                else:
                    proj = self.get_proj(_dtypes[state.dtype.char])
                    proj_state = proj * state * proj.T.conj()

                    if compute_signs:  # imprint fermion signs
                        proj_state *= _np.outer(sign_array, sign_array)

                    shape0 = proj_state.shape
                    proj_state = proj_state.reshape((1,) + shape0)

                    rdm_A, rdm_B = _lattice_partial_trace_mixed(
                        proj_state, sub_sys_A, N, sps, return_rdm=return_rdm
                    )

            elif state.ndim == 3:  # 3D DM
                proj = self.get_proj(_dtypes[state.dtype.char])
                state = state.transpose((2, 0, 1))

                Ns_full = proj.shape[0]
                n_states = state.shape[0]

                if compute_signs:  # imprint fermion signs
                    signs_mask = _np.outer(sign_array, sign_array)
                    gen = (
                        _np.multiply(proj * s * proj.T.conj(), signs_mask) for s in state[:]
                    )
                else:
                    gen = (proj * s * proj.T.conj() for s in state[:])

                proj_state = _np.zeros(
                    (n_states, Ns_full, Ns_full), dtype=_dtypes[state.dtype.char]
                )

                for i, s in enumerate(gen):
                    proj_state[i, ...] += s[...]

                rdm_A, rdm_B = _lattice_partial_trace_mixed(
                    proj_state, sub_sys_A, N, sps, return_rdm=return_rdm
                )
            else:
                raise ValueError("state must have ndim < 4")

        if return_rdm == "A":
            return rdm_A
        elif return_rdm == "B":
            return rdm_B
        else:
            return rdm_A, rdm_B

    def _ent_entropy(
        self,
        state,
        sub_sys_A=None,
        density=True,
        subsys_ordering=True,
        return_rdm=None,
        enforce_pure=False,
        return_rdm_EVs=False,
        sparse=False,
        alpha=1.0,
        sparse_diag=True,
        maxiter=None,
        svd_solver=None,
        svd_kwargs=None,
    ):
        """Calculates entanglement entropy of subsystem A and the corresponding reduced density matrix"""
        if sub_sys_A is None:
            sub_sys_A = list(range(self.N // 2))
        else:
            sub_sys_A = list(sub_sys_A)

        if len(sub_sys_A) >= self.N:
            raise ValueError(
                "Size of subsystem must be strictly smaller than total system size N!"
            )

        N_A = len(sub_sys_A)
        N_B = self.N - N_A

        if any(not _np.issubdtype(type(s), _np.integer) for s in sub_sys_A):
            raise ValueError(
                "sub_sys_A must iterable of integers with values in {0,...,N-1}!"
            )

        if any(s < 0 or s > self.N for s in sub_sys_A):
            raise ValueError(
                "sub_sys_A must iterable of integers with values in {0,...,N-1}"
            )

        doubles = tuple(s for s in set(sub_sys_A) if sub_sys_A.count(s) > 1)
        if len(doubles) > 0:
            raise ValueError("sub_sys_A contains repeated values: {}".format(doubles))

        if return_rdm not in set(["A", "B", "both", None]):
            raise ValueError("return_rdm must be: 'A','B','both' or None")

        if subsys_ordering:
            sub_sys_A = sorted(sub_sys_A)

        if not hasattr(state, "shape"):
            state = _np.asanyarray(state)
            state = state.squeeze()  # avoids artificial higher-dim reps of ndarray

        if state.shape[0] != self.Ns:
            raise ValueError(
                "state shape {0} not compatible with Ns={1}".format(
                    state.shape, self._Ns
                )
            )

        sps = self.sps
        N = self.N

        # compute signs for fermion bases AND non-contiguous subsystems w.r.t. the sign convention
        compute_signs = len(self.noncommuting_bits) != 0

        pure = True  # set pure state parameter to True
        if _sp.issparse(state) or sparse:
            if state.ndim == 1:
                state = state.reshape((-1, 1))

            sparse = True  # set sparse flag to True
            if state.shape[1] == 1:
                # calls _partial_trace
                p, rdm_A, rdm_B = self._p_pure_sparse(
                    state,
                    sub_sys_A,
                    return_rdm=return_rdm,
                    sparse_diag=sparse_diag,
                    maxiter=maxiter,
                )
            else:
                if state.shape[0] != state.shape[1] or enforce_pure:
                    # calls _partial_trace
                    p, rdm_A, rdm_B = self._p_pure_sparse(
                        state, sub_sys_A, return_rdm=return_rdm
                    )
                else:
                    raise ValueError("Expecting a dense array for mixed states.")

        else:
            if state.ndim == 1:
                state = state.reshape((-1, 1))
                p, rdm_A, rdm_B = self._p_pure(
                    state,
                    sub_sys_A,
                    svd_solver=svd_solver,
                    svd_kwargs=svd_kwargs,
                    return_rdm=return_rdm,
                    compute_signs=compute_signs,
                )

            elif state.ndim == 2:

                if state.shape[0] != state.shape[1] or enforce_pure:
                    p, rdm_A, rdm_B = self._p_pure(
                        state,
                        sub_sys_A,
                        svd_solver=svd_solver,
                        svd_kwargs=svd_kwargs,
                        return_rdm=return_rdm,
                        compute_signs=compute_signs,
                    )
                else:  # 2D mixed
                    pure = False
                    """
					# check if DM's are positive definite
					try:
						_np.linalg.cholesky(state)
					except:
						raise ValueError("LinAlgError: (collection of) DM(s) not positive definite")
					# check oif trace of DM is unity
					if _np.any( abs(_np.trace(state) - 1.0 > 1E3*_np.finfo(state.dtype).eps)  ):
						raise ValueError("Expecting eigenvalues of DM to sum to unity!")
					"""
                    shape0 = state.shape
                    state = state.reshape(shape0 + (1,))
                    p_A, p_B, rdm_A, rdm_B = self._p_mixed(
                        state,
                        sub_sys_A,
                        return_rdm=return_rdm,
                        compute_signs=compute_signs,
                    )

            elif state.ndim == 3:  # 3D DM
                pure = False

                """
				# check if DM's are positive definite
				try:
					_np.linalg.cholesky(state)
				except:
					raise ValueError("LinAlgError: (collection of) DM(s) not positive definite")

				# check oif trace of DM is unity
				if _np.any( abs(_np.trace(state, axis1=1,axis2=2) - 1.0 > 1E3*_np.finfo(state.dtype).eps)  ):
					raise ValueError("Expecting eigenvalues of DM to sum to unity!")
				"""
                p_A, p_B, rdm_A, rdm_B = self._p_mixed(
                    state, sub_sys_A, return_rdm=return_rdm, compute_signs=compute_signs
                )

            else:
                raise ValueError("state must have ndim < 4")

        if pure:
            p_A, p_B = p, p

        Sent_A, Sent_B = None, None
        if alpha == 1.0:
            if p_A is not None:
                Sent_A = -_np.nansum((p_A * _np.log(p_A)), axis=-1)
                if density:
                    Sent_A /= N_A
            if p_B is not None:
                Sent_B = -_np.nansum((p_B * _np.log(p_B)), axis=-1)
                if density:
                    Sent_B /= N_B
        elif alpha >= 0.0:
            if p_A is not None:
                Sent_A = _np.log(_np.nansum(_np.power(p_A, alpha), axis=-1)) / (
                    1.0 - alpha
                )
                if density:
                    Sent_A /= N_A
            if p_B is not None:
                Sent_B = _np.log(_np.nansum(_np.power(p_B, alpha), axis=-1)) / (
                    1.0 - alpha
                )
                if density:
                    Sent_B /= N_B
        else:
            raise ValueError("alpha >= 0")

        # initiate variables
        variables = ["Sent_A"]
        if return_rdm_EVs:
            variables.append("p_A")

        if return_rdm == "A":
            variables.append("rdm_A")

        elif return_rdm == "B":
            variables.extend(["Sent_B", "rdm_B"])
            if return_rdm_EVs:
                variables.append("p_B")

        elif return_rdm == "both":
            variables.extend(["rdm_A", "Sent_B", "rdm_B"])
            if return_rdm_EVs:
                variables.extend(["p_A", "p_B"])

        # store variables to dictionar
        return_dict = {}
        for i in variables:
            if locals()[i] is not None:
                if sparse and "rdm" in i:
                    return_dict[i] = locals()[i]  # don't squeeze sparse matrix
                else:
                    return_dict[i] = _np.squeeze(locals()[i])

        return return_dict

    ##### private methods

    def _p_pure_sparse(
        self, state, sub_sys_A, return_rdm=None, sparse_diag=True, maxiter=None
    ):  # calls _partial_trace

        partial_trace_args = dict(sub_sys_A=sub_sys_A, sparse=True, enforce_pure=True)

        N_A = len(sub_sys_A)
        N_B = self.N - N_A

        rdm_A = None
        rdm_B = None

        if return_rdm is None:
            if N_A <= N_B:
                partial_trace_args["return_rdm"] = "A"
                rdm = self._partial_trace(state, **partial_trace_args)
            else:
                partial_trace_args["return_rdm"] = "B"
                rdm = self._partial_trace(state, **partial_trace_args)

        elif return_rdm == "A" and N_A <= N_B:
            partial_trace_args["return_rdm"] = "A"
            rdm_A = self._partial_trace(state, **partial_trace_args)
            rdm = rdm_A

        elif return_rdm == "B" and N_B <= N_A:
            partial_trace_args["return_rdm"] = "B"
            rdm_B = self._partial_trace(state, **partial_trace_args)
            rdm = rdm_B

        else:
            partial_trace_args["return_rdm"] = "both"
            rdm_A, rdm_B = self._partial_trace(state, **partial_trace_args)

            if N_A < N_B:
                rdm = rdm_A
            else:
                rdm = rdm_B

        if sparse_diag and rdm.shape[0] > 16:

            def get_p_patchy(rdm):
                n = rdm.shape[0]
                p_LM = eigsh(
                    rdm,
                    k=n // 2 + n % 2,
                    which="LM",
                    maxiter=maxiter,
                    return_eigenvectors=False,
                )  # get upper half
                p_SM = eigsh(
                    rdm,
                    k=n // 2,
                    which="SM",
                    maxiter=maxiter,
                    return_eigenvectors=False,
                )  # get lower half
                p = _np.concatenate((p_LM[::-1], p_SM)) + _np.finfo(p_LM.dtype).eps
                return p

            if _sp.issparse(rdm):
                p = get_p_patchy(rdm)
                p = p.reshape((1, -1))
            else:
                p_gen = [get_p_patchy(dm) for dm in rdm[:]]
                p = _np.stack(p_gen)

        else:
            if _sp.issparse(rdm):
                p = eigvalsh(rdm.todense())[::-1] + _np.finfo(rdm.dtype).eps
                p = p.reshape((1, -1))
            else:
                p_gen = [
                    eigvalsh(dm.todense())[::-1] + _np.finfo(dm.dtype).eps
                    for dm in rdm[:]
                ]
                p = _np.stack(p_gen)

        return p, rdm_A, rdm_B

    def _p_pure(
        self,
        state,
        sub_sys_A,
        svd_solver=None,
        svd_kwargs=None,
        return_rdm=None,
        compute_signs=False,
    ):  # default is None

        if svd_kwargs is None:
            svd_kwargs = dict()

        # calculate full H-space representation of state
        state = self.project_from(state, sparse=False)

        if compute_signs:  # imprint fermion signs
            sign_array = self._ptrace_signs(sub_sys_A)
            state = (state.T * sign_array).T

        # put states in rows
        state = state.T
        # reshape state according to sub_sys_A
        v = _lattice_reshape_pure(state, sub_sys_A, self.N, self._sps)

        rdm_A = None
        rdm_B = None

        # perform SVD
        if return_rdm is None:
            if (svd_solver is None) or (svd_solver == _np.linalg.svd):
                lmbda = _np.linalg.svd(v, compute_uv=False)
            else:  # custom solver
                # preallocate
                lmbda = _np.zeros(v.shape[0:2], dtype=state.dtype)
                # loop over states
                for j in range(v.shape[0]):
                    lmbda[j, ...] = svd_solver(v[j, ...], **svd_kwargs)
        else:
            if (svd_solver is None) or (svd_solver == _np.linalg.svd):
                U, lmbda, V = _np.linalg.svd(v, full_matrices=False)
            else:  # custom solver
                # preallocate
                lmbda = _np.zeros(v.shape[0:2], dtype=state.dtype)
                U = _np.zeros(v.shape, dtype=state.dtype)
                V = _np.zeros_like(U)
                # loop over states
                for j in range(v.shape[0]):
                    U[j, ...], lmbda[j, ...], V[j, ...] = svd_solver(
                        v[j, ...], **svd_kwargs
                    )

            if return_rdm == "A":
                rdm_A = _np.einsum("...ij,...j,...kj->...ik", U, lmbda**2, U.conj())
            elif return_rdm == "B":
                rdm_B = _np.einsum("...ji,...j,...jk->...ik", V.conj(), lmbda**2, V)
            elif return_rdm == "both":
                rdm_A = _np.einsum("...ij,...j,...kj->...ik", U, lmbda**2, U.conj())
                rdm_B = _np.einsum("...ji,...j,...jk->...ik", V.conj(), lmbda**2, V)

        return lmbda**2 + _np.finfo(lmbda.dtype).eps, rdm_A, rdm_B

    def _p_mixed(
        self,
        state,
        sub_sys_A,
        return_rdm=None,
        compute_signs=False,
    ):
        """
        This function calculates the eigenvalues of the reduced density matrix.
        It will first calculate the partial trace of the full density matrix and
        then diagonalizes it to get the eigenvalues. It will automatically choose
        the subsystem with the smaller hilbert space to do the diagonalization in order
        to reduce the calculation time but will only return the desired reduced density
        matrix.
        """
        N = self.N
        sps = self.sps

        N_A = len(sub_sys_A)
        N_B = N - N_A

        proj = self.get_proj(_dtypes[state.dtype.char])
        state = state.transpose((2, 0, 1))

        Ns_full = proj.shape[0]
        n_states = state.shape[0]

        if compute_signs:  # imprint fermion signs
            sign_array = self._ptrace_signs(sub_sys_A)
            signs_mask = _np.outer(sign_array, sign_array)
            gen = (_np.multiply(proj * s * proj.T.conj(), signs_mask) for s in state[:])
        else:
            gen = (proj * s * proj.T.conj() for s in state[:])

        proj_state = _np.zeros(
            (n_states, Ns_full, Ns_full), dtype=_dtypes[state.dtype.char]
        )

        for i, s in enumerate(gen):
            proj_state[i, ...] += s[...]

        rdm_A, p_A = None, None
        rdm_B, p_B = None, None

        if return_rdm == "both":
            rdm_A, rdm_B = _lattice_partial_trace_mixed(
                proj_state, sub_sys_A, N, sps, return_rdm="both"
            )

            p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
            p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

        elif return_rdm == "A":
            rdm_A, rdm_B = _lattice_partial_trace_mixed(
                proj_state, sub_sys_A, N, sps, return_rdm="A"
            )
            p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps

        elif return_rdm == "B":
            rdm_A, rdm_B = _lattice_partial_trace_mixed(
                proj_state, sub_sys_A, N, sps, return_rdm="B"
            )
            p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

        else:
            rdm_A, rdm_B = _lattice_partial_trace_mixed(
                proj_state, sub_sys_A, N, sps, return_rdm="A"
            )
            p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps

        return p_A, p_B, rdm_A, rdm_B

    def _get__str__(self):
        from quspin.basis.basis_general import basis_int_to_python_int

        Ns = self._Ns
        if Ns > MAXPRINT:
            half = MAXPRINT // 2

            basis_iter = chain(
                zip(range(half), self.states[:half]),
                zip(range(Ns - half, Ns, 1), self.states[-half:]),
            )

        else:
            basis_iter = enumerate(self.states)

        str_list = []

        if self.states.dtype != int:
            convert_int = lambda i: basis_int_to_python_int(i)
        else:
            convert_int = lambda i: I

        b_int = convert_int(self.states[0])
        b_str = self.int_to_state(self.states[0])
        len_Ns = len(str(self.Ns))
        len_b_str = len(b_str)
        len_b_int = len(str(b_int))
        temp = " {{:{:d}d}}.  {{:{:d}s}}  {{:{:d}d}} ".format(
            len_Ns, len_b_str, len_b_int
        )

        for i, b in basis_iter:
            b_int = convert_int(b)
            b_str = self.int_to_state(b)
            str_list.append(temp.format(i, b_str, b_int))

        return tuple(str_list)
