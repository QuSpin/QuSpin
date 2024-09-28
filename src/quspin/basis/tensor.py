from __future__ import print_function
from quspin.basis.base import basis, MAXPRINT

import numpy as _np
from scipy import sparse as _sp
from scipy.sparse import linalg as _sla
from scipy import linalg as _la
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigvalsh, svd
from quspin.basis._reshape_subsys import (
    _tensor_reshape_pure,
    _tensor_partial_trace_pure,
)
from quspin.basis._reshape_subsys import (
    _tensor_partial_trace_mixed,
    _tensor_partial_trace_sparse_pure,
)
import warnings

_dtypes = {"f": _np.float32, "d": _np.float64, "F": _np.complex64, "D": _np.complex128}

__all__ = ["tensor_basis"]


# gives the basis for the kronecker/Tensor product of two basis: |basis_left> (x) |basis_right>
class tensor_basis(basis):
    """Constructs basis in tensor product Hilbert space.

    The `tensor_basis` class combines two basis objects `basis1` and `basis2` together into a new basis object which can be then used, e.g., to create the Hamiltonian over the tensor product Hilbert space:

    .. math::
            \\mathcal{H}=\\mathcal{H}_1\\otimes\\mathcal{H}_2

    Notes
    -----

    * The `tensor_basis` operator strings are separated by a pipe symbol, "|". However, the index array has NO pipe symbol.

    * If two fermion basis constructors are used, `tensor_basis` will assume that the two fermion species are distinguishable, i.e. their operators will commute (instead of anti-commute). For anticommuting fermion species, use the `spinful_fermion_basis_*` constructors.

    The `tensor_basis` class does not allow one to make use of symmetries, save for particle conservation.


    Examples
    --------
    The following code shows how to construct the Fermi-Hubbard Hamiltonian by tensoring two
    `spinless_fermion_basis_1d` objects. This model can also be set up using the `spinful_fermion_basis_1d` class),
    which also allows the implementation of symmetries.

    Notice that the operator strings for constructing Hamiltonians with a `tensor_basis` object are separated by
    a pipe symbol, '|', while the index array has no splitting pipe character.

    The code snippet below initiates the class, and is required to run the example codes for the function methods.

    .. literalinclude:: ../../doc_examples/tensor_basis-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(self, *basis_list):
        """Initialises the `tensor_basis` object (basis for tensor product Hilbert spaces).

        Parameters
        ----------
        basis_list : list[:obj:`basis`]
                List of `basis` objects to tensor together. Required minimum number is two.

        """
        if len(basis_list) < 2:
            raise ValueError("basis_list must contain at least 2 basis objects.")
        if not isinstance(basis_list[0], basis):
            raise ValueError("basis_list must contain instances of basis class")

        self._check_herm = True
        fermion_list = []
        for b in basis_list:
            try:
                is_fermion = b._fermion_basis
                is_pcon = not ((b._check_pcon is None) or (not basis._check_pcon))
                fermion_list.append(is_fermion and is_pcon)
            except:
                pass

        if len(fermion_list) > 1 and not all(fermion_list):
            warnings.warn(
                "Tensor basis does not handle more than one non-particle conserving fermion basis objects because of the fermionic sign."
            )

        self._basis_left = basis_list[0]
        if len(basis_list) == 2:
            self._basis_right = basis_list[1]
        else:
            self._basis_right = tensor_basis(*basis_list[1:])

        self._Ns = self._basis_left.Ns * self._basis_right.Ns
        self._dtype = _np.min_scalar_type(-self._Ns)

        self._blocks = self._basis_left._blocks.copy()
        self._blocks.update(self._basis_right._blocks)

        self._check_symm = None
        self._check_pcon = (
            None  # update once check_pcon is implemented for tensor_basis
        )

        self._unique_me = self._basis_left._unique_me and self._basis_right._unique_me
        self._operators = (
            self._basis_left._operators + "\n" + self._basis_right._operators
        )

    @property
    def basis_left(self):
        """:obj:`basis`: first basis constructor out of the `basis` objects list to be tensored."""
        return self._basis_left

    @property
    def basis_right(self):
        """:obj:`basis`: all others basis constructors except for the first one of the `basis` objects list to be tensored."""
        return self._basis_right

    @property
    def N(self):
        """tuple: the value of `N` attribute from all the basis objects tensored together in a tuple ordered according to the input basis list."""
        if not isinstance(self._basis_right, tensor_basis):
            return (self._basis_left.N,) + (self._basis_right.N,)
        else:
            return (self._basis_left.N,) + self._basis_right.N

    def Op(self, opstr, indx, J, dtype):
        """Constructs operator from a site-coupling list and an operator string in the tensor basis.

        Parameters
        ----------
        opstr : str
                Operator string in the tensor basis format. For instance:
                >>> opstr = "z|z"
        indx : list(int)
                List of integers to designate the sites the tensor basis operator is defined on. For instance:
                >>> indx = [1,5]
        J : scalar
                Coupling strength.
        dtype : 'type'
                Data type (e.g. numpy.float64) to construct the operator with.

        Returns
        -------
        tuple
                `(ME,row,col)`, where
                        * numpy.ndarray(scalar): `ME`: matrix elements of type `dtype`.
                        * numpy.ndarray(int): `row`: row indices of matrix representing the operator in the tensor basis,
                                such that `row[i]` is the row index of `ME[i]`.
                        * numpy.ndarray(int): `col`: column index of matrix representing the operator in the tensor basis,
                                such that `col[i]` is the column index of `ME[i]`.

        Examples
        --------

        >>> J = 1.41
        >>> indx = [1,5]
        >>> opstr = "z|z"
        >>> dtype = np.float64
        >>> ME, row, col = Op(opstr,indx,J,dtype)

        """

        # if opstr.count("|") > 1:
        # 	raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

        if len(opstr) - opstr.count("|") != len(indx):
            raise ValueError(
                "not enough indices for opstr in: {0}, {1}".format(opstr, indx)
            )

        i = opstr.index("|")
        indx_left = indx[:i]
        indx_right = indx[i:]

        opstr_left, opstr_right = opstr.split("|", 1)

        if self._basis_left._Ns < self._basis_right._Ns:
            ME_left, row_left, col_left = self._basis_left.Op(
                opstr_left, indx_left, J, dtype
            )
            ME_right, row_right, col_right = self._basis_right.Op(
                opstr_right, indx_right, 1.0, dtype
            )
        else:
            ME_left, row_left, col_left = self._basis_left.Op(
                opstr_left, indx_left, 1.0, dtype
            )
            ME_right, row_right, col_right = self._basis_right.Op(
                opstr_right, indx_right, J, dtype
            )

        n1 = row_left.shape[0]
        n2 = row_right.shape[0]

        if n1 > 0 and n2 > 0:
            row_left = row_left.astype(self._dtype, copy=False)
            row_right = row_right.astype(self._dtype, copy=False)
            row_left *= self._basis_right.Ns

            row = _np.kron(row_left, _np.ones_like(row_right, dtype=_np.int8))
            row += _np.kron(_np.ones_like(row_left, dtype=_np.int8), row_right)

            del row_left, row_right

            col_left = col_left.astype(self._dtype, copy=False)
            col_right = col_right.astype(self._dtype, copy=False)
            col_left *= self._basis_right.Ns
            col = _np.kron(col_left, _np.ones_like(col_right, dtype=_np.int8))
            col += _np.kron(_np.ones_like(col_left, dtype=_np.int8), col_right)

            del col_left, col_right

            ME = _np.kron(ME_left, ME_right)

            del ME_left, ME_right
        else:
            row = _np.array([])
            col = _np.array([])
            ME = _np.array([])

        return ME, row, col

    def index(self, *states):
        """Finds the index of user-defined Fock state in tensor basis.

        Notes
        -----
        Particularly useful for defining initial Fock states through a unit vector in the direction specified
        by `index()`.

        Parameters
        ----------
        states : list(str)
                List of strings which separately define the Fock state in each of the `basis` used to construct
                the `tensor_basis` object.

        Returns
        -------
        int
                Position of tensor Fock state in the `tensor_basis`.

        Examples
        --------

        >>> s_1 = "".join("1" for i in range(2)) + "".join("0" for i in range(2))
        >>> s_2 = "".join("1" for i in range(4))
        >>> print( basis.index(s_1,s_2) )

        """
        if len(states) < 2:
            raise ValueError("states must be list of atleast 2 elements long")
        s_left = self.basis_left.index(states[0])
        s_right = self.basis_right.index(*states[1:])
        return s_right + self.basis_right.Ns * s_left

    def get_vec(self, v0, sparse=True, full_left=True, full_right=True):
        """DEPRECATED (cf `project_from`). Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        This function is :red:`deprecated`. Use `project_from()` instead (the inverse function, `project_to()`, is currently available in the `basis_general` classes only).

        """

        return self.project_from(
            v0, sparse=sparse, full_left=full_left, full_right=full_right
        )

    def project_from(self, v0, sparse=True, full_left=True, full_right=True):
        """Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
        in a straightforward manner.

        Supports parallelisation to multiple states listed in the columns.

        Parameters
        ----------
        v0 : numpy.ndarray
                Contains in its columns the states in the symmetry-reduced basis.
        sparse : bool, optional
                Whether or not the output should be in sparse format. Default is `True`.
        full_left : bool, optional
                Whether or not to transform the state to the full state in `basis_left`. Default is `True`.
        full_right : bool, optional
                Whether or not to transform the state to the full state in `basis_right`. Default is `True`.

        Returns
        -------
        numpy.ndarray
                Array containing the state `v0` in the full basis.

        Examples
        --------

        >>> v_full = project_from(v0)
        >>> print(v_full.shape, v0.shape)

        """

        if self._Ns <= 0:
            return _np.array([])

        if not hasattr(v0, "shape"):
            v0 = _np.asanyarray(v0)

        if v0.shape[0] != self._Ns:
            raise ValueError("v0 has incompatible dimensions with basis")

        if v0.ndim == 1:
            v0 = v0.reshape((-1, 1))
            if sparse:
                return _combine_project_froms(self, v0, sparse, full_left, full_right)
            else:
                return _combine_project_froms(
                    self, v0, sparse, full_left, full_right
                ).reshape((-1,))
        elif v0.ndim == 2:

            if _sp.issparse(v0):
                return self.get_proj(
                    v0.dtype, full_left=full_left, full_right=full_right
                ).dot(v0)

            return _combine_project_froms(self, v0, sparse, full_left, full_right)
        else:
            raise ValueError("excpecting v0 to have ndim at most 2")

    def get_proj(self, dtype, full_left=True, full_right=True):
        """Calculates transformation/projector from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
        in a straightforward manner.

        Parameters
        ----------
        dtype : 'type'
                Data type (e.g. numpy.float64) to construct the projector with.
        full_left : bool, optional
                Whether or not to transform the state to the full state in `basis_left`. Default is `True`.
        full_right : bool, optional
                Whether or not to transform the state to the full state in `basis_right`. Default is `True`.

        Returns
        -------
        numpy.ndarray
                Transformation/projector between the symmetry-reduced and the full basis.

        Examples
        --------

        >>> P = get_proj(np.float64)
        >>> print(P.shape)

        """

        if full_left:
            proj1 = self._basis_left.get_proj(dtype)
        else:
            proj1 = _sp.identity(self._basis_left.Ns, dtype=dtype)

        if full_right:
            proj2 = self._basis_right.get_proj(dtype)
        else:
            proj2 = _sp.identity(self._basis_right.Ns, dtype=dtype)

        return _sp.kron(proj1, proj2, format="csr")

    def partial_trace(
        self, state, sub_sys_A="left", return_rdm="A", enforce_pure=False, sparse=False
    ):
        """Calculates reduced density matrix, through a partial trace of a quantum state in `tensor_basis`.

        Parameters
        ----------
        state : obj
                State of the quantum system. Can be either one of:

                        * numpy.ndarray [shape (Ns,)]: pure state (default).
                        * numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
        sub_sys_A : str, optional
                Defines subsystem A. Can be either one of:

                        * "left": refers to `basis_left` (Default).
                        * "right": refers to `basis_right`.
                        * "both": for initial mixed states the Renyi entropy of subsystem A and its complement
                                B need not be the same. This option automatically sets `return_rdm=both`.

        return_rdm : str, required
                Toggles returning the reduced DM. Can be either one of:

                        * "A": returns reduced DM of subsystem A.
                        * "B": returns reduced DM of subsystem B (complement of A).
                        * "both": returns reduced DM of both subsystems A and B.
        enforce_pure : bool, optional
                Whether or not to assume `state` is a collection of pure states or a mixed density matrix, if
                it is a square array. Default is `False`.
        sparse : bool, optional
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
            sub_sys_A = "left"

        if return_rdm not in set(["A", "B", "both", None]):
            raise ValueError("return_rdm must be: 'A','B','both' or None")

        if sub_sys_A not in set(["left", "right", "both", None]):
            raise ValueError("sub_sys_A must be 'left' or 'right' or 'both'.")

        if not hasattr(state, "shape"):
            state = _np.asanyarray(state)
            state = state.squeeze()  # avoids artificial higher-dim reps of ndarray

        Ns_left = self._basis_left.Ns
        Ns_right = self._basis_right.Ns
        tensor_Ns = Ns_left * Ns_right

        if state.shape[0] != tensor_Ns:
            raise ValueError(
                "state shape {0} not compatible with Ns={1}".format(
                    state.shape, tensor_Ns
                )
            )

        if _sp.issparse(state) or sparse:
            if not _sp.issparse(state):
                state = _sp.csr_matrix(state)

            state = state.T
            if state.shape[0] == 1:
                # sparse_pure partial trace
                rdm_A, rdm_B = _tensor_partial_trace_sparse_pure(
                    state, sub_sys_A, Ns_left, Ns_right, return_rdm=return_rdm
                )
            else:
                if state.shape[0] != state.shape[1] or enforce_pure:
                    # vectorize sparse_pure partial trace
                    state = state.tocsr()
                    try:
                        state_gen = (
                            _tensor_partial_trace_sparse_pure(
                                state.getrow(i),
                                sub_sys_A,
                                Ns_left,
                                Ns_right,
                                return_rdm=return_rdm,
                            )
                            for i in xrange(state.shape[0])
                        )
                    except NameError:
                        state_gen = (
                            _tensor_partial_trace_sparse_pure(
                                state.getrow(i),
                                sub_sys_A,
                                Ns_left,
                                Ns_right,
                                return_rdm=return_rdm,
                            )
                            for i in range(state.shape[0])
                        )

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
                rdm_A, rdm_B = _tensor_partial_trace_pure(
                    state.T, sub_sys_A, Ns_left, Ns_right, return_rdm=return_rdm
                )

            elif state.ndim == 2:
                if state.shape[0] != state.shape[1] or enforce_pure:
                    rdm_A, rdm_B = _tensor_partial_trace_pure(
                        state.T, sub_sys_A, Ns_left, Ns_right, return_rdm=return_rdm
                    )

                else:
                    shape0 = state.shape
                    state = state.reshape((1,) + shape0)

                    rdm_A, rdm_B = _tensor_partial_trace_mixed(
                        state, sub_sys_A, Ns_left, Ns_right, return_rdm=return_rdm
                    )

            elif state.ndim == 3:  # 3D DM
                rdm_A, rdm_B = _tensor_partial_trace_mixed(
                    state, sub_sys_A, Ns_left, Ns_right, return_rdm=return_rdm
                )
            else:
                raise ValueError("state must have ndim < 4")

        if return_rdm == "A":
            return rdm_A
        elif return_rdm == "B":
            return rdm_B
        else:
            return rdm_A, rdm_B

    def ent_entropy(
        self,
        state,
        sub_sys_A="left",
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
                S_\\mathrm{ent}(\\alpha) = \\frac{1}{1-\\alpha}\\log \\mathrm{tr}_{A} \\left( \\mathrm{tr}_{A^c} \\vert\\psi\\rangle\\langle\\psi\\vert \\right)^\\alpha

        **Note:** The logarithm used is the natural logarithm (base e).

        Notes
        -----
        Algorithm is based on both partial tracing and sigular value decomposition (SVD), optimised for speed.

        Parameters
        ----------
        state : obj
                State of the quantum system. Can be either one of:

                        * numpy.ndarray [shape (Ns,)]: pure state (default).
                        * numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
        sub_sys_A : str, optional
                Defines subsystem A. Can be either one of:

                        * "left": refers to `basis_left` (Default).
                        * "right": refers to `basis_right`.
                        * "both": for initial mixed states the Renyi entropy of subsystem A and its complement
                                B need not be the same. This option automatically sets `return_rdm=both`.
        return_rdm : str, optional
                Toggles returning the reduced DM. Can be tierh one of:

                        * "A": returns reduced DM of subsystem A.
                        * "B": returns reduced DM of subsystem B (complement of A).
                        * "both": returns reduced DM of both subsystems A and B.
        enforce_pure : bool, optional
                Whether or not to assume `state` is a collection of pure states or a mixed density matrix, if
                it is a square array. Default is `False`.
        sparse : bool, optional
                Whether or not to return a sparse DM. Default is `False`.
        return_rdm_EVs : bool, optional
                Whether or not to return the eigenvalues of rthe educed DM. If `return_rdm` is specified,
                the eigenvalues of the corresponding DM are returned. If `return_rdm` is NOT specified,
                the spectrum of `rdm_A` is returned by default. Default is `False`.
        alpha : float, optional
                Renyi :math:`\\alpha` parameter for the entanglement entropy. Default is :math:`\\alpha=1`.
        sparse_diag : bool, optional
                When `sparse=True`, this flag enforces the use of
                `scipy.sparse.linalg.eigsh() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_
                to calculate the eigenvaues of the reduced DM.
        maxiter : int, optional
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

        >>> ent_entropy(state,sub_sys_A="left",return_rdm="A",enforce_pure=False,return_rdm_EVs=False,
        >>>				sparse=False,alpha=1.0,sparse_diag=True)

        """

        if sub_sys_A is None:
            sub_sys_A = "left"

        if return_rdm not in set(["A", "B", "both", None]):
            raise ValueError("return_rdm must be: 'A','B','both' or None")

        if sub_sys_A not in set(["left", "right", "both"]):
            raise ValueError("sub_sys_A must be 'left' or 'right' or 'both'.")

        if not hasattr(state, "shape"):
            state = _np.asanyarray(state)
            state = state.squeeze()  # avoids artificial higher-dim reps of ndarray

        tensor_Ns = self._basis_left.Ns * self._basis_right.Ns

        if state.shape[0] != tensor_Ns:
            raise ValueError(
                "state shape {0} not compatible with Ns={1}".format(
                    state.shape, tensor_Ns
                )
            )

        pure = True  # set pure state parameter to True
        if _sp.issparse(state) or sparse:

            if not _sp.issparse(state):
                if state.ndim == 1:
                    state = _sp.csr_matrix(state).T
                else:
                    state = _sp.csr_matrix(state)

            if state.shape[1] == 1:
                p, rdm_A, rdm_B = self._p_pure_sparse(
                    state,
                    sub_sys_A,
                    return_rdm=return_rdm,
                    sparse_diag=sparse_diag,
                    maxiter=maxiter,
                )
            else:
                if state.shape[0] != state.shape[1] or enforce_pure:
                    p, rdm_A, rdm_B = self._p_pure_sparse(
                        state, sub_sys_A, return_rdm=return_rdm
                    )
                else:
                    raise ValueError("Expecting a dense array for mixed states.")

        else:

            if state.ndim == 1:
                state = state.reshape((-1, 1))
                p, rdm_A, rdm_B = self._p_pure(state, sub_sys_A, return_rdm=return_rdm)

            elif state.ndim == 2:

                if state.shape[0] != state.shape[1] or enforce_pure:
                    p, rdm_A, rdm_B = self._p_pure(
                        state, sub_sys_A, return_rdm=return_rdm
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
                        state, sub_sys_A, return_rdm=return_rdm
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
                    state, sub_sys_A, return_rdm=return_rdm
                )

            else:
                raise ValueError("state must have ndim < 4")

        if pure:
            p_A, p_B = p, p

        Sent_A, Sent_B = None, None
        if alpha == 1.0:
            if p_A is not None:
                Sent_A = -_np.nansum(p_A * _np.log(p_A), axis=-1)
            if p_B is not None:
                Sent_B = -_np.nansum(p_B * _np.log(p_B), axis=-1)
        elif alpha >= 0.0:
            if p_A is not None:
                Sent_A = _np.log(
                    _np.nansum(_np.power(p_A, alpha), axis=-1) / (1.0 - alpha)
                )
            if p_B is not None:
                Sent_B = _np.log(
                    _np.nansum(_np.power(p_B, alpha), axis=-1) / (1.0 - alpha)
                )
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

    def _p_pure(self, state, sub_sys_A, return_rdm=None):

        # put states in rows
        state = state.T
        # reshape state according to sub_sys_A
        Ns_left = self._basis_left.Ns
        Ns_right = self._basis_right.Ns
        v = _tensor_reshape_pure(state, sub_sys_A, Ns_left, Ns_right)

        rdm_A = None
        rdm_B = None

        # perform SVD
        if return_rdm is None:
            lmbda = svd(v, compute_uv=False)
        else:
            U, lmbda, V = svd(v, full_matrices=False)
            if return_rdm == "A":
                rdm_A = _np.einsum("...ij,...j,...kj->...ik", U, lmbda**2, U.conj())
            elif return_rdm == "B":
                rdm_B = _np.einsum("...ji,...j,...jk->...ik", V.conj(), lmbda**2, V)
            elif return_rdm == "both":
                rdm_A = _np.einsum("...ij,...j,...kj->...ik", U, lmbda**2, U.conj())
                rdm_B = _np.einsum("...ji,...j,...jk->...ik", V.conj(), lmbda**2, V)

        return (lmbda**2) + _np.finfo(lmbda.dtype).eps, rdm_A, rdm_B

    def _p_pure_sparse(
        self, state, sub_sys_A, return_rdm=None, sparse_diag=True, maxiter=None
    ):

        partial_trace_args = dict(sub_sys_A=sub_sys_A, sparse=True, enforce_pure=True)

        if sub_sys_A == "left":
            Ns_A = self._basis_left.Ns
            Ns_B = self._basis_right.Ns
        else:
            Ns_A = self._basis_right.Ns
            Ns_B = self._basis_left.Ns

        rdm_A = None
        rdm_B = None

        if return_rdm is None:
            if Ns_A <= Ns_B:
                partial_trace_args["return_rdm"] = "A"
                rdm = tensor_basis.partial_trace(self, state, **partial_trace_args)
            else:
                partial_trace_args["return_rdm"] = "B"
                rdm = tensor_basis.partial_trace(self, state, **partial_trace_args)

        elif return_rdm == "A" and Ns_A <= Ns_B:
            partial_trace_args["return_rdm"] = "A"
            rdm_A = tensor_basis.partial_trace(self, state, **partial_trace_args)
            rdm = rdm_A

        elif return_rdm == "B" and Ns_B <= Ns_A:
            partial_trace_args["return_rdm"] = "B"
            rdm_B = tensor_basis.partial_trace(self, state, **partial_trace_args)
            rdm = rdm_B

        else:
            partial_trace_args["return_rdm"] = "both"
            rdm_A, rdm_B = tensor_basis.partial_trace(self, state, **partial_trace_args)

            if Ns_A <= Ns_B:
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
                p_gen = (get_p_patchy(dm) for dm in rdm[:])
                p = _np.stack(p_gen)

        else:
            if _sp.issparse(rdm):
                p = eigvalsh(rdm.todense())[::-1] + _np.finfo(rdm.dtype).eps
                p = p.reshape((1, -1))
            else:
                p_gen = (
                    eigvalsh(dm.todense())[::-1] + _np.finfo(dm.dtype).eps
                    for dm in rdm[:]
                )
                p = _np.stack(p_gen)

        return p, rdm_A, rdm_B

    def _p_mixed(self, state, sub_sys_A, return_rdm=None):
        """
        This function calculates the eigenvalues of the reduced density matrix.
        It will first calculate the partial trace of the full density matrix and
        then diagonalizes it to get the eigenvalues. It will automatically choose
        the subsystem with the smaller hilbert space to do the diagonalization in order
        to reduce the calculation time but will only return the desired reduced density
        matrix.
        """

        state = state.transpose((2, 0, 1))

        Ns_left = self._basis_left.Ns
        Ns_right = self._basis_right.Ns

        rdm_A, p_A = None, None
        rdm_B, p_B = None, None

        if return_rdm == "both":
            rdm_A, rdm_B = _tensor_partial_trace_mixed(
                state, sub_sys_A, Ns_left, Ns_right, return_rdm="both"
            )

            p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
            p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

        elif return_rdm == "A":
            rdm_A, rdm_B = _tensor_partial_trace_mixed(
                state, sub_sys_A, Ns_left, Ns_right, return_rdm="A"
            )
            p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps

        elif return_rdm == "B":
            rdm_A, rdm_B = _tensor_partial_trace_mixed(
                state, sub_sys_A, Ns_left, Ns_right, return_rdm="B"
            )
            p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

        else:
            rdm_A, rdm_B = _tensor_partial_trace_mixed(
                state, sub_sys_A, Ns_left, Ns_right, return_rdm="A"
            )
            p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps

        return p_A, p_B, rdm_A, rdm_B

    def __name__(self):
        return "<type 'qspin.basis.tensor_basis'>"

    def _sort_opstr(self, op):
        op = list(op)
        opstr = op[0]
        indx = op[1]

        if opstr.count("|") == 0:
            raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr, indx))

        # if opstr.count("|") > 1:
        # 	raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

        if len(opstr) - opstr.count("|") != len(indx):
            raise ValueError(
                "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
            )

        i = opstr.index("|")
        indx_left = indx[:i]
        indx_right = indx[i:]

        opstr_left, opstr_right = opstr.split("|", 1)

        op1 = list(op)
        op1[0] = opstr_left
        op1[1] = tuple(indx_left)

        op2 = list(op)
        op2[0] = opstr_right
        op2[1] = tuple(indx_right)

        op1 = self._basis_left._sort_opstr(op1)
        op2 = self._basis_right._sort_opstr(op2)

        op[0] = "|".join((op1[0], op2[0]))
        op[1] = op1[1] + op2[1]

        return tuple(op)

    def _hc_opstr(self, op):
        op = list(op)
        opstr = op[0]
        indx = op[1]

        # if opstr.count("|") > 1:
        # 	raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

        if len(opstr) - opstr.count("|") != len(indx):
            raise ValueError(
                "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
            )

        i = opstr.index("|")
        indx_left = indx[:i]
        indx_right = indx[i:]

        opstr_left, opstr_right = opstr.split("|", 1)

        op1 = list(op)
        op1[0] = opstr_left
        op1[1] = indx_left
        op1[2] = op[2]

        op2 = list(op)
        op2[0] = opstr_right
        op2[1] = indx_right
        op2[2] = complex(1.0)

        op1 = self._basis_left._hc_opstr(op1)
        op2 = self._basis_right._hc_opstr(op2)

        op[0] = "|".join((op1[0], op2[0]))
        op[1] = op1[1] + op2[1]

        op[2] = op1[2] * op2[2]

        return tuple(op)

    def _non_zero(self, op):
        op = list(op)
        opstr = op[0]
        indx = op[1]

        # if opstr.count("|") > 1:
        # 	raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

        if len(opstr) - opstr.count("|") != len(indx):
            raise ValueError(
                "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
            )

        i = opstr.index("|")
        indx_left = indx[:i]
        indx_right = indx[i:]

        opstr_left, opstr_right = opstr.split("|", 1)

        op1 = list(op)
        op1[0] = opstr_left
        op1[1] = indx_left

        op2 = list(op)
        op2[0] = opstr_right
        op2[1] = indx_right

        return self._basis_left._non_zero(op1) and self._basis_right._non_zero(op2)

    def _expand_opstr(self, op, num):
        op = list(op)
        opstr = op[0]
        indx = op[1]

        # if opstr.count("|") > 1:
        # 	raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

        if len(opstr) - opstr.count("|") != len(indx):
            raise ValueError(
                "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
            )

        i = opstr.index("|")
        indx_left = indx[:i]
        indx_right = indx[i:]

        opstr_left, opstr_right = opstr.split("|", 1)

        op1 = list(op)
        op1[0] = opstr_left
        op1[1] = indx_left
        op1[2] = 1.0

        op2 = list(op)
        op2[0] = opstr_right
        op2[1] = indx_right

        op1_list = self._basis_left._expand_opstr(op1, num)
        op2_list = self._basis_right._expand_opstr(op2, num)

        op_list = []
        for new_op1 in op1_list:
            for new_op2 in op2_list:
                new_op = list(new_op1)
                new_op[0] = "|".join((new_op1[0], new_op2[0]))
                new_op[1] += tuple(new_op2[1])
                new_op[2] *= new_op2[2]

                op_list.append(tuple(new_op))

        return tuple(op_list)

    def _get__str__(self):
        if not hasattr(self._basis_left, "_get__str__"):
            warnings.warn(
                "basis class {0} missing _get__str__ function, can not print out basis representatives.".format(
                    type(self._basis_left)
                ),
                UserWarning,
                stacklevel=3,
            )
            return "reference states: \n\t not availible"

        if not hasattr(self._basis_right, "_get__str__"):
            warnings.warn(
                "basis class {0} missing _get__str__ function, can not print out basis representatives.".format(
                    type(self._basis_right)
                ),
                UserWarning,
                stacklevel=3,
            )
            return "reference states: \n\t not availible"

        n_digits = int(_np.ceil(_np.log10(self._Ns)))

        str_list_1 = self._basis_left._get__str__()
        str_list_2 = self._basis_right._get__str__()
        Ns2 = self._basis_right.Ns
        temp = "\t{0:" + str(n_digits) + "d}.  "
        str_list = []
        for basis_left in str_list_1:
            basis_left, s1 = basis_left.split(".  ")
            i1 = int(basis_left)
            for basis_right in str_list_2:
                basis_right, s2 = basis_right.split(".  ")
                i2 = int(basis_right)
                str_list.append((temp.format(i2 + Ns2 * i1)) + s1 + s2)

        if self._Ns > MAXPRINT:
            half = MAXPRINT // 2
            str_list_1 = str_list[:half]
            str_list_2 = str_list[-half:]

            str_list = str_list_1
            str_list.extend(str_list_2)

        return str_list


def _combine_project_froms(basis, v0, sparse, full_left, full_right):
    Ns1 = basis._basis_left.Ns
    Ns2 = basis._basis_right.Ns

    Nvecs = v0.shape[1]

    Ns = min(Ns1, Ns2)

    # reshape vector to matrix to rewrite vector as an outer product.
    v0 = v0.T.reshape((Nvecs, Ns1, Ns2))
    # take singular value decomposition to get which decomposes the matrix into separate parts.
    # the outer/tensor product of the cols of V1 and V2 are the product states which make up the original vector

    V1, S, V2 = _np.linalg.svd(v0, full_matrices=False)
    S = S.T
    V1 = V1.transpose((2, 1, 0))
    V2 = V2.transpose((1, 2, 0))

    # combining all the vectors together with the tensor product as opposed to the outer product
    if sparse:
        # take the vectors and convert them to their full hilbert space
        v1 = V1[-1]
        v2 = V2[-1]

        if full_left:
            v1 = basis._basis_left.project_from(v1, sparse=True)

        if full_right:
            try:
                v2 = basis._basis_right.project_from(
                    v2, sparse=True, full_left=True, full_right=True
                )
            except TypeError:
                v2 = basis._basis_right.project_from(v2, sparse=True)

        temp1 = _np.ones((v1.shape[0], 1), dtype=_np.int8)
        temp2 = _np.ones((v2.shape[0], 1), dtype=_np.int8)

        v1 = _sp.kron(v1, temp2, format="csr")
        v2 = _sp.kron(temp1, v2, format="csr")

        s = _np.array(S[-1])
        s = _np.broadcast_to(s, v1.shape)

        v0 = v1.multiply(v2).multiply(s)

        for i, s in enumerate(S[:-1]):
            v1 = V1[i]
            v2 = V2[i]

            if full_left:
                v1 = basis._basis_left.project_from(v1, sparse=True)

            if full_right:
                try:
                    v2 = basis._basis_right.project_from(
                        v2, sparse=True, full_left=True, full_right=True
                    )
                except TypeError:
                    v2 = basis._basis_right.project_from(v2, sparse=True)

            v1 = _sp.kron(v1, temp2, format="csr")
            v2 = _sp.kron(temp1, v2, format="csr")

            s = _np.broadcast_to(s, v1.shape)
            v = v1.multiply(v2).multiply(s)

            v0 = v0 + v
    else:
        # take the vectors and convert them to their full hilbert space
        v1 = V1[-1]
        v2 = V2[-1]

        if full_left:
            v1 = basis._basis_left.project_from(v1, sparse=False)

        if full_right:
            try:
                v2 = basis._basis_right.project_from(
                    v2, sparse=False, full_left=True, full_right=True
                )
            except TypeError:
                v2 = basis._basis_right.project_from(v2, sparse=False)

        temp1 = _np.ones((v1.shape[0], 1), dtype=_np.int8)
        temp2 = _np.ones((v2.shape[0], 1), dtype=_np.int8)

        v1 = _np.kron(v1, temp2)
        v2 = _np.kron(temp1, v2)
        v0 = _np.multiply(v1, v2)
        v0 *= S[-1]

        for i, s in enumerate(S[:-1]):
            v1 = V1[i]
            v2 = V2[i]

            if full_left:
                v1 = basis._basis_left.project_from(v1, sparse=False)

            if full_right:
                try:
                    v2 = basis._basis_right.project_from(
                        v2, sparse=False, full_left=True, full_right=True
                    )
                except TypeError:
                    v2 = basis._basis_right.project_from(v2, sparse=False)

            v1 = _np.kron(v1, temp2)
            v2 = _np.kron(temp1, v2)
            v = _np.multiply(v1, v2)
            v0 += s * v

    return v0


"""
def _combine_get_vec(basis,v0,sparse,full_left,full_right):
	Ns1=basis._basis_left.Ns
	Ns2=basis._basis_right.Ns

	Ns = min(Ns1,Ns2)

	# reshape vector to matrix to rewrite vector as an outer product.
	v0=_np.reshape(v0,(Ns1,Ns2))
	# take singular value decomposition to get which decomposes the matrix into separate parts.
	# the outer/tensor product of the cols of V1 and V2 are the product states which make up the original vector 

	if sparse:
		V1,S,V2=_sla.svds(v0,k=Ns-1,which='SM',maxiter=1E10)
		V12,[S2],V22=_sla.svds(v0,k=1,which='LM',maxiter=1E10)

		S.resize((Ns,))
		S[-1] = S2
		V1.resize((Ns1,Ns))
		V1[:,-1] = V12[:,0]
		V2.resize((Ns,Ns2))
		V2[-1,:] = V22[0,:]
	else:
		V1,S,V2=_la.svd(v0)
		
	# svd returns V2.T.conj() so take the hc to reverse that
	V2=V2.T.conj()
	eps = _np.finfo(S.dtype).eps
	# for any values of s which are 0, remove those vectors because they do not contribute.
	mask=(S >= 10*eps)
	V1=V1[:,mask]
	V2=V2[:,mask]
	S=S[mask]


	# Next thing to do is take those vectors and convert them to their full hilbert space
	if full_left:
		V1=basis._basis_left.project_from(V1,sparse)

	if full_right:
		V2=basis._basis_right.project_from(V2,sparse)


	# calculate the dimension total hilbert space with no symmetries
	Ns=V2.shape[0]*V1.shape[0]		


	if sparse:
		v0=_sp.csr_matrix((Ns,1),dtype=V2.dtype)
		# combining all the vectors together with the tensor product as opposed to the outer product
		for i,s in enumerate(S):
			v1=V1.getcol(i)
			v2=V2.getcol(i)
			v=_sp.kron(v1,v2)
			v0 = v0 + s*v
		n=_np.sqrt(v0.multiply(v0.conj()).sum())
	#	v0=v0/n
		v0=v0.astype(V1.dtype)
		
		
	else:
		v0=_np.zeros((Ns,),dtype=V2.dtype)
		for i,s in enumerate(S):
			v1=V1[:,i]
			v2=V2[:,i]
			v=_np.kron(v1,v2)
			v0 += s*v
		v0 /= _la.norm(v0)


	return v0


def _combine_project_froms(basis,V0,sparse,full_left,full_right):
	v0_list=[]
	V0=V0.T
	for v0 in V0:
		v0_list.append(_combine_get_vec(basis,v0,sparse,full_left,full_right))

	if sparse:
		V0=_sp.hstack(v0_list)
	else:
		V0=_np.hstack(v0_list)

	return V0
"""
