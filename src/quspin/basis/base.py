from __future__ import print_function
import numpy as _np
import scipy.sparse as _sp
import warnings, numba


def _get_index_type(Ns):
    if Ns < _np.iinfo(_np.int32).max:
        return _np.int32
    else:
        return _np.int64


@numba.njit
def _coo_dot(v_in, v_out, row, col, ME):
    n = row.shape[0]
    m = v_in.shape[1]
    for i in range(n):
        r = row[i]
        c = col[i]
        me = ME[i]
        for j in range(m):
            v_out[r, j] += me * v_in[c, j]


@numba.njit
def _is_diagonal(row, col):
    for i in range(row.size):
        if row[i] != col[i]:
            return False

    return True


@numba.njit
def _update_diag(diag, ind, ME):
    for i in range(ind.size):
        diag[ind[i]] += ME[i]


MAXPRINT = 50
# this file stores the base class for all basis classes

__all__ = ["basis", "isbasis"]


class basis(object):

    def __init__(self):
        self._Ns = 0
        self._basis = _np.asarray([])
        self._operators = "no operators for base."
        self._unique_me = True
        if self.__class__.__name__ == "basis":
            raise ValueError(
                "This class is not intended" " to be instantiated directly."
            )

    def __str__(self):

        string = (
            "reference states: \narray index   /   Fock state   /   integer repr. \n"
        )

        if self._Ns == 0:
            return string

        str_list = list(self._get__str__())
        if self._Ns > MAXPRINT:
            try:
                i1 = list(str_list[-1]).index("|")
                i2 = list(str_list[-1]).index(">", -1)
                l = (i1 + i2) // 2 + (1 - (i1 + i2) % 2)
            except:
                l = len(str_list[-1]) // 2

            t = (" ".join(["" for i in range(l)])) + ":"
            str_list.insert(MAXPRINT // 2, t)

        string += "\n".join(str_list)

        if any("block" in x for x in self._blocks):
            string += "\nThe states printed do NOT correspond to the physical states: see review arXiv:1101.3281 for more details about reference states for symmetry-reduced blocks.\n"
        return string

    @property
    def Ns(self):
        """int: number of states in the Hilbert space."""
        return self._Ns

    @property
    def operators(self):
        """set: set of available operator strings."""
        return self._operators

    @property
    def N(self):
        """int: number of sites the basis is constructed with."""
        return self._N

    @property
    def blocks(self):
        """dict: contains the quantum numbers (blocks) for the symmetry sectors."""
        return dict(self._blocks)

    @property
    def sps(self):
        """int: number of states per site (ie, the on-site Hilbert space dimension)."""
        try:
            return self._sps
        except AttributeError:
            raise NotImplementedError(
                "basis class: {0} missing local number of degrees of freedom per site 'sps' required for entanglement entropy calculations!".format(
                    self.__class__
                )
            )

    def _Op(self, opstr, indx, J, dtype):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_Op' required for calculating matrix elements!".format(
                self.__class__
            )
        )

    def _inplace_Op(
        self,
        v_in,
        op_list,
        dtype,
        transposed=False,
        conjugated=False,
        v_out=None,
        a=1.0,
    ):
        """default version for all basis classes which works so long as _Op is implemented."""

        v_in = _np.asanyarray(v_in)

        result_dtype = _np.result_type(v_in.dtype, dtype)

        if v_in.shape[0] != self.Ns:
            raise ValueError("dimension mismatch")

        if v_out is None:
            v_out = _np.zeros_like(v_in, dtype=result_dtype)

        v_out = v_out.reshape((self.Ns, -1))
        v_in = v_in.reshape((self.Ns, -1))

        for opstr, indx, J in op_list:

            if not transposed:
                ME, row, col = self.Op(opstr, indx, a * J, dtype)
            else:
                ME, col, row = self.Op(opstr, indx, a * J, dtype)

            if conjugated:
                ME = ME.conj()

            _coo_dot(v_in, v_out, row, col, ME)

        return v_out.squeeze()

    def inplace_Op(
        self,
        v_in,
        op_list,
        dtype,
        transposed=False,
        conjugated=False,
        a=1.0,
        v_out=None,
    ):
        """Calculates the action of an operator on a state.

        Notes
        -----
        This function works with the `tensor_basis` and other basis which use the "|" symbol in the opstr.

        Parameters
        ----------
        v_in : array_like
                state (or states stored in columns) to act on with the operator.
        op_list : list
                Operator string list which defines the operator to apply. Follows the format `[["z",[i],Jz[i]] for i in range(L)], ["x",[i],Jx[j]] for j in range(L)],...]`.
        dtype : 'type'
                Data type (e.g. `numpy.float64`) to construct the operator with.
        transposed : bool, optional
                if `True` this function will act with the trasposed operator.
        conjugated : bool, optional
                if `True` this function will act with the conjugated operator.
        a : scalar, optional
                value to rescale resulting vector after performing the action of the operators. Same as rescaling all couplings by value a.
        v_out : array_like
                output array, must be the same shape as `v_in` and must match the type of the output.

        Returns
        -------
        numpy.ndarray
                * if `v_out` is not `None`, this function modifies `v_out` inplace and returns it.


        Examples
        --------

        >>> J = 1.41
        >>> indx = [2,3]
        >>> opstr = "zz"
        >>> dtype = np.float64
        >>> op_list=[[opstr,indx,J]]
        >>> ME, row, col = inplace_Op(op_list,dtype)

        """

        return self._inplace_Op(
            v_in,
            op_list,
            dtype,
            transposed=transposed,
            conjugated=conjugated,
            v_out=v_out,
            a=a,
        )

    def Op(self, opstr, indx, J, dtype):
        """Constructs operator from a site-coupling list and an operator string in a lattice basis.

        Parameters
        ----------
        opstr : str
                Operator string in the lattice basis format. For instance:

                >>> opstr = "zz"
        indx : list(int)
                List of integers to designate the sites the lattice basis operator is defined on. For instance:

                >>> indx = [2,3]
        J : scalar
                Coupling strength.
        dtype : 'type'
                Data type (e.g. numpy.float64) to construct the operator with.

        Returns
        -------
        tuple
                `(ME,row,col)`, where
                        * numpy.ndarray(scalar): `ME`: matrix elements of type `dtype`.
                        * numpy.ndarray(int): `row`: row indices of matrix representing the operator in the lattice basis,
                                such that `row[i]` is the row index of `ME[i]`.
                        * numpy.ndarray(int): `col`: column index of matrix representing the operator in the lattice basis,
                                such that `col[i]` is the column index of `ME[i]`.

        Examples
        --------

        >>> J = 1.41
        >>> indx = [2,3]
        >>> opstr = "zz"
        >>> dtype = np.float64
        >>> ME, row, col = Op(opstr,indx,J,dtype)

        """
        return self._Op(opstr, indx, J, dtype)

    def _make_matrix(self, op_list, dtype):
        """takes list of operator strings and couplings to create matrix."""
        off_diag = None
        diag = None
        index_type = _get_index_type(self.Ns)

        for opstr, indx, J in op_list:
            ME, row, col = self.Op(opstr, indx, J, dtype)
            if len(ME) > 0:
                imax = max(row.max(), col.max())
                row = row.astype(index_type)
                col = col.astype(index_type)
                if _is_diagonal(row, col):
                    if diag is None:
                        diag = _np.zeros(self.Ns, dtype=dtype)
                    _update_diag(diag, row, ME)
                else:
                    if off_diag is None:
                        off_diag = _sp.csr_matrix(
                            (ME, (row, col)), shape=(self.Ns, self.Ns), dtype=dtype
                        )
                    else:
                        off_diag = off_diag + _sp.csr_matrix(
                            (ME, (row, col)), shape=(self.Ns, self.Ns), dtype=dtype
                        )

        if diag is not None and off_diag is not None:
            indptr = _np.arange(self.Ns + 1)
            return off_diag + _sp.csr_matrix(
                (diag, indptr[: self.Ns], indptr), shape=(self.Ns, self.Ns), dtype=dtype
            )

        elif off_diag is not None:
            return off_diag
        elif diag is not None:
            return _sp.dia_matrix(
                (_np.atleast_2d(diag), [0]), shape=(self.Ns, self.Ns), dtype=dtype
            )
        else:
            return _sp.dia_matrix((self.Ns, self.Ns), dtype=dtype)

    def partial_trace(
        self,
        state,
        sub_sys_A=None,
        subsys_ordering=True,
        return_rdm="A",
        enforce_pure=False,
        sparse=False,
    ):
        """Calculates reduced density matrix, through a partial trace of a quantum state in a lattice `basis`.

        Notes
        -----
        This function can also be applied to trace out operators/observables defined by the input `state`, in which case one has to additionally normalize the final output by the Hilbert space dimension of the traced-out space. However, if an operator is defined in a symmetry-reduced basis, there is a :red:`caveat`. In such a case, one has to:
                (1) use the `basis.get_proj()` function to lift the operator to the full basis;
                (2) apply `basis.partial_trace()`;
                (3) repeat this procedure for all symmetry sectors, and sum up the resulting reduced operators [this is becauce one has to add in the information about how the operator acts on the full Hilbert space].

        Parameters
        ----------
        state : obj
                State of the quantum system. Can be either one of:

                        * numpy.ndarray [shape (Ns,)]: pure state (default).
                        * numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
        sub_sys_A : tuple/list, optional
                Defines the sites contained in subsystem A [by python convention the first site of the chain is labelled j=0].
                Default is `tuple(range(N//2))` with `N` the number of lattice sites.
        return_rdm : str, optional
                Toggles returning the reduced DM. Can be tierh one of:

                        * "A": returns reduced DM of subsystem A.
                        * "B": returns reduced DM of subsystem B.
                        * "both": returns reduced DM of both A and B subsystems.
        subsys_ordering : bool, optional
                Whether or not to reorder the sites in `sub_sys_A` in ascending order. Default is `True`.
        enforce_pure : bool, optional
                Whether or not to assume `state` is a colelction of pure states or a mixed density matrix, if
                it is a square array. Default is `False`.
        sparse : bool, optional
                Whether or not to return a sparse DM. Default is `False`.

        Returns
        -------
        numpy.ndarray
                Density matrix associated with `state`. Depends on optional arguments.

        Examples
        --------

        >>> partial_trace(state,sub_sys_A=tuple(range(basis.N//2),return_rdm="A",enforce_pure=False,sparse=False,subsys_ordering=True)


        """

        return self._partial_trace(
            state,
            sub_sys_A=sub_sys_A,
            subsys_ordering=subsys_ordering,
            return_rdm=return_rdm,
            enforce_pure=enforce_pure,
            sparse=sparse,
        )

    def ent_entropy(
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
        """Calculates entanglement entropy of subsystem A and the corresponding reduced density matrix

        .. math::
                S_\\mathrm{ent}(\\alpha) = \\frac{1}{N_A}\\frac{1}{1-\\alpha}\\log \\mathrm{tr}_{A} \\left( \\mathrm{tr}_{A^c} \\vert\\psi\\rangle\\langle\\psi\\vert \\right)^\\alpha

        where the normalization :math:`N_A` can be switched on and off using the optional argument `density`. This expression reduces to the familiar von Neumann entropy in the limit :math:`\\alpha=1`.

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
        sub_sys_A : tuple/list, optional
                Defines the sites contained in subsystem A [by python convention the first site of the chain is labelled j=0].
                Default is `tuple(range(N//2))` with `N` the number of lattice sites.
        density : bool, optional
                Toggles whether to return entanglement entropy normalized by the number of sites in the subsystem.
        return_rdm : str, optional
                Toggles returning the reduced DM. Can be tierh one of:

                        * "A": returns reduced DM of subsystem A.
                        * "B": returns reduced DM of subsystem B.
                        * "both": returns reduced DM of both A and B subsystems.
        enforce_pure : bool, optional
                Whether or not to assume `state` is a collection of pure states or a mixed density matrix, if
                it is a square array. Default is `False`.
        subsys_ordering : bool, optional
                Whether or not to reorder the sites in `sub_sys_A` in ascending order. Default is `True`.
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
        svd_solver : object, optional
                Specifies the svd solver to be used, e.g. `numpy.linalg.svd` or `scipy.linalg.svd`, or a custom solver. Effective when `enforce_pure=True` or `sparse=False`.
        svd_kwargs : dict, optional
                Specifies additional arguments for `svd_solver`.

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

        >>> ent_entropy(state,sub_sys_A=[0,3,4,7],return_rdm="A",enforce_pure=False,return_rdm_EVs=False,
        >>>				sparse=False,alpha=1.0,sparse_diag=True,subsys_ordering=True)

        """

        return self._ent_entropy(
            state,
            sub_sys_A=sub_sys_A,
            density=density,
            subsys_ordering=subsys_ordering,
            return_rdm=return_rdm,
            enforce_pure=enforce_pure,
            return_rdm_EVs=return_rdm_EVs,
            sparse=sparse,
            alpha=alpha,
            sparse_diag=sparse_diag,
            maxiter=maxiter,
            svd_solver=svd_solver,
            svd_kwargs=svd_kwargs,
        )

    def expanded_form(self, static=[], dynamic=[]):
        """Splits up operator strings containing "x" and "y" into operator combinations of "+" and "-". This function is useful for higher spin hamiltonians where "x" and "y" operators are not appropriate operators.

        Notes
        -----
        This function works with the `tensor_basis` and other basis which use the "|" symbol in the opstr.

        Parameters
        ----------
        static: list
                Static operators formatted to be passed into the static argument of the `hamiltonian` class.
        dynamic: list
                Dynamic operators formatted to be passed into the dynamic argument of the `hamiltonian` class.

        Returns
        -------
        tuple
                `(static, dynamic)`, where
                        * list: `static`: operator strings with "x" and "y" expanded into "+" and "-", formatted to
                                be passed into the static argument of the `hamiltonian` class.
                        * list: `dynamic`: operator strings with "x" and "y" expanded into "+" and "-", formatted to
                                be passed into the dynamic argument of the `hamiltonian` class.

        Examples
        --------

        >>> static = [["xx",[[1.0,0,1]]],["yy",[[1.0,0,1]]]]
        >>> dynamic = [["y",[[1.0,0]],lambda t: t,[]]]
        >>> expanded_form(static,dynamic)

        """
        static_list, dynamic_list = self._get_local_lists(static, dynamic)
        static_list = self._expand_local_list(static_list)
        dynamic_list = self._expand_local_list(dynamic_list)
        static_list, dynamic_list = self._consolidate_local_lists(
            static_list, dynamic_list
        )

        static_dict = {}
        for opstr, indx, J, ii in static_list:
            indx = list(indx)
            indx.insert(0, J)
            if opstr in static_dict:
                static_dict[opstr].append(indx)
            else:
                static_dict[opstr] = [indx]

        static = [[str(key), list(value)] for key, value in static_dict.items()]

        dynamic_dict = {}
        for opstr, indx, J, f, f_args, ii in dynamic_list:
            indx = list(indx)
            indx.insert(0, J)
            if (opstr, f, f_args) in dynamic_dict:
                dynamic_dict[(opstr, f, f_args)].append(indx)
            else:
                dynamic_dict[(opstr, f, f_args)] = [indx]

        dynamic = [
            [opstr, indx, f, f_args]
            for (opstr, f, f_args), indx in dynamic_dict.items()
        ]

        return static, dynamic

    def check_hermitian(self, static, dynamic):
        """Checks operator string lists for hermiticity of the combined operator.

        Parameters
        ----------
        static: list
                Static operators formatted to be passed into the static argument of the `hamiltonian` class.
        dynamic: list
                Dynamic operators formatted to be passed into the dynamic argument of the `hamiltonian` class.

        Examples
        --------

        """
        if self._check_herm is None:
            warnings.warn(
                "Test for hermiticity not implemented for {0}, to turn off this warning set check_symm=False in hamiltonian".format(
                    type(self)
                ),
                UserWarning,
                stacklevel=3,
            )
            return

        static_list, dynamic_list = self._get_local_lists(static, dynamic)
        static_expand, static_expand_hc, dynamic_expand, dynamic_expand_hc = (
            self._get_hc_local_lists(static_list, dynamic_list)
        )
        # calculate non-hermitian elements
        diff = set(tuple(static_expand)) - set(tuple(static_expand_hc))

        if diff:
            unique_opstrs = list(set(next(iter(zip(*tuple(diff))))))
            warnings.warn(
                "The following static operator strings contain non-hermitian couplings: {}".format(
                    unique_opstrs
                ),
                UserWarning,
                stacklevel=3,
            )
            try:
                user_input = raw_input(
                    "Display all {} non-hermitian couplings? (y or n) ".format(
                        len(diff)
                    )
                )
            except NameError:
                user_input = input(
                    "Display all {} non-hermitian couplings? (y or n) ".format(
                        len(diff)
                    )
                )

            if user_input == "y":
                print("   (opstr, indices, coupling)")
                for i in range(len(diff)):
                    print("{}. {}".format(i + 1, list(diff)[i]))
            raise TypeError(
                "Hamiltonian not hermitian! To turn this check off set check_herm=False in hamiltonian."
            )

        # define arbitrarily complicated weird-ass number
        t = _np.cos((_np.pi / _np.exp(0)) ** (1.0 / _np.euler_gamma))

        # calculate non-hermitian elements
        diff = set(tuple(dynamic_expand)) - set(tuple(dynamic_expand_hc))
        if diff:
            unique_opstrs = list(set(next(iter(zip(*tuple(diff))))))
            warnings.warn(
                "The following dynamic operator strings contain non-hermitian couplings: {}".format(
                    unique_opstrs
                ),
                UserWarning,
                stacklevel=3,
            )
            try:
                user_input = raw_input(
                    "Display all {} non-hermitian couplings at time t = {}? (y or n) ".format(
                        len(diff), _np.round(t, 5)
                    )
                )
            except NameError:
                user_input = input(
                    "Display all {} non-hermitian couplings at time t = {}? (y or n) ".format(
                        len(diff), _np.round(t, 5)
                    )
                )

            if user_input == "y":
                print("   (opstr, indices, coupling(t))")
                for i in range(len(diff)):
                    print("{}. {}".format(i + 1, list(diff)[i]))
            raise TypeError(
                "Hamiltonian not hermitian! To turn this check off set check_herm=False in hamiltonian."
            )

        print("Hermiticity check passed!")

    def check_symm(self, static, dynamic):
        """Checks operator string lists for the required symmetries of the combined operator.

        Parameters
        ----------
        static: list
                Static operators formatted to be passed into the static argument of the `hamiltonian` class.
        dynamic: list
                Dynamic operators formatted to be passed into the dynamic argument of the `hamiltonian` class.

        Examples
        --------

        """
        if self._check_symm is None:
            warnings.warn(
                "Test for symmetries not implemented for {0}, to turn off this warning set check_symm=False in hamiltonian".format(
                    type(self)
                ),
                UserWarning,
                stacklevel=3,
            )
            return

        static_blocks, dynamic_blocks = self._check_symm(static, dynamic)

        # define arbitrarily complicated weird-ass number
        t = _np.cos((_np.pi / _np.exp(0)) ** (1.0 / _np.euler_gamma))

        for symm in static_blocks.keys():
            if len(static_blocks[symm]) == 2:
                odd_ops, missing_ops = static_blocks[symm]
                ops = list(missing_ops)
                ops.extend(odd_ops)
                unique_opstrs = list(set(next(iter(zip(*tuple(ops))))))
                if unique_opstrs:
                    unique_missing_ops = []
                    unique_odd_ops = []
                    [
                        unique_missing_ops.append(ele)
                        for ele in missing_ops
                        if ele not in unique_missing_ops
                    ]
                    [
                        unique_odd_ops.append(ele)
                        for ele in odd_ops
                        if ele not in unique_odd_ops
                    ]
                    warnings.warn(
                        "The following static operator strings do not obey {0}: {1}".format(
                            symm, unique_opstrs
                        ),
                        UserWarning,
                        stacklevel=4,
                    )
                    try:
                        user_input = raw_input(
                            "Display all {0} couplings? (y or n) ".format(
                                len(unique_missing_ops) + len(unique_odd_ops)
                            )
                        )
                    except NameError:
                        user_input = input(
                            "Display all {0} couplings? (y or n) ".format(
                                len(unique_missing_ops) + len(unique_odd_ops)
                            )
                        )
                    if user_input == "y":
                        print(" these operators are needed for {}:".format(symm))
                        print("   (opstr, indices, coupling)")
                        for i, op in enumerate(unique_missing_ops):
                            print("{0}. {1}".format(i + 1, op))
                        print(" ")
                        print(" these do not obey the {}:".format(symm))
                        print("   (opstr, indices, coupling)")
                        for i, op in enumerate(unique_odd_ops):
                            print("{0}. {1}".format(i + 1, op))
                    raise TypeError(
                        "Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(
                            symm
                        )
                    )

            elif len(static_blocks[symm]) == 1:
                (missing_ops,) = static_blocks[symm]
                unique_opstrs = list(set(next(iter(zip(*tuple(missing_ops))))))
                if unique_opstrs:
                    unique_missing_ops = []
                    [
                        unique_missing_ops.append(ele)
                        for ele in missing_ops
                        if ele not in unique_missing_ops
                    ]
                    warnings.warn(
                        "The following static operator strings do not obey {0}: {1}".format(
                            symm, unique_opstrs
                        ),
                        UserWarning,
                        stacklevel=4,
                    )
                    try:
                        user_input = raw_input(
                            "Display all {0} couplings? (y or n) ".format(
                                len(unique_missing_ops)
                            )
                        )
                    except NameError:
                        user_input = input(
                            "Display all {0} couplings? (y or n) ".format(
                                len(unique_missing_ops)
                            )
                        )

                    if user_input == "y":
                        print(" these operators are needed for {}:".format(symm))
                        print("   (opstr, indices, coupling)")
                        for i, op in enumerate(unique_missing_ops):
                            print("{0}. {1}".format(i + 1, op))
                    raise TypeError(
                        "Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(
                            symm
                        )
                    )
            else:
                continue

        for symm in dynamic_blocks.keys():
            if len(dynamic_blocks[symm]) == 2:
                odd_ops, missing_ops = dynamic_blocks[symm]
                ops = list(missing_ops)
                ops.extend(odd_ops)
                unique_opstrs = list(set(next(iter(zip(*tuple(ops))))))
                if unique_opstrs:
                    unique_missing_ops = []
                    unique_odd_ops = []
                    [
                        unique_missing_ops.append(ele)
                        for ele in missing_ops
                        if ele not in unique_missing_ops
                    ]
                    [
                        unique_odd_ops.append(ele)
                        for ele in odd_ops
                        if ele not in unique_odd_ops
                    ]
                    warnings.warn(
                        "The following dynamic operator strings do not obey {0}: {1}".format(
                            symm, unique_opstrs
                        ),
                        UserWarning,
                        stacklevel=4,
                    )
                    try:
                        user_input = raw_input(
                            "Display all {0} couplings? (y or n) ".format(
                                len(unique_missing_ops) + len(unique_odd_ops)
                            )
                        )
                    except NameError:
                        user_input = input(
                            "Display all {0} couplings? (y or n) ".format(
                                len(unique_missing_ops) + len(unique_odd_ops)
                            )
                        )

                    if user_input == "y":
                        print(" these operators are missing for {}:".format(symm))
                        print("   (opstr, indices, coupling)")
                        for i, op in enumerate(unique_missing_ops):
                            print("{0}. {1}".format(i + 1, op))
                        print(" ")
                        print(" these do not obey {}:".format(symm))
                        print("   (opstr, indices, coupling)")
                        for i, op in enumerate(unique_odd_ops):
                            print("{0}. {1}".format(i + 1, op))
                    raise TypeError(
                        "Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(
                            symm
                        )
                    )

            elif len(dynamic_blocks[symm]) == 1:
                (missing_ops,) = dynamic_blocks[symm]
                unique_opstrs = list(set(next(iter(zip(*tuple(missing_ops))))))
                if unique_opstrs:
                    unique_missing_ops = []
                    [
                        unique_missing_ops.append(ele)
                        for ele in missing_ops
                        if ele not in unique_missing_ops
                    ]
                    warnings.warn(
                        "The following dynamic operator strings do not obey {0}: {1}".format(
                            symm, unique_opstrs
                        ),
                        UserWarning,
                        stacklevel=4,
                    )
                    try:
                        user_input = raw_input(
                            "Display all {0} couplings? (y or n) ".format(
                                len(unique_missing_ops)
                            )
                        )
                    except NameError:
                        user_input = input(
                            "Display all {0} couplings? (y or n) ".format(
                                len(unique_missing_ops)
                            )
                        )

                    if user_input == "y":
                        print(" these operators are needed for {}:".format(symm))
                        print("   (opstr, indices, coupling)")
                        for i, op in enumerate(unique_missing_ops):
                            print("{0}. {1}".format(i + 1, op))
                    raise TypeError(
                        "Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(
                            symm
                        )
                    )
            else:
                continue

        print("Symmetry checks passed!")

    def check_pcon(self, static, dynamic):
        """Checks operator string lists for particle number (magnetisation) conservartion of the combined operator.

        Parameters
        ----------
        static: list
                Static operators formatted to be passed into the static argument of the `hamiltonian` class.
        dynamic: list
                Dynamic operators formatted to be passed into the dynamic argument of the `hamiltonian` class.

        Examples
        --------

        """
        if self._check_pcon is None:
            warnings.warn(
                "Test for particle conservation not implemented for {0}, to turn off this warning set check_pcon=False in hamiltonian".format(
                    type(self)
                ),
                UserWarning,
                stacklevel=3,
            )
            return

        if self._check_pcon:
            static_list, dynamic_list = self._get_local_lists(static, dynamic)
            static_list_exp = self._expand_local_list(static_list)
            dynamic_list_exp = self._expand_local_list(dynamic_list)
            static_list_exp, dynamic_list_exp = self._consolidate_local_lists(
                static_list_exp, dynamic_list_exp
            )
            con = ""

            odd_ops = []
            for opstr, indx, J, ii in static_list_exp:
                p = opstr.count("+")
                m = opstr.count("-")

                if (p - m) != 0:
                    for i in ii:
                        if static_list[i] not in odd_ops:
                            odd_ops.append(static_list[i])

            if odd_ops:
                unique_opstrs = list(set(next(iter(zip(*tuple(odd_ops))))))
                unique_odd_ops = []
                [
                    unique_odd_ops.append(ele)
                    for ele in odd_ops
                    if ele not in unique_odd_ops
                ]
                warnings.warn(
                    "The following static operator strings do not conserve particle number{1}: {0}".format(
                        unique_opstrs, con
                    ),
                    UserWarning,
                    stacklevel=4,
                )
                try:
                    user_input = raw_input(
                        "Display all {0} couplings? (y or n) ".format(len(odd_ops))
                    )
                except NameError:
                    user_input = input(
                        "Display all {0} couplings? (y or n) ".format(len(odd_ops))
                    )

                if user_input == "y":
                    print(
                        " these operators do not conserve particle number{0}:".format(
                            con
                        )
                    )
                    print("   (opstr, indices, coupling)")
                    for i, op in enumerate(unique_odd_ops):
                        print("{0}. {1}".format(i + 1, op))
                raise TypeError(
                    "Hamiltonian does not conserve particle number{0} To turn off check, use check_pcon=False in hamiltonian.".format(
                        con
                    )
                )

            odd_ops = []
            for opstr, indx, J, f, f_args, ii in dynamic_list_exp:
                p = opstr.count("+")
                m = opstr.count("-")
                if (p - m) != 0:
                    for i in ii:
                        if dynamic_list[i] not in odd_ops:
                            odd_ops.append(dynamic_list[i])

            if odd_ops:
                unique_opstrs = list(set(next(iter(zip(*tuple(odd_ops))))))
                unique_odd_ops = []
                [
                    unique_odd_ops.append(ele)
                    for ele in odd_ops
                    if ele not in unique_odd_ops
                ]
                warnings.warn(
                    "The following static operator strings do not conserve particle number{1}: {0}".format(
                        unique_opstrs, con
                    ),
                    UserWarning,
                    stacklevel=4,
                )
                try:
                    user_input = raw_input(
                        "Display all {0} couplings? (y or n) ".format(len(odd_ops))
                    )
                except NameError:
                    user_input = input(
                        "Display all {0} couplings? (y or n) ".format(len(odd_ops))
                    )
                if user_input == "y":
                    print(
                        " these operators do not conserve particle number{0}:".format(
                            con
                        )
                    )
                    print("   (opstr, indices, coupling)")
                    for i, op in enumerate(unique_odd_ops):
                        print("{0}. {1}".format(i + 1, op))
                raise TypeError(
                    "Hamiltonian does not conserve particle number{0} To turn off check, use check_pcon=False in hamiltonian.".format(
                        con
                    )
                )

            print("Particle conservation check passed!")

    def _get__str__(self):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_get__str__' required to print out the basis!".format(
                self.__class__
            )
        )

    # this methods are optional and are not required for main functions:
    def __iter__(self):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '__iter__' required for iterating over basis!".format(
                self.__class__
            )
        )

    def __getitem__(self, *args, **kwargs):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '__getitem__' required for '[]' operator!".format(
                self.__class__
            )
        )

    # thes methods are required for the symmetry, particle conservation, and hermiticity checks
    def _hc_opstr(self, *args, **kwargs):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_hc_opstr' required for hermiticity check! turn this check off by setting test_herm=False".format(
                self.__class__
            )
        )

    def _sort_opstr(self, *args, **kwargs):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_sort_opstr' required for symmetry and hermiticity checks! turn this check off by setting check_herm=False".format(
                self.__class__
            )
        )

    def _expand_opstr(self, *args, **kwargs):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_expand_opstr' required for particle conservation check! turn this check off by setting check_pcon=False".format(
                self.__class__
            )
        )

    def _non_zero(self, *args, **kwargs):
        raise NotImplementedError(
            "basis class: {0} missing implementation of '_non_zero' required for particle conservation check! turn this check off by setting check_pcon=False".format(
                self.__class__
            )
        )

    def _sort_local_list(self, op_list):
        sorted_op_list = []
        for op in op_list:
            sorted_op_list.append(self._sort_opstr(op))
        sorted_op_list = tuple(sorted_op_list)

        return sorted_op_list

    # this function flattens out the static and dynamics lists to:
    # [[opstr1,indx11,J11,...],[opstr1,indx12,J12,...],...,[opstrn,indxnm,Jnm,...]]
    # this function gets overridden in photon_basis because the index must be extended to include the photon index.
    def _get_local_lists(self, static, dynamic):
        static_list = []
        for opstr, bonds in static:
            for bond in bonds:
                indx = list(bond[1:])
                J = complex(bond[0])
                static_list.append((opstr, indx, J))

        dynamic_list = []
        for opstr, bonds, f, f_args in dynamic:
            for bond in bonds:
                indx = list(bond[1:])
                J = complex(bond[0])
                dynamic_list.append((opstr, indx, J, f, f_args))

        return self._sort_local_list(static_list), self._sort_local_list(dynamic_list)

    # takes the list from the format given by _get_local_lists and takes the hermitian conjugate of operators.
    def _get_hc_local_lists(self, static_list, dynamic_list):
        static_list_hc = []
        for op in static_list:
            static_list_hc.append(self._hc_opstr(op))

        static_list_hc = tuple(static_list_hc)

        # define arbitrarily complicated weird-ass number
        t = _np.cos((_np.pi / _np.exp(0)) ** (1.0 / _np.euler_gamma))

        dynamic_list_hc = []
        dynamic_list_eval = []
        for opstr, indx, J, f, f_args in dynamic_list:
            J *= f(t, *f_args)
            op = (opstr, indx, J)
            dynamic_list_hc.append(self._hc_opstr(op))
            dynamic_list_eval.append(self._sort_opstr(op))

        dynamic_list_hc = tuple(dynamic_list_hc)

        return static_list, static_list_hc, dynamic_list_eval, dynamic_list_hc

    # this function takes the list format giveb by get_local_lists and expands any operators into the most basic components
    # 'n'(or 'z' for spins),'+','-' If by default one doesn't need to do this then _expand_opstr must do nothing.
    def _expand_local_list(self, op_list):
        op_list_exp = []
        for i, op in enumerate(op_list):
            new_ops = self._expand_opstr(op, [i])
            for new_op in new_ops:
                if self._non_zero(new_op):
                    op_list_exp.append(new_op)

        return self._sort_local_list(op_list_exp)

    def _consolidate_local_lists(self, static_list, dynamic_list):
        eps = 10 * _np.finfo(_np.float64).eps

        static_dict = {}
        for opstr, indx, J, ii in static_list:
            if opstr in static_dict:
                if indx in static_dict[opstr]:
                    static_dict[opstr][indx][0] += J
                    static_dict[opstr][indx][1].extend(ii)
                else:
                    static_dict[opstr][indx] = [J, ii]
            else:
                static_dict[opstr] = {indx: [J, ii]}

        static_list = []
        for opstr, opstr_dict in static_dict.items():
            for indx, (J, ii) in opstr_dict.items():
                if _np.abs(J) > eps:
                    static_list.append((opstr, indx, J, ii))

        dynamic_dict = {}
        for opstr, indx, J, f, f_args, ii in dynamic_list:
            if opstr in dynamic_dict:
                if indx in dynamic_dict[opstr]:
                    dynamic_dict[opstr][indx][0] += J
                    dynamic_dict[opstr][indx][3].extend(ii)
                else:
                    dynamic_dict[opstr][indx] = [J, f, f_args, ii]
            else:
                dynamic_dict[opstr] = {indx: [J, f, f_args, ii]}

        dynamic_list = []
        for opstr, opstr_dict in dynamic_dict.items():
            for indx, (J, f, f_args, ii) in opstr_dict.items():
                if _np.abs(J) > eps:
                    dynamic_list.append((opstr, indx, J, f, f_args, ii))

        return static_list, dynamic_list


def isbasis(obj):
    """Checks if instance is object of `basis` class.

    Parameters
    ----------
    obj :
            Arbitraty python object.

    Returns
    -------
    bool
            Can be either of the following:

            * `True`: `obj` is an instance of `basis` class.
            * `False`: `obj` is NOT an instance of `basis` class.

    """
    return isinstance(obj, basis)
