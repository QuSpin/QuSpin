import numpy as _np
import scipy.sparse as _sp
import os, numexpr
from quspin_extensions.basis.basis_general._basis_general_core.general_basis_utils import (
    basis_int_to_python_int,
    _get_basis_index,
)
from quspin_extensions.basis.basis_general._basis_general_core.general_basis_utils import (
    _basis_argsort,
    _is_sorted_decending,
)
from quspin_extensions.basis.basis_general._basis_general_core import basis_zeros
from quspin.basis.lattice import lattice_basis
from quspin.basis.base import _get_index_type
import warnings


class GeneralBasisWarning(Warning):
    pass


def process_map(map, q):
    map = _np.asarray(map, dtype=_np.int32)
    i_map = map.copy()
    i_map[map < 0] = -(i_map[map < 0] + 1)  # site mapping
    s_map = map < 0  # sites with spin-inversion

    sites = _np.arange(len(map), dtype=_np.int32)
    order = sites.copy()

    if _np.any(_np.sort(i_map) - order):
        raise ValueError("map must be a one-to-one site mapping.")

    per = 0
    group = [tuple(order)]
    while True:
        sites[s_map] = -(sites[s_map] + 1)
        sites = sites[i_map]
        per += 1
        group.append(tuple(sites))
        if _np.array_equal(order, sites):
            break

    if per == 1:
        warnings.warn(
            "identity mapping found in set of transformations.",
            GeneralBasisWarning,
            stacklevel=5,
        )

    return map, per, q, set(group)


def check_symmetry_maps(item1, item2):
    grp1 = item1[1][-1]
    map1 = item1[1][0]
    block1 = item1[0]

    i_map1 = map1.copy()
    i_map1[map1 < 0] = -(i_map1[map1 < 0] + 1)  # site mapping
    s_map1 = map1 < 0  # sites with spin-inversion

    grp2 = item2[1][-1]
    map2 = item2[1][0]
    block2 = item2[0]

    i_map2 = map2.copy()
    i_map2[map2 < 0] = -(i_map2[map2 < 0] + 1)  # site mapping
    s_map2 = map2 < 0  # sites with spin-inversion

    if grp1 == grp2:
        warnings.warn(
            "mappings for block {} and block {} produce the same symmetry.".format(
                block1, block2
            ),
            GeneralBasisWarning,
            stacklevel=5,
        )

    sites1 = _np.arange(len(map1))
    sites2 = _np.arange(len(map2))

    sites1[s_map1] = -(sites1[s_map1] + 1)
    sites1 = sites1[i_map1]
    sites1[s_map2] = -(sites1[s_map2] + 1)
    sites1 = sites1[i_map2]

    sites2[s_map2] = -(sites2[s_map2] + 1)
    sites2 = sites2[i_map2]
    sites2[s_map1] = -(sites2[s_map1] + 1)
    sites2 = sites2[i_map1]

    if not _np.array_equal(sites1, sites2):
        warnings.warn(
            "using non-commuting symmetries can lead to unwanted behaviour of general basis, make sure that quantum numbers are invariant under non-commuting symmetries!",
            GeneralBasisWarning,
            stacklevel=5,
        )


class basis_general(lattice_basis):
    def __init__(self, N, block_order=None, **kwargs):
        lattice_basis.__init__(self)
        self._unique_me = True
        self._check_herm = True

        self._check_pcon = None
        self._basis_pcon = None

        self._get_proj_pcon = False
        self._made_basis = False  # keeps track of whether the basis has been made
        self._Ns_block_est = 0  # initialize number of states variable

        if self.__class__ is basis_general:
            raise TypeError("general_basis class is not to be instantiated.")

        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        # if not kwargs:
        # 	raise ValueError("require at least one map.")

        n_maps = len(kwargs)

        if n_maps > 32:
            raise ValueError("general basis can only support up to 32 symmetries.")

        if n_maps > 0:
            self._conserved = "custom symmetries"
        else:
            self._conserved = ""

        if any((type(map) is not tuple) and (len(map) != 2) for map in kwargs.values()):
            raise ValueError("blocks must contain tuple: (map,q).")

        kwargs = {block: process_map(*item) for block, item in kwargs.items()}

        if block_order is None:
            # sort by periodicies smallest to largest for speed up
            sorted_items = sorted(kwargs.items(), key=lambda x: x[1][1])
            # sorted_items.reverse()
        else:
            block_order = list(block_order)
            missing = set(kwargs.keys()) - set(block_order)
            if len(missing) > 0:
                raise ValueError(
                    "{} names found in block names but missing from block_order.".format(
                        missing
                    )
                )

            missing = set(block_order) - set(kwargs.keys())
            if len(missing) > 0:
                raise ValueError(
                    "{} names found in block_order but missing from block names.".format(
                        missing
                    )
                )

            block_order.reverse()
            sorted_items = [(key, kwargs[key]) for key in block_order]

        self._blocks = {
            block: ((-1) ** q if per == 2 else q)
            for block, (_, per, q, _) in sorted_items
        }
        self._maps_dict = {block: map for block, (map, _, _, _) in sorted_items}
        remove_index = []
        for i, item1 in enumerate(sorted_items[:-1]):
            if item1[1][1] == 1:
                remove_index.append(i)
            for j, item2 in enumerate(sorted_items[i + 1 :]):
                check_symmetry_maps(item1, item2)

        remove_index.sort()

        if sorted_items:
            blocks, items = zip(*sorted_items)
            items = list(items)

            for i in remove_index:
                items.pop(i)

            n_maps = len(items)
            maps, pers, qs, _ = zip(*items)

            self._maps = _np.vstack(maps)
            self._qs = _np.asarray(qs, dtype=_np.int32)
            self._pers = _np.asarray(pers, dtype=_np.int32)

            if any(map.ndim != 1 for map in self._maps[:]):
                raise ValueError("maps must be a 1-dim array/list of integers.")

            if any(map.shape[0] != N for map in self._maps[:]):
                raise ValueError("size of map is not equal to N.")

            if self._maps.shape[0] != self._qs.shape[0]:
                raise ValueError(
                    "number of maps must be the same as the number of quantum numbers provided."
                )

            for j in range(n_maps - 1):
                for i in range(j + 1, n_maps, 1):
                    if _np.all(self._maps[j] == self._maps[i]):
                        ValueError("repeated map in maps list.")

        else:
            self._maps = _np.array([[]], dtype=_np.int32)
            self._qs = _np.array([], dtype=_np.int32)
            self._pers = _np.array([], dtype=_np.int32)

        nmax = self._pers.prod()
        self._n_dtype = _np.min_scalar_type(nmax)

    def __getstate__(self):
        obj_dict = dict(self.__dict__)
        obj_dict.pop("_core")
        return obj_dict

    # @property
    # def _fermion_basis(self):
    # 	return False

    @property
    def description(self):
        """str: information about `basis` object."""
        blocks = ""

        for symm in self._blocks:
            blocks += symm + " = {" + symm + "}, "

        blocks = blocks.format(**self._blocks)

        if len(self._conserved) == 0:
            symm = "no symmetry"
        elif len(self._conserved) == 1:
            symm = "symmetry"
        else:
            symm = "symmetries"

        string = """general basis for lattice of N = {0} sites containing {5} states \n\t{1}: {2} \n\tquantum numbers: {4} \n\n""".format(
            self._N, symm, self._conserved, "", blocks, self._Ns
        )
        string += self.operators
        return string

    def _int_to_state(self, state, bracket_notation=True):
        state = basis_int_to_python_int(state)

        n_space = len(str(self.sps))
        if self.N <= 64:
            bits = (
                state // int(self.sps ** (self.N - i - 1)) % self.sps
                for i in range(self.N)
            )
            s_str = " ".join(("{:" + str(n_space) + "d}").format(bit) for bit in bits)
        else:
            left_bits = (
                state // int(self.sps ** (self.N - i - 1)) % self.sps for i in range(32)
            )
            right_bits = (
                state // int(self.sps ** (self.N - i - 1)) % self.sps
                for i in range(self.N - 32, self.N, 1)
            )

            str_list = [("{:" + str(n_space) + "d}").format(bit) for bit in left_bits]
            str_list.append("...")
            str_list.extend(
                ("{:" + str(n_space) + "d}").format(bit) for bit in right_bits
            )
            s_str = " ".join(str_list)

        if bracket_notation:
            return "|" + s_str + ">"
        else:
            return s_str.replace(" ", "")

    def _state_to_int(self, state):
        state = state.replace("|", "").replace(">", "").replace("<", "")
        return basis_int_to_python_int(self._basis[self.index(state)])

    def _index(self, s):
        if type(s) is str:
            s = int(s, self.sps)

        return _get_basis_index(self.states, s)

    def _reduce_n_dtype(self):
        if len(self._n) > 0:
            self._n_dtype = _np.min_scalar_type(self._n.max())
            self._n = self._n.astype(self._n_dtype)

    def _Op(self, opstr, indx, J, dtype):

        if not self._made_basis:
            raise AttributeError(
                "this function requires the basis to be constructed first; use basis.make()."
            )

        indx = _np.asarray(indx, dtype=_np.int32)

        if len(opstr) != len(indx):
            raise ValueError("length of opstr does not match length of indx")

        if _np.any(indx >= self._N) or _np.any(indx < 0):
            raise ValueError("values in indx falls outside of system")

        extra_ops = set(opstr) - self._allowed_ops
        if extra_ops:
            raise ValueError(
                "unrecognized characters {} in operator string.".format(extra_ops)
            )

        if self._Ns <= 0:
            return (
                _np.array([], dtype=dtype),
                _np.array([], dtype=self._index_type),
                _np.array([], dtype=self._index_type),
            )

        col = _np.empty(self._Ns, dtype=self._index_type)
        row = _np.empty(self._Ns, dtype=self._index_type)
        ME = _np.empty(self._Ns, dtype=dtype)
        # print(self._Ns)
        self._core.op(
            row,
            col,
            ME,
            opstr,
            indx,
            J,
            self._basis,
            self._n,
            self._basis_begin,
            self._basis_end,
            self._N_p,
        )

        if _np.iscomplexobj(ME):
            if ME.dtype == _np.complex64:
                mask = ME.real != 0
                mask1 = ME.imag != 0
                _np.logical_or(mask, mask1, out=mask)
            else:
                mask = numexpr.evaluate("(real(ME)!=0) | (imag(ME)!=0)")
        else:
            mask = numexpr.evaluate("ME!=0")

        col = col[mask]
        row = row[mask]
        ME = ME[mask]

        return ME, row, col

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
        if not self._made_basis:
            raise AttributeError(
                "this function requires the basis to be constructed first; use basis.make()."
            )

        v_in = _np.asanyarray(v_in)

        result_dtype = _np.result_type(v_in.dtype, dtype)
        v_in = v_in.astype(result_dtype, order="C", copy=False)

        if v_in.shape[0] != self.Ns:
            raise ValueError("dimension mismatch")

        if v_out is None:
            v_out = _np.zeros_like(v_in, dtype=result_dtype, order="C")
        else:
            if v_out.dtype != result_dtype:
                raise TypeError("v_out does not have the correct data type.")
            if not v_out.flags["CARRAY"]:
                raise ValueError("v_out is not a writable C-contiguous array")
            if v_out.shape != v_in.shape:
                raise ValueError(
                    "invalid shape for v_out and v_in: v_in.shape != v_out.shape"
                )

        v_out = v_out.reshape((self.Ns, -1))
        v_in = v_in.reshape((self.Ns, -1))

        for opstr, indx, J in op_list:
            indx = _np.ascontiguousarray(indx, dtype=_np.int32)

            self._core.inplace_op(
                v_in,
                v_out,
                conjugated,
                transposed,
                opstr,
                indx,
                a * J,
                self._basis,
                self._n,
                self._basis_begin,
                self._basis_end,
                self._N_p,
            )

        return v_out.squeeze()

    def Op_shift_sector(self, other_basis, op_list, v_in, v_out=None, dtype=None):
        """Applies symmetry non-conserving operator to state in symmetry-reduced basis.

        An operator, which does not conserve a symmetry, induces a change in the quantum number of a state defined in the corresponding symmetry sector. Hence, when the operator is applied on a quantum state, the state shifts the symmetry sector. `Op_shift_sector()` handles this automatically.

        :red:`NOTE: One has to make sure that (i) the operator moves the state between the two sectors, and (ii) the two bases objects have the same symmetries. This function will not give the correct results otherwise.`

        Formally  equivalent to:

        >>> P1 = basis_sector_1.get_proj(np.complex128) # projector between full and initial basis
        >>> P2 = basis_sector_2.get_proj(np.complex128) # projector between full and target basis
        >>> v_in_full = P1.dot(v_in) # go from initial basis to to full basis
        >>> v_out_full = basis_full.inplace_Op(v_in_full,op_list,np.complex128) # apply Op
        >>> v_out = P2.T.conj().dot(v_out_full) # project to target basis

        Notes
        -----
        * particularly useful when computing correlation functions.
        * supports parallelization to multiple states listed in the columns of `v_in`.
        * the user is strongly advised to use the code under "Formally equivalent" above to check the results of this function for small system sizes.

        Parameters
        ----------
        other_basis : `basis` object
                `basis_general` object for the initial symmetry sector. Must be the same `basis` class type as the basis whose instance is `Op_shift_sector()` (i.e. the basis in `basis.Op_shift_sector()`).
        op_list : list
                Operator string list which defines the operator to apply. Follows the format `[["z",[i],Jz[i]] for i in range(L)], ["x",[i],Jx[j]] for j in range(L)],...]`.
        v_in : array_like, (other_basis.Ns,...)
                Initial state to apply the symmetry non-conserving operator on. Must have the same length as `other_basis.Ns`.
        v_out : array_like, (basis.Ns,...), optional
                Optional array to write the result for the final/target state in.
        dtype : numpy dtype for matrix elements, optional
                Data type (e.g. `numpy.float64`) to construct the operator with.

        Returns
        -------
        (basis.Ns, ) numpy.ndarray
                Array containing the state `v_out` in the current basis, i.e. the basis in `basis.Op_shift_sector()`.

        Examples
        --------

        >>> v_out = basis.Op_shift_sector(initial_basis, op_list, v_in)
        >>> print(v_out.shape, basis.Ns, v_in.shape, initial_basis.Ns)

        """

        # consider flag to do calc with projectors instead to use as a check.

        if not isinstance(other_basis, self.__class__):
            raise ValueError("other_basis must be the same type as the given basis.")

        if not self._made_basis:
            raise AttributeError(
                "this function requires the basis to be constructed first; use basis.make()."
            )

        if not other_basis._made_basis:
            raise AttributeError(
                "this function requires the basis to be constructed first; use basis.make()."
            )

        _, _, J_list = zip(*op_list)

        J_list = _np.asarray(J_list)

        if dtype is not None:
            J_list = J_list.astype(dtype)

        v_in = _np.asanyarray(v_in)

        result_dtype = _np.result_type(_np.float32, J_list.dtype, v_in.dtype)

        v_in = v_in.astype(result_dtype, order="C", copy=False)
        v_in = v_in.reshape((other_basis.Ns, -1))
        nvecs = v_in.shape[1]

        if v_in.shape[0] != other_basis.Ns:
            raise ValueError("invalid shape for v_in")

        if v_out is None:
            v_out = _np.zeros((self.Ns, nvecs), dtype=result_dtype, order="C")
        else:
            if v_out.dtype != result_dtype:
                raise TypeError("v_out does not have the correct data type.")
            if not v_out.flags["CARRAY"]:
                raise ValueError("v_out is not a writable C-contiguous array")
            if v_out.shape != (self.Ns, nvecs):
                raise ValueError("invalid shape for v_out")

        for opstr, indx, J in op_list:
            indx = _np.ascontiguousarray(indx, dtype=_np.int32)
            self._core.op_shift_sector(
                v_in,
                v_out,
                opstr,
                indx,
                J,
                self._basis,
                self._n,
                other_basis._basis,
                other_basis._n,
            )

        if nvecs == 1:
            return v_out.squeeze()
        else:
            return v_out

    def get_proj(self, dtype, pcon=False):
        """Calculates transformation/projector from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        * particularly useful when a given operation canot be carried out in the symmetry-reduced basis in a straightforward manner.

        * see also `Op_shift_sector()`.

        Parameters
        ----------
        dtype : 'type'
                Data type (e.g. numpy.float64) to construct the projector with.
        pcon : bool, optional
                Whether or not to return the projector to the particle number (magnetisation) conserving basis
                (useful in bosonic/single particle systems). Default is `pcon=False`.

        Returns
        -------
        scipy.sparse.csc_matrix
                Transformation/projector between the symmetry-reduced and the full basis.

        Examples
        --------

        >>> P = get_proj(np.float64,pcon=False)
        >>> print(P.shape)

        """

        if not self._made_basis:
            raise AttributeError(
                "this function requires the basis to be constructed first; use basis.make()."
            )

        basis_pcon = None
        Ns_full = self._sps**self._N

        if pcon and self._get_proj_pcon:

            if self._basis_pcon is None:
                self._basis_pcon = self.__class__(**self._pcon_args)

            basis_pcon = self._basis_pcon._basis
            Ns_full = basis_pcon.shape[0]
        elif pcon and self._get_proj_pcon:
            raise TypeError(
                "pcon=True only works for basis of a single particle number sector."
            )

        sign = _np.ones_like(self._basis, dtype=_np.int8)
        c = self._n.astype(dtype, copy=True)
        c *= self._pers.prod()
        _np.sqrt(c, out=c)
        _np.power(c, -1, out=c)
        index_type = _np.result_type(_np.min_scalar_type(-Ns_full), _np.int32)
        indptr = _np.arange(self._Ns + 1, dtype=index_type)
        indices = _np.arange(self._Ns, dtype=index_type)

        return self._core.get_proj(
            self._basis, dtype, sign, c, indices, indptr, basis_pcon=basis_pcon
        )

    def project_to(self, v0, sparse=True, pcon=False):
        """Transforms state from full (symmetry-free) basis to symmetry-reduced basis.

        Notes
        -----
        * particularly useful when a given operation cannot be carried out in the full basis.
        * supports parallelisation to multiple states listed in the columns.
        * inverse function to `project_from`.


        Parameters
        ----------
        v0 : numpy.ndarray
                Contains in its columns the states in the full (symmetry-free) basis.
        sparse : bool, optional
                Whether or not the output should be in sparse format. Default is `True`.
        pcon : bool, optional
                Whether or not to return the output in the particle number (magnetisation) conserving basis
                (useful in bosonic/single particle systems). Default is `pcon=False`.

        Returns
        -------
        numpy.ndarray
                Array containing the state `v0` in the symmetry-reduced basis.

        Examples
        --------

        >>> v_symm = project_to(v0)
        >>> print(v_symm.shape, v0.shape)

        """

        basis_pcon = None

        if pcon == True:
            if self._basis_pcon is None:
                self._basis_pcon = self.__class__(**self._pcon_args, make_basis=False)
                self._basis_pcon.make(N_p=0)

            basis_pcon = self._basis_pcon._basis

        if not self._made_basis:
            raise AttributeError(
                "this function requires the basis to be cosntructed first, see basis.make()."
            )

        if not hasattr(v0, "shape"):
            v0 = _np.asanyarray(v0)

        squeeze = False
        if pcon:
            Ns_full = basis_pcon.size
        else:
            Ns_full = self._sps**self._N

        if v0.ndim == 1:
            v0 = v0.reshape((-1, 1))
            shape = (self._Ns, 1)
            squeeze = True
        elif v0.ndim == 2:
            shape = (self._Ns, v0.shape[1])
        else:
            raise ValueError("excpecting v0 to have ndim > 0 and at most 2")

        if self._Ns <= 0:
            # CHECK later
            if sparse:
                return _sp.csr_matrix(
                    ([], ([], [])), shape=(self._Ns, 0), dtype=v0.dtype
                )
            else:
                return _np.zeros((self._Ns, 0), dtype=v0.dtype)

        if v0.shape[0] != Ns_full:
            raise ValueError(
                "v0 shape {0} not compatible with Ns_full={1}".format(v0.shape, Ns_full)
            )

        if _sp.issparse(v0):  # current work around for sparse states.
            # return self.get_proj(v0.dtype).dot(v0)
            raise ValueError

        v0 = _np.ascontiguousarray(v0)

        if sparse:
            # current work-around for sparse
            return self.get_proj(v0.dtype, pcon=pcon).T.dot(_sp.csr_matrix(v0))
        else:
            v_out = _np.zeros(
                shape,
                dtype=v0.dtype,
            )
            self._core.project_to_dense(
                self._basis, self._n, v0, v_out, basis_pcon=basis_pcon
            )
            if squeeze:
                return _np.squeeze(v_out)
            else:
                return v_out

    def get_vec(self, v0, sparse=True, pcon=False):
        """DEPRECATED (cf `project_from`). Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        This function is :red:`deprecated`. Use `project_from()` instead; see also the inverse function `project_to()`.

        """

        return self.project_from(v0, sparse=sparse, pcon=pcon)

    def project_from(self, v0, sparse=True, pcon=False):
        """Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

        Notes
        -----
        * particularly useful when a given operation cannot be carried out in the symmetry-reduced basis in a straightforward manner.
        * supports parallelisation to multiple states listed in the columns.
        * inverse function to `project_to`.

        Parameters
        ----------
        v0 : numpy.ndarray
                Contains in its columns the states in the symmetry-reduced basis.
        sparse : bool, optional
                Whether or not the output should be in sparse format. Default is `True`.
        pcon : bool, optional
                Whether or not to return the output in the particle number (magnetisation) conserving basis
                (useful in bosonic/single particle systems). Default is `pcon=False`.

        Returns
        -------
        numpy.ndarray
                Array containing the state `v0` in the full basis.

        Examples
        --------

        >>> v_full = project_from(v0)
        >>> print(v_full.shape, v0.shape)

        """

        basis_pcon = None

        if pcon == True:
            if self._basis_pcon is None:
                self._basis_pcon = self.__class__(**self._pcon_args, make_basis=False)
                self._basis_pcon.make(N_p=0)

            basis_pcon = self._basis_pcon._basis

        if not self._made_basis:
            raise AttributeError(
                "this function requires the basis to be cosntructed first, see basis.make()."
            )

        if not hasattr(v0, "shape"):
            v0 = _np.asanyarray(v0)

        squeeze = False
        if pcon:
            Ns_full = basis_pcon.size
        else:
            Ns_full = self._sps**self._N

        if v0.ndim == 1:
            v0 = v0.reshape((-1, 1))
            shape = (Ns_full, 1)
            squeeze = True
        elif v0.ndim == 2:
            shape = (Ns_full, v0.shape[1])
        else:
            raise ValueError("excpecting v0 to have ndim > 0 and at most 2")

        if self._Ns <= 0:
            if sparse:
                return _sp.csr_matrix(
                    ([], ([], [])), shape=(Ns_full, 0), dtype=v0.dtype
                )
            else:
                return _np.zeros((Ns_full, 0), dtype=v0.dtype)

        if v0.shape[0] != self._Ns:
            raise ValueError(
                "v0 shape {0} not compatible with Ns={1}".format(v0.shape, self._Ns)
            )

        if _sp.issparse(v0):  # current work around for sparse states.
            # return self.get_proj(v0.dtype).dot(v0)
            raise ValueError

        v0 = _np.ascontiguousarray(v0)

        if sparse:
            # current work-around for sparse
            return self.get_proj(v0.dtype, pcon=pcon).dot(_sp.csc_matrix(v0))
        else:
            v_out = _np.zeros(
                shape,
                dtype=v0.dtype,
            )
            self._core.project_from_dense(
                self._basis, self._n, v0, v_out, basis_pcon=basis_pcon
            )
            if squeeze:
                return _np.squeeze(v_out)
            else:
                return v_out

    def _check_symm(self, static, dynamic, photon_basis=None):
        if photon_basis is None:
            basis_sort_opstr = self._sort_opstr
            static_list, dynamic_list = self._get_local_lists(static, dynamic)
        else:
            basis_sort_opstr = photon_basis._sort_opstr
            static_list, dynamic_list = photon_basis._get_local_lists(static, dynamic)

        static_blocks = {}
        dynamic_blocks = {}
        for block, map in self._maps_dict.items():
            key = block + " symm"
            odd_ops, missing_ops = _check_symm_map(map, basis_sort_opstr, static_list)
            if odd_ops or missing_ops:
                static_blocks[key] = (tuple(odd_ops), tuple(missing_ops))

            odd_ops, missing_ops = _check_symm_map(map, basis_sort_opstr, dynamic_list)
            if odd_ops or missing_ops:
                dynamic_blocks[key] = (tuple(odd_ops), tuple(missing_ops))

        return static_blocks, dynamic_blocks

    def make(self, Ns_block_est=None, N_p=None):
        """Creates the entire basis by calling the basis constructor.

        Parameters
        ----------
        Ns_block_est: int, optional
                Overwrites the internal estimate of the size of the reduced Hilbert space for the given symmetries. This can be used to help conserve memory if the exact size of the H-space is known ahead of time.
        N_p: int, optional
                number of bits to use in the prefix label used to generate blocks for searching positions of representatives.

        Returns
        -------
        int
                Total number of states in the (symmetry-reduced) Hilbert space.

        Notes
        -----
        The memory stored in the basis grows exponentially as exactly :math:`2^{N_p+1}`. The default behavior is to use `N_p` such that
        the size of the stored information for the representative bounds is approximately as large as the basis. This is not as effective
        for basis which small particle numbers as the blocks have very uneven sizes. To not use the blocks just set N_p=0.

        Examples
        --------

        >>> N, Nup = 8, 4
        >>> basis=spin_basis_general(N,Nup=Nup,make_basis=False)
        >>> print(basis)
        >>> basis.make()
        >>> print(basis)

        """

        if Ns_block_est is not None:
            if Ns_block_est > self._Ns_block_est:
                Ns = Ns_block_est
            else:
                Ns = self._Ns_block_est
        else:
            Ns = max([self._Ns, 1000, self._Ns_block_est])

        # preallocate variables
        basis = basis_zeros(Ns, dtype=self._basis_dtype)
        n = _np.zeros(Ns, dtype=self._n_dtype)

        # make basis
        if self._count_particles and (self._Np is not None):
            Np_list = _np.zeros_like(basis, dtype=_np.uint8)
            Ns = self._core.make_basis(basis, n, Np=self._Np, count=Np_list)
        else:
            Np_list = None
            Ns = self._core.make_basis(basis, n, Np=self._Np)

        if Ns < 0:
            raise ValueError(
                "estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size."
            )

        if Ns > 0:
            if _is_sorted_decending(basis[:Ns]):
                self._basis = basis[:Ns].copy()
                self._n = n[:Ns].copy()
                if Np_list is not None:
                    self._Np_list = Np_list[:Ns].copy()
            else:
                indices = _basis_argsort(basis[:Ns])
                self._basis = basis[indices]
                self._n = n[indices]
                if Np_list is not None:
                    self._Np_list = Np_list[indices]
        else:
            self._basis = _np.array([], dtype=basis.dtype)
            self._n = _np.array([], dtype=n.dtype)
            if Np_list is not None:
                self._Np_list = _np.array([], dtype=Np_list.dtype)

        self._Ns = Ns
        self._Ns_block_est = Ns

        self._index_type = _get_index_type(self._Ns)
        self._reduce_n_dtype()

        self._made_basis = True
        self.make_basis_blocks(N_p=N_p)

    def make_basis_blocks(self, N_p=None):
        """Creates/modifies the bounds for representatives based on prefix tages.

        Parameters
        ----------
        N_p: int, optional
                number of bits to use in the prefix label used to generate blocks for searching positions of representatives.

        Notes
        -----
        The memory stored in the basis grows exponentially as exactly :math:`2^{N_p+1}`. The default behavior is to use `N_p` such that
        the size of the stored information for the representative bounds is approximately as large as the basis. This is not as effective
        for basis which small particle numbers as the blocks have very uneven sizes. To not use the blocks just set N_p=0.

        Examples
        --------

        >>> N, Nup = 8, 4
        >>> basis=spin_basis_general(N,Nup=Nup,make_basis=False)
        >>> print(basis)
        >>> basis.make()
        >>> print(basis)

        """
        if not self._made_basis:
            raise ValueError(
                "reference states are not constructed yet. basis must be constructed before calculating blocks"
            )

        sps = self.sps
        if sps is None:
            sps = 2

        if N_p is None:
            N_p = int(_np.floor(_np.log(self._Ns // 2 + 1) / _np.log(sps)))
        else:
            N_p = int(N_p)

        if len(self._pers) == 0 and self._Np is None:
            N_p = 0  # do not use blocks for full basis

        self._N_p = min(max(N_p, 0), self.N)

        if self._N_p > 0:
            self._basis_begin, self._basis_end = self._core.make_basis_blocks(
                self._N_p, self._basis
            )
        else:
            self._basis_begin = _np.array([], dtype=_np.intp)
            self._basis_end = _np.array([], dtype=_np.intp)

    def Op_bra_ket(self, opstr, indx, J, dtype, ket_states, reduce_output=True):
        """Finds bra states which connect given ket states by operator from a site-coupling list and an operator string.

        Given a set of ket states :math:`|s\\rangle`, the function returns the bra states :math:`\\langle s'|` which connect to them through an operator, together with the corresponding matrix elements.

        Notes
        -----
                * Similar to `Op` but instead of returning the matrix indices (row,col), it returns the states (bra,ket) in integer representation.
                * Does NOT require the full basis (see `basis` optional argument `make_basis`).
                * If a state from `ket_states` does not have a non-zero matrix element, it is removed from the returned list. See otional argument `reduce_output`.

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
                Data type (e.g. numpy.float64) to construct the matrix elements with.
        ket_states : numpy.ndarray(int)
                Ket states in integer representation. Must be of same data type as `basis`.
        reduce_output: bool, optional
                If set to `False`, the returned arrays have the same size as `ket_states`; If set to `True` zeros are purged.

        Returns
        -------
        tuple
                `(ME,bra,ket)`, where
                        * numpy.ndarray(scalar): `ME`: matrix elements of type `dtype`, which connects the ket and bra states.
                        * numpy.ndarray(int): `bra`: bra states, obtained by applying the matrix representing the operator in the lattice basis,
                                to the ket states, such that `bra[i]` corresponds to `ME[i]` and connects to `ket[i]`.
                        * numpy.ndarray(int): `ket`: ket states, such that `ket[i]` corresponds to `ME[i]` and connects to `bra[i]`.


        Examples
        --------

        >>> J = 1.41
        >>> indx = [2,3]
        >>> opstr = "zz"
        >>> dtype = np.float64
        >>> ME, bra, ket = Op_bra_ket(opstr,indx,J,dtype,ket_states)

        """

        indx = _np.asarray(indx, dtype=_np.int32)
        ket_states = _np.array(ket_states, dtype=self._basis.dtype, ndmin=1)

        if len(opstr) != len(indx):
            raise ValueError("length of opstr does not match length of indx")

        if _np.any(indx >= self._N) or _np.any(indx < 0):
            raise ValueError("values in indx falls outside of system")

        extra_ops = set(opstr) - self._allowed_ops
        if extra_ops:
            raise ValueError(
                "unrecognized characters {} in operator string.".format(extra_ops)
            )

        bra = _np.zeros_like(ket_states)  # row
        ME = _np.zeros(ket_states.shape[0], dtype=dtype)

        self._core.op_bra_ket(ket_states, bra, ME, opstr, indx, J, self._Np)

        if reduce_output:
            # remove nan's matrix elements
            mask = _np.logical_not(_np.logical_or(_np.isnan(ME), _np.abs(ME) == 0.0))
            bra = bra[mask]
            ket_states = ket_states[mask]
            ME = ME[mask]
        else:
            mask = _np.isnan(ME)
            ME[mask] = 0.0

        return ME, bra, ket_states

    def representative(self, states, out=None, return_g=False, return_sign=False):
        """Maps states to their representatives under the `basis` symmetries.

        Parameters
        ----------
        states : array_like(int)
                Fock-basis (z-basis) states to find the representatives of. States are stored in integer representations.
        out : numpy.ndarray(int), optional
                variable to store the representative states in. Must be a `numpy.ndarray` of same datatype as `basis`, and same shape as `states`.
        return_g : bool, optional
                if set to `True`, the function also returns the integer `g` corresponding to the number of times each basis symmetry needs to be applied to a given state to obtain its representative.
        return_sign : bool, optional
                if set to `True`, the function returns the `sign` of the representative relative to the original state (nontrivial only for fermionic bases).

        Returns
        -------
        tuple
                ( representatives, g_array, sign_array )
                * array_like(int): `representatives`: Representatives under `basis` symmetries, corresponding to `states`.
                * array_like(int): `g_array` of size (number of states, number of symmetries). Requires `return_g=True`. Contains integers corresponding to the number of times each basis symmetry needs to be applied to a given state to obtain its representative.
                * array_like(int): `sign_array` of size (number of states,). Requires `return_sign=True`. Contains `sign` of the representative relative to the original state (nontrivial only for fermionic bases).

        Examples
        --------

        >>> basis=spin_basis_general(N,Nup=Nup,make_basis=False)
        >>> s = 17
        >>> r = basis.representative(s)
        >>> print(s,r)

        """

        states = _np.asarray(states, order="C", dtype=self._basis.dtype)
        states = _np.atleast_1d(states)

        if states.ndim != 1:
            raise TypeError("dimension of array_like states must not exceed 1.")

        if return_g:
            g_out = _np.zeros(
                (states.shape[0], self._qs.shape[0]), dtype=_np.int32, order="C"
            )

        if return_sign:
            sign_out = _np.zeros(states.shape, dtype=_np.int8, order="C")

        if out is None:
            out = _np.zeros(states.shape, dtype=self._basis.dtype, order="C")

            if return_g and return_sign:
                self._core.representative(states, out, g_out=g_out, sign_out=sign_out)
                return out, g_out, sign_out
            elif return_g:
                self._core.representative(states, out, g_out=g_out)
                return out, g_out
            elif return_sign:
                self._core.representative(states, out, sign_out=sign_out)
                return out, sign_out
            else:
                self._core.representative(states, out)
                return out

        else:
            if not isinstance(out, _np.ndarray):
                raise TypeError("out must be a numpy.ndarray")
            if states.shape != out.shape:
                raise TypeError("states and out must have same shape.")
            if out.dtype != self._basis.dtype:
                raise TypeError("out must have same type as basis")
            if not out.flags["CARRAY"]:
                raise ValueError("out must be C-contiguous array.")

            if return_g and return_sign:
                self._core.representative(states, out, g_out=g_out, sign_out=sign_out)
                return g_out, sign_out
            elif return_g:
                self._core.representative(states, out, g_out=g_out)
                return g_out
            elif return_sign:
                self._core.representative(states, out, sign_out=sign_out)
                return sign_out
            else:
                self._core.representative(states, out)

    def normalization(self, states, out=None):
        """Computes normalization of `basis` states.

        Notes
        -----
                * Returns zero, if the state is not part of the symmetry-reduced basis.
                * The normalizations can be used to compute matrix elements in the symmetry-reduced basis.

        Parameters
        ----------
        states : array_like(int)
                Fock-basis (z-basis) states to find the normalizations of. States are stored in integer representations.
        out : numpy.ndarray(unsigned int), optional
                variable to store the normalizations of the states in. Must be a `numpy.ndarray` of datatype `unsigned int` (e.g. `numpy.uint16`), and same shape as `states`.

        Returns
        -------
        array_like(int)
                normalizations of `states` for the given (symmetry-reduced) `basis`.

        Examples
        --------

        >>> basis=spin_basis_general(N,Nup=Nup,make_basis=False)
        >>> s = 17
        >>> norm_s = basis.normalization(s)
        >>> print(s,norm_s)

        """

        states = _np.asarray(states, order="C", dtype=self._basis.dtype)
        states = _np.atleast_1d(states)

        if out is None:
            # determine appropriate dtype
            out_dtype = _np.min_scalar_type(
                _np.iinfo(self._n_dtype).max * self._pers.prod()
            )
            out = _np.zeros(states.shape, dtype=out_dtype)
            self._core.normalization(states, out)

            # reduce dtype
            out_dtype = _np.min_scalar_type(out.max())
            out = out.astype(out_dtype)

            return out.squeeze()

        else:
            if states.shape != out.shape:
                raise TypeError("states and out must have same shape.")

            if _np.issubdtype(out.dtype, _np.signedinteger):
                raise TypeError(
                    "out must have datatype numpy.uint8, numpy.uint16, numpy.uint32, or numpy.uint64."
                )

            if not out.flags["CARRAY"]:
                raise ValueError("out must be C-contiguous array.")

            self._core.normalization(states, out)

            out_dtype = _np.min_scalar_type(out.max())
            out = out.astype(out_dtype)

    def get_amp(self, states, out=None, amps=None, mode="representative"):
        """Computes the rescale factor of state amplitudes between the symmetry-reduced and full basis.

        Given a quantum state :math:`s` and a state amplitude in the full basis :math:`\\psi_s`, its representative (under the symemtries)
        :math:`r(s)` with a corresponding amplitude :math:`\\psi^\\text{sym}_r`, the function computes the ratio :math:`C`, defined as

        .. math::
                \\psi_s = C\\psi_r^\\text{sym}


        Notes
        -----
                * Particularly useful when a given operation cannot be carried away in the symmetry-reduced basis in a straightforward manner.
                * To transform an entire state from a symmetry-reduced basis to the full (symmetry-free) basis, use the `basis.get_vec()` function.
                * Returns zero, if the state passed to the function is not part of the symmetry-reduced basis.
                * If `amps` is passed, the user has to make sure that the input data in `amps` correspond to the `states`.
                * The function assumes that `states` comply with the particle conservation symmetry the `basis` was constructed with.

        Parameters
        ----------
        states : array_like(int)
                Fock-basis (z-basis) states to find the amplitude rescale factor :math:`C` of. States are stored in integer representations.
        out : numpy.ndarray(float), optional
                variable to store the rescale factors :math:`C` of the states in. Must be a real or complex-valued `numpy.ndarray` of the same shape as `states`.
        amps : numpy.ndarray(float), optional
                array of amplitudes to rescale by the amplitude factor :math:`C` (see `mode`). Updated in-place. Must be a real or complex-valued `numpy.ndarray` of the same shape as `states`.
        mode : string, optional
                * if `mode='representative'` (default), then the function assumes that
                        (i) `states` already contains representatives (i.e. states in the symmetry-reduced basis);
                        (ii) `amps` (if passed) are amplitudes in the symmetry-reduced basis (:math:`\\psi_r^\\text{symm}`). The function will update `amps` in-place to :math:`\\psi_s`.
                * if `mode='full_basis'`, then the function assumes that
                        (i) `states` contains full-basis states (the funciton will compute the corresponding representatives);
                        (ii) `amps` (if passed) are amplitudes in the full basis (:math:`\\psi_s`). The function will update `amps` in-place to :math:`\\psi_r^\\text{symm}`;
                                **Note**: the function will also update the variable `states` in place with the corresponding representatives.

        Returns
        -------
        array_like(float)
                amplitude rescale factor :math:`C` (see expression above).

        Examples
        --------

        >>> C = get_amp(states,out=None,amps=None,mode='representative')

        """

        states = _np.asarray(states, order="C", dtype=self._basis.dtype)
        states = _np.atleast_1d(states)

        states_shape = states.shape

        if out is not None:
            if states_shape != out.shape:
                raise TypeError("states and out must have same shape.")
            if out.dtype not in [
                _np.float32,
                _np.float64,
                _np.complex64,
                _np.complex128,
            ]:
                raise TypeError(
                    "out must have datatype numpy.float32, numpy.float64, numpy.complex64, or numpy.complex128."
                )
            if not out.flags["CARRAY"]:
                raise ValueError("out must be C-contiguous array.")
        elif amps is not None:
            out = _np.zeros(states_shape, dtype=amps.dtype)
        else:
            out = _np.zeros(states_shape, dtype=_np.complex128)

        self._core.get_amp(states, out, states_shape[0], mode)

        if amps is not None:
            if states.shape != amps.shape:
                raise TypeError("states and amps must have same shape.")

            if mode == "representative":
                amps *= out  # compute amplitudes in full basis
            elif mode == "full_basis":
                amps /= out  # compute amplitudes in symmetery-rduced basis
            else:
                raise ValueError(
                    "mode accepts only the values 'representative' and 'full_basis'."
                )

        return out.squeeze()


def _check_symm_map(map, sort_opstr, operator_list):
    missing_ops = []
    odd_ops = []
    for op in operator_list:
        opstr = str(op[0])
        indx = list(op[1])
        J = op[2]
        for j, ind in enumerate(op[1]):
            i = map[ind]
            if i < 0:
                if opstr[j] == "n":
                    odd_ops.append(op)

                J *= -1 if opstr[j] in ["z", "y"] else 1
                opstr = opstr.replace("+", "#").replace("-", "+").replace("#", "-")
                i = -(i + 1)

            indx[j] = i

        new_op = list(op)
        new_op[0] = opstr
        new_op[1] = indx
        new_op[2] = J

        new_op = sort_opstr(new_op)
        if not (new_op in operator_list):
            missing_ops.append(new_op)

    return odd_ops, missing_ops
