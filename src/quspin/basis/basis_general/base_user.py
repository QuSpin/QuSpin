from quspin.basis.basis_general.base_general import basis_general
from quspin_extensions.basis.basis_general._basis_general_core import user_core_wrap
import numpy as _np
from numba import cfunc, types, njit

try:
    from numba.ccallback import CFunc  # numba < 0.49.0
except ModuleNotFoundError:
    from numba.core.ccallback import CFunc  # numba >= 0.49.0

map_sig_32 = types.uint32(
    types.uint32, types.intc, types.CPointer(types.int8), types.CPointer(types.uint32)
)
map_sig_64 = types.uint64(
    types.uint64, types.intc, types.CPointer(types.int8), types.CPointer(types.uint64)
)

next_state_sig_32 = types.uint32(
    types.uint32, types.uint32, types.uint32, types.CPointer(types.uint32)
)
next_state_sig_64 = types.uint64(
    types.uint64, types.uint64, types.uint64, types.CPointer(types.uint64)
)

pre_check_state_sig_32 = types.uint32(
    types.uint32, types.uint32, types.CPointer(types.uint32)
)
pre_check_state_sig_64 = types.uint64(
    types.uint64, types.uint64, types.CPointer(types.uint64)
)

op_results_32 = types.Record.make_c_struct(
    [
        ("matrix_ele", types.complex128),
        ("state", types.uint32),
    ]
)

op_results_64 = types.Record.make_c_struct(
    [("matrix_ele", types.complex128), ("state", types.uint64)]
)

op_sig_32 = types.intc(
    types.CPointer(op_results_32),
    types.char,
    types.intc,
    types.intc,
    types.CPointer(types.uint32),
)
op_sig_64 = types.intc(
    types.CPointer(op_results_64),
    types.char,
    types.intc,
    types.intc,
    types.CPointer(types.uint64),
)

count_particles_sig_32 = types.void(
    types.uint32, types.CPointer(types.intc), types.CPointer(types.intc)
)
count_particles_sig_64 = types.void(
    types.uint64, types.CPointer(types.intc), types.CPointer(types.intc)
)

__all__ = [
    "map_sig_32",
    "map_sig_64",
    "next_state_sig_32",
    "next_state_sig_64",
    "op_func_sig_32",
    "op_func_sig_64",
    "user_basis",
]

# @njit
# def _is_sorted_decending(a):
# 	for i in range(a.size-1):
# 		if(a[i]<a[i+1]):
# 			return False

# 	return True


def _process_user_blocks(use_32bit, blocks_dict, block_order):

    if not all((type(v) is tuple) and (len(v) == 4) for v in blocks_dict.values()):
        raise ValueError(
            "blocks must contain tuple with a CFunc, the period of the symmetry, the quantum number, and the extra arguments."
        )

    if not all(isinstance(f, CFunc) for f, _, _, _ in blocks_dict.values()):
        raise ValueError("map_func must be instance of numba.CFunc.")

    if use_32bit:
        if not all(f._sig == map_sig_32 for f, _, _, _ in blocks_dict.values()):
            raise ValueError(
                "map_func does not have the correct signature, \
					try using map_sig_32 from quspin.basis.user module."
            )
    else:
        if not all(f._sig == map_sig_64 for f, _, _, _ in blocks_dict.values()):
            raise ValueError(
                "map_func does not have the correct signature, \
					try using map_sig_64 from quspin.basis.user module."
            )

    if block_order is None:  # sort by periodicies largest to smallest
        sorted_items = sorted(blocks_dict.items(), key=lambda x: x[1][1])
        sorted_items.reverse()
    else:
        block_order = list(block_order)
        missing = set(blocks_dict.keys()) - set(block_order)
        if len(missing) > 0:
            raise ValueError(
                "{} names found in block names but missing from block_order.".format(
                    missing
                )
            )

        missing = set(block_order) - set(blocks_dict.keys())
        if len(missing) > 0:
            raise ValueError(
                "{} names found in block_order but missing from block names.".format(
                    missing
                )
            )

        sorted_items = [(key, blocks_dict[key]) for key in block_order]

    if len(sorted_items) > 0:

        blocks = {
            block: ((-1) ** q if per == 2 else q)
            for block, (_, per, q, _) in sorted_items
        }

        _, items = zip(*sorted_items)
        map_funcs, pers, qs, map_args = zip(*items)

        # exit()

        return blocks, map_funcs, pers, qs, map_args

    else:
        return {}, [], [], [], []


def _noncommuting_bits(N, noncommuting_bits):

    completed_noncommuting_bits = []

    for bits, swap_phase in noncommuting_bits:
        bits = _np.asarray(bits)
        swap_phase = _np.asarray(swap_phase)

        if not _np.issubdtype(bits.dtype, int):
            raise TypeError("site list most contain only integers.")

        if _np.array(swap_phase).ndim != 0:
            raise ValueError("swap_phase must be scalar.")

        if _np.any(_np.logical_and(bits < 0, bits >= N)):
            raise ValueError("bit number outside of range of system.")

        if swap_phase == -1:
            swap_phase = swap_phase.astype(_np.int8)
        elif np.abs(swap_phase) != 1:
            raise ValueError("must have |swap_phase|==1")
        else:
            if swap_phase == 1:
                continue  # skip
            else:
                # swap_phase = swap_phase.astype(_np.complex128)
                raise NotImplementedError(
                    "Abealian anyons are not supported for user_basis."
                )

        completed_noncommuting_bits.append((bits, swap_phase))

    return completed_noncommuting_bits


class user_basis(basis_general):
    """Constructs basis for USER-DEFINED functionality of a basis object.


    The `user_basis` unveils the inner workings of QuSpin. This is the most advanced usage of the package, and requires some understanding of python,
    the `numba` package used to interface QuSpin's underlying cpp code with python, and some experience with bitwise operations to manipulate integers.

    Since we believe that the users will benefit from a more detailed discussion on how the `user_basis` is intended to work, we also provide a detailed
    tutorial: :ref:`user_basis-label`, which covers the general concepts and provides six complete examples of various complexity.

    Examples
    --------

    The following example shows how to use the `user_basis` class to construct the Hamiltonian

    .. math::

            H = \\sum_j P_{j-1}\\sigma^x_j P_{j+1},\\quad P_j = |\\downarrow_j\\rangle\\langle\\downarrow_j|

    using translation and reflection symmetry. The projector operator :math:`P_j`, which only allows a spin-up state in the basis to be preceded and succeeded by a spin-down,
    is incorporated by constructing the corresponding `user_basis` object. One can then just build the Hamiltonian :math:`H=\\sum_j\\sigma^x_j` in the
    constrained Hilbert space.

    More examples (including explanations of the class methods and attributes) can be found at: :ref:`user_basis-label`.

    .. literalinclude:: ../../doc_examples/user_basis-example.py
            :linenos:
            :language: python
            :lines: 11-

    """

    def __init__(
        self,
        basis_dtype,
        N,
        op_dict,
        sps=2,
        pcon_dict=None,
        pre_check_state=None,
        allowed_ops=None,
        parallel=False,
        Ns_block_est=None,
        _make_basis=True,
        block_order=None,
        noncommuting_bits=[],
        _Np=None,
        **blocks,
    ):
        """Intializes the `user_basis_general` object (basis for user defined ED calculations).

        Parameters
        ----------
        basis_dtype: numpy.dtype object
                the data type used to represent the states in the basis: must be either uint32 or uint64.
        N: int
                Number of sites.
        op_dict: dict
                used to define the `basis.Op` function; the dictionary contais the following items:
                        * **op(op_struct_ptr,op_str,site_ind,N,args): numba.CFunc object**
                                This is a numba-compiled function (CFunc) which calculates the matrix elements :math:`\\mathrm{me}` given a state :math:`|s\\rangle` together with a character to
                                represent the operator, an integer `site_ind` specifying the site of that local operator, the total number of sites `N`, and a set of optional `uint`-dtype arguments `args`. See the above example for how
                                one would use this for spin-1/2 system.
                        * **op_args: np.ndarray[basis_dtype]**
                                used to pass the arguments `args` to the CFunc `op(...,args)`. The corresponding key must be a `np.ndarray[basis_dtype]`.
        pcon_dict: dict, optional
                This dictionary contains the following items which are required to use particle conservation in this basis:
                        *minimum requirements*:
                                * **Np: tuple/int, list(tuple/int)**
                                        specifies the particle sector(s).
                                * **next_state(s,counter,N,args): numba.CFunc object**
                                        given a quantum state :math:`|s\\rangle` in the integer-representation `s`, this CFunc generates the next lexicographically ordered particle conservation state.
                                        `counter` is an intrinsic variable which increments by unity every time the function is called, `N` is the total number of lattice sites, and `args` holds any optional arguments stored in a `np.ndarray[basis_dtype]`.
                                * **next_state_args: np.ndarray(basis_dtype)**
                                        optional arguments for `next_state(...,args)`.
                                * **get_Ns_pcon(N,Np): python function**
                                        when called as get_Ns_pcon(N,Np), this python function returns the size of the symmetery-free particle conservation basis, given the `N` lattice sites and `Np` (see above).
                                * **get_s0_pcon(N,Np): python function**
                                        when called as get_s0_pcon(N,Np), this python function returns the starting state to generate the whole particle conservation basis by repeatedly calling `next_state()`.
                        *advanced requirements* to access `basis.Op_bra_ket()` functionality (on top of the minimum requirements):
                                * **n_sectors: int, list(int)**
                                        number of integers which parameterize the particle sectors, e.g. with spinful fermions there is a particle number for both the up and the down sectors, and hence `n_sectors=2`.
                                * **count_particles(s,p_number_ptr,args): numba.CFunc object**
                                        For a quantum state `s` in the integer representation, this CFunc counts the number of particles in each particle sector and places them into a pointer `p_number_ptr` (`count_particles` does **not** return any output). The pointer provided will have `n_sector` slots of memory allocated. The components of the pointer `p_number_ptr` must correspond to the ordering of `Np`. The integer `s` cannot be changed.
                                * **count_particles_args: np.ndarray(int)**
                                        compulsory arguments for `count_particles(...,args)` (whenever used).
        pre_check_state(s,N,args): numba.CFunc object or tuple(numba.CFunc object,ndarray(C-contiguous,dtype=basis_dtype)), optional
                This CFunc allows the user to specify a boolean criterion used to discard/filter states from the basis. In the low-level code, this function is applied before checking if a given state is
                representative state (i.e. belogs to a given symmetry sector) or not. This allows the user to, e.g., enforce a local Hilbert-space constraint
                (e.g. for a spinful fermion basis to never have a doubly occupied site). One can pass additional arguments `args` using a `np.ndarray[basis_dtype]`.
        allowed_ops: list/set, optional
                A list of allowed characters, each of which is to be passed in to the `op` in the form of `op_str` (see above).
        parallel: bool, optional
                turns on parallel code when constructing the basis even when no symmetries are implemented. Useful when constructing highly constrained Hilbert spaces with pre_check_state.
        sps: int, optional
                The number of states per site (i.e. the local on-site Hilbert space dimension).
        Ns_full: int, optional
                Total number of states in the Hilbert space without any symmetries. For a single species this value is `sps**N`
        Ns_block_est: int, optional
                An estimate for the size of the symmetry reduced block, QuSpin does a simple estimate which is not always correct.
        block_order: tuple/list, optional
                A list of strings containing the names of the symmetry blocks which specifies the order in which the symmetries will be applied to the state when calculating the basis. The first element in the list is applied to the state first followed by the second element, etc. If the list is not specificed the ordering is such that the symmetry with the largest cycle is the first, followed by the second largest, etc.
        noncommuting_bits: list, optional
                A list of tuples specifying if bits belong to a group of sites that do not commute. The first element in each tuple represents the group of sites, and the second element represents the phase-factor that is given during the exchange.
        **blocks: optional
                keyword arguments which pass the symmetry generator arrays. For instance:

                >>> basis(...,kxblock=(CFunc,m_Q,q,args),...)

                The key names of the symmetry sector, e.g. `kxblock`, can be defined arbitrarily by the user. The
                values are tuples where the first entry contains the numba-CFunc which generates the symmetry transformation :math:`Q`
                acting on the state (see class example), the second entry is an integer :math:`m_Q` which gives the periodicity
                of the symmetry sector (:math:`Q^{m_Q} = 1`), and :math:`q` is the quantum number for the given sector. Optional arguments can be passed using the`args` argument which is a `np.ndarray[basis_dtype]`. Note that if the periodicity is wrong
                the basis will give undefined behavior.
        """

        # photon basis not supported hence this flag is always False.
        self._count_particles = False
        if _Np is not None:
            raise ValueError("cannot use photon basis with user_basis_general.")

        # disable checks for this basis.
        self._check_herm = None
        self._check_pcon = None
        self._check_symm = None

        # this basis only supports unique matrix elements.
        self._unique_me = True

        # no particle conservation basis created at this point.
        self._basis_pcon = None
        self._get_proj_pcon = False
        self._made_basis = False  # keeps track of whether the basis has been made

        self._noncommuting_bits = _noncommuting_bits(N, noncommuting_bits)
        Ns_full = sps**N
        self._N = N
        if basis_dtype not in [_np.uint32, _np.uint64]:
            raise ValueError(
                "basis_dtype must be either uint32 or uint64 for the given basis representation."
            )

        use_32bit = basis_dtype == _np.uint32

        if (
            not all(
                isinstance(map_args, _np.ndarray)
                for _, _, _, map_args in blocks.values()
            )
            or not all(
                map_args.flags["CARRAY"] for _, _, _, map_args in blocks.values()
            )
            or not all(
                map_args.dtype == basis_dtype for _, _, _, map_args in blocks.values()
            )
        ):

            raise ValueError(
                "map_args must be a C-contiguous numpy array with dtype {}".format(
                    basis_dtype
                )
            )

        if type(op_dict) is dict:
            if len(op_dict) != 2:
                raise ValueError("op_dict must contain exactly two items.")
            else:
                op_func = op_dict["op"]
                op_args = op_dict["op_args"]

                if not isinstance(op_func, CFunc):
                    raise ValueError("op_func must be a numba.CFunc object.")

                if not isinstance(op_args, _np.ndarray):
                    raise ValueError(
                        "op_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
                if not op_args.flags["CARRAY"]:
                    raise ValueError(
                        "op_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
                if op_args.dtype != basis_dtype:
                    raise ValueError(
                        "op_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
        else:
            raise ValueError("op_dict input not understood.")

        if use_32bit:
            if op_func._sig != op_sig_32:
                raise ValueError(
                    "op_func does not have the correct signature, \
					try using op_sig_32 from quspin.basis.user module."
                )
        else:
            if op_func._sig != op_sig_64:
                raise ValueError(
                    "op_func does not have the correct signature, \
					try using op_sig_64 from quspin.basis.user module."
                )

        if pcon_dict is not None:

            if type(pcon_dict) is dict:

                next_state_args = pcon_dict["next_state_args"]

                if not isinstance(next_state_args, _np.ndarray):
                    raise ValueError(
                        "next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
                if not next_state_args.flags["CARRAY"]:
                    raise ValueError(
                        "next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
                if next_state_args.dtype != basis_dtype:
                    raise ValueError(
                        "next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )

                if len(pcon_dict) == 5:  # basic usage
                    Np = pcon_dict["Np"]
                    next_state = pcon_dict["next_state"]
                    get_Ns_pcon = pcon_dict["get_Ns_pcon"]
                    get_s0_pcon = pcon_dict["get_s0_pcon"]
                    n_sectors = None
                    count_particles = None
                    count_particles_args = None

                elif len(pcon_dict) == 8:
                    Np = pcon_dict["Np"]
                    next_state = pcon_dict["next_state"]
                    get_Ns_pcon = pcon_dict["get_Ns_pcon"]
                    get_s0_pcon = pcon_dict["get_s0_pcon"]
                    n_sectors = pcon_dict["n_sectors"]
                    count_particles = pcon_dict["count_particles"]
                    count_particles_args = pcon_dict["count_particles_args"]

                    if not isinstance(count_particles_args, _np.ndarray):
                        raise ValueError(
                            "count_particles_args must be a C-contiguous numpy \
							array with dtype {} or {}".format(
                                _np.int32, _np.int64
                            )
                        )
                    if not count_particles_args.flags["CARRAY"]:
                        raise ValueError(
                            "count_particles_args must be a C-contiguous numpy \
							array with dtype {} or {}".format(
                                _np.int32, _np.int64
                            )
                        )
                    if (
                        count_particles_args.dtype != _np.int32
                        and count_particles_args.dtype != _np.int64
                    ):
                        raise ValueError(
                            "count_particles_args must be a C-contiguous numpy \
							array with dtype {} or {}".format(
                                _np.int32, _np.int64
                            )
                        )

                else:
                    raise ValueError(
                        "pcon_dict input not understood. Check if you passed all keys (currently pcon_dict can have either 5 or 8 keys."
                    )
            else:
                raise ValueError("pcon_dict input not understood.")

            if Np is None:
                Ns = Ns_full
            elif type(Np) is tuple or type(Np) is int:
                self._get_proj_pcon = True
                if n_sectors is not None:
                    if type(Np) is int and n_sectors != 1:
                        raise ValueError(
                            "n_sectors is {} when the size \
							of the particle sector is 1".format(
                                n_sectors
                            )
                        )
                    elif type(Np) is tuple and n_sectors != len(Np):
                        raise ValueError(
                            "n_sectors is {} when the size \
							of the particle sector is {}".format(
                                n_sectors, len(np)
                            )
                        )
                    # else:
                    # 	raise ValueError("Np must be tuple, int, or a list of tuples/integers.")
                else:
                    if type(Np) is int:
                        n_sectors = 1
                    elif type(Np) is tuple:
                        n_sectors = len(Np)
                    else:
                        raise ValueError(
                            "Np must be tuple, int, or a list of tuples/integers."
                        )

                Ns = get_Ns_pcon(N, Np)
            else:
                try:
                    Np_iter = iter(Np)
                except TypeError:
                    raise TypeError("Np must be integer or iteratable object.")

                Np = list(Np)

                for np in Np:
                    if n_sectors is not None:
                        if type(np) is int and n_sectors != 1:
                            raise ValueError(
                                "n_sectors is {} when the size \
								of the particle sector is 1".format(
                                    n_sectors
                                )
                            )
                        elif type(np) is tuple and n_sectors != len(np):
                            raise ValueError(
                                "n_sectors is {} when the size \
								of the particle sector is {}".format(
                                    n_sectors, len(np)
                                )
                            )
                        else:
                            raise ValueError(
                                "Np must be tuple, int, or a list of tuples/integers."
                            )
                    else:
                        if type(np) is int:
                            n_sectors = 1
                        elif type(np) is tuple:
                            n_sectors = len(np)
                        else:
                            raise ValueError(
                                "Np must be tuple, int, or a list of tuples/integers."
                            )

                Ns = sum(get_Ns_pcon(N, np) for np in Np)

        else:
            self._get_proj_pcon = False
            Ns = Ns_full
            Np = None
            next_state_args = None
            next_state = None
            count_particles = None
            count_particles_args = None
            get_s0_pcon = None
            get_Ns_pcon = None
            n_sectors = -1

        # check_state function BEFORE symmetry checks
        # this can be used to impose local hilbert space constraints.
        if pre_check_state is not None:
            try:
                pre_check_state, check_state_nosymm_args = pre_check_state
            except TypeError:
                check_state_nosymm_args = None

            if not isinstance(pre_check_state, CFunc):
                raise ValueError("pre_check_state must be a numba.CFunc object.")

            if use_32bit:
                if pre_check_state._sig != pre_check_state_sig_32:
                    raise ValueError(
                        "pre_check_state does not have the correct signature, \
					try using pre_check_state_sig_32 from quspin.basis.user module."
                    )
            else:
                if pre_check_state._sig != pre_check_state_sig_64:
                    raise ValueError(
                        "pre_check_state does not have the correct signature, \
					try using pre_check_state_sig_64 from quspin.basis.user module."
                    )

            if check_state_nosymm_args is not None:
                if not isinstance(check_state_nosymm_args, _np.ndarray):
                    raise ValueError(
                        "next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
                if not check_state_nosymm_args.flags["CARRAY"]:
                    raise ValueError(
                        "next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
                if check_state_nosymm_args.dtype != basis_dtype:
                    raise ValueError(
                        "next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
        else:
            pre_check_state, check_state_nosymm_args = None, None

        if next_state is not None:
            if not isinstance(next_state, CFunc):
                raise ValueError("next_state must be a numba.CFunc object.")

            if use_32bit:
                if next_state._sig != next_state_sig_32:
                    raise ValueError(
                        "next_state does not have the correct signature, \
					try using next_state_sig_32 from quspin.basis.user module."
                    )
            else:
                if next_state._sig != next_state_sig_64:
                    raise ValueError(
                        "next_state does not have the correct signature, \
					try using next_state_sig_64 from quspin.basis.user module."
                    )

        if count_particles is not None:
            if not isinstance(count_particles, CFunc):
                raise ValueError("count_particles must be a numba.CFunc object.")

            if use_32bit:
                if count_particles._sig != count_particles_sig_32:
                    raise ValueError(
                        "count_particles does not have the correct signature, \
					try using count_particles_sig_64 from quspin.basis.user module."
                    )
            else:
                if count_particles._sig != count_particles_sig_64:
                    raise ValueError(
                        "count_particles does not have the correct signature, \
					try using count_particles_sig_64 from quspin.basis.user module."
                    )

            if count_particles_args is not None:
                if not isinstance(count_particles_args, _np.ndarray):
                    raise ValueError(
                        "count_particles_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
                if not count_particles_args.flags["CARRAY"]:
                    raise ValueError(
                        "count_particles_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            basis_dtype
                        )
                    )
                if not _np.issubdtype(count_particles_args.dtype, _np.integer):
                    raise ValueError(
                        "count_particles_args must be a C-contiguous numpy \
						array with dtype {}".format(
                            _np.integer
                        )
                    )

        self._blocks, map_funcs, pers, qs, map_args = _process_user_blocks(
            use_32bit, blocks, block_order
        )

        self.map_funcs = map_funcs
        self._pers = _np.array(pers, dtype=int)
        self._qs = _np.array(qs, dtype=int)
        self.map_args = map_args

        if Ns_block_est is None:
            if len(self._pers) > 0:
                Ns = int(float(Ns) / _np.multiply.reduce(self._pers)) * 2
        else:
            if type(Ns_block_est) is not int:
                raise TypeError("Ns_block_est must be integer value.")

            Ns = Ns_block_est

        self._basis_dtype = basis_dtype
        self._core_args = (
            Ns_full,
            basis_dtype,
            N,
            sps,
            map_funcs,
            pers,
            qs,
            map_args,
            n_sectors,
            get_Ns_pcon,
            get_s0_pcon,
            next_state,
            next_state_args,
            pre_check_state,
            check_state_nosymm_args,
            parallel,
            count_particles,
            count_particles_args,
            op_func,
            op_args,
        )
        self._core = user_core_wrap(*self._core_args)
        self._N = N
        self._Ns = Ns
        self._Ns_block_est = self._Ns
        self._Np = Np
        self._sps = sps
        self._allowed_ops = set(allowed_ops)

        nmax = _np.prod(self._pers)
        self._n_dtype = _np.min_scalar_type(nmax)

        if _make_basis:
            self.make()
        else:
            self._Ns = 1
            self._basis = basis_zeros(self._Ns, dtype=self._basis_dtype)
            self._n = _np.zeros(self._Ns, dtype=self._n_dtype)

        # if not _is_sorted_decending(self._basis):
        # 	ind = _np.argsort(self._basis,kind="heapsort")[::-1]
        # 	self._basis = _np.asarray(self._basis[ind],order="C")
        # 	self._n =  _np.asarray(self._n[ind],order="C")
        # self.make_basis_blocks()

        if allowed_ops is None:
            allowed_ops = set([chr(i) for i in range(256)])  # all characters allowed.

    def __type__(self):
        return "<type 'qspin.basis.user.user_basis'>"

    def __repr__(self):
        return "< instance of 'qspin.basis.user.user_basis' with {0} states >".format(
            self._Ns
        )

    def __name__(self):
        return "<type 'qspin.basis.user.user_basis'>"

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._core = user_core_wrap(*self._core_args)
