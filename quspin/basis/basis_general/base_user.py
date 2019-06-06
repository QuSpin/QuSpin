from .base_general import basis_general
from ._basis_general_core import user_core_wrap
import numpy as _np
from numba import cfunc, types, njit
from numba.ccallback import CFunc

map_sig_32 = types.uint32(types.uint32,types.intc,types.CPointer(types.intc))
map_sig_64 = types.uint64(types.uint64,types.intc,types.CPointer(types.intc))

next_state_sig_32 = types.uint32(types.uint32,types.uint32,
	types.uint32,types.CPointer(types.uint32))
next_state_sig_64 = types.uint64(types.uint64,types.uint64,
	types.uint64,types.CPointer(types.uint64))

check_state_nosymm_sig_32 = types.uint32(types.uint32,
	types.uint32,types.CPointer(types.uint32))
check_state_nosymm_sig_64 = types.uint64(types.uint64,
	types.uint64,types.CPointer(types.uint64))

op_results_32 = types.Record.make_c_struct([
   ('matrix_ele', types.complex128),('state', types.uint32),
])

op_results_64 = types.Record.make_c_struct([
   ('matrix_ele', types.complex128),('state', types.uint64)
])

op_sig_32 = types.intc(types.CPointer(op_results_32),
								types.char,
								types.intc,
								types.intc)
op_sig_64 = types.intc(types.CPointer(op_results_64),
								types.char,
								types.intc,
								types.intc)

count_particles_sig_32 = types.void(types.uint32,
							types.CPointer(types.intc))
count_particles_sig_64 = types.void(types.uint64,
							types.CPointer(types.intc))

__all__ = ["map_sig_32","map_sig_64","next_state_sig_32",
	"next_state_sig_64","op_func_sig_32","op_func_sig_64","user_basis"]

@njit
def is_sorted_decending(a):
	for i in range(a.size-1):
		if(a[i]<a[i+1]):
			return False

	return True

def _process_user_blocks(use_32bit,blocks_dict,block_order):

	if any((type(v) is not tuple) and (len(v)!=3) for v in blocks_dict.values()):
		raise ValueError

	if not all(isinstance(f,CFunc) for f,_,_ in blocks_dict.values()):
		raise ValueError("map_func must be instance of numba.CFunc.")

	if use_32bit:
		if not all(f._sig==map_sig_32 for f,_,_ in blocks_dict.values()):
			raise ValueError("map_func does not have the correct signature, \
					try using map_sig_32 from quspin.basis.user module.")
	else:
		if not all(f._sig==map_sig_64 for f,_,_ in blocks_dict.values()):
			raise ValueError("map_func does not have the correct signature, \
					try using map_sig_64 from quspin.basis.user module.")

	if block_order is None: # sort by periodicies largest to smallest
		sorted_items = sorted(blocks_dict.items(),key=lambda x:x[1][1])
		sorted_items.reverse()
	else:
		block_order = list(block_order)
		missing = set(blocks_dict.keys()) - set(block_order)
		if len(missing)>0:
			raise ValueError("{} names found in block names but missing from block_order.".format(missing))

		missing = set(block_order) - set(blocks_dict.keys())
		if len(missing)>0:
			raise ValueError("{} names found in block_order but missing from block names.".format(missing))

		sorted_items = [(key,blocks_dict[key]) for key in block_order]

	if len(sorted_items)>0:

		blocks = {block:((-1)**q if per==2 else q) for block,(_,per,q) in sorted_items}


		_,items = zip(*sorted_items)
		map_funcs,pers,qs = zip(*items)

		return blocks,map_funcs,pers,qs

	else:
		return {},[],[],[]



class user_basis(basis_general):
	"""Constructs basis for USER-DEFINED functionality for a basis.

	Any unitary symmetry transformation :math:`Q` of multiplicity :math:`m_Q` (:math:`Q^{m_Q}=1`) has
	eigenvalues :math:`\\exp(-2\\pi i q/m_Q)`, labelled by an ingeter :math:`q\\in\\{0,1,\\dots,m_Q-1\\}`.
	These integers :math:`q` are used to define the symmetry blocks.

	For instance, if :math:`Q=P` is parity (reflection), then :math:`q=0,1`. If :math:`Q=T` is translation by one lattice site,
	then :math:`q` labels the mometum blocks in the same fashion as for the `..._basis_1d` classes. 

	User-defined symmetries with the `user_basis` class can be programmed as follows. 

	Examples
	--------


	.. literalinclude:: ../../doc_examples/user_basis_general-example.py
		:linenos:
		:language: python
		:lines: 7-

	"""
	def __init__(self,basis_dtype,N,Ns_full,op_func,pcon_args=None,check_state_nosymm=None,allowed_ops=None,sps=None,
		Ns_block_est=None,_make_basis=True,block_order=None,_Np=None,**blocks):
		"""Intializes the `user_basis_general` object (basis for user defined ED calculations).

		Parameters
		-----------
		basis_dtype: numpy.dtype object
			the data type used to represent the states in the basis: must be either uint32 or uint64.
		N: int
			Number of sites.
		Ns_full: int
			The size of the full hilbert space with no symmetries, e.g. 2^N for spin-1/2 system.
		op_func: numba.CFunc object
			This is a compiled function which calculates the matrix elements given a state and a character which
			represent the operator and an integer specifying the site of that local operator. Note that this functionality
			will not support branching, e.g. O|state> = me|new_state> and can't be a linear combination of multiple
			states in the basis, e.g. O|state> = me1|new_state_1> + me2|new_state_2> + ... see the above example for how
			one would use this for spin-1/2 system.
		pcon_args: dict, optional
			This dictionary contains the following items which are required to use particle conservation in this basis:
				minimum requirements:
					* "Np"			: list of tuple/int or a single tuple/int which specify the particle sector(s). 
					* "next_state"  : CFunc which, given an integer, generates the next lexigraphically ordered particle conservation state.
					* "get_Ns_pcon" : python function which being called as get_Ns_pcon(N,Np) returns the size of the 
									  non-symmeterized particle conservation basis.
					* "get_s0_pcon" : python function which being called as get_s0_pcon(N,Np) returns the starting state to generate the whole 
									  particle conservation basis by repeatidly calling next_sate(s).
				advanced requirements to use Op_bra_ket functionality (on top of the minimum):
					* "n_sectors"       : the number of integers which parameterized the particle sectors, e.g. with spinful fermions 
										  there is a particle number for both the up and the down sectors, so this number would be 2. 
					* "count_particles" : CFunc which counts the number of particles in each sector and places them into a pointer passed
										  into the CFunc. The pointer provided will have `n_sector` of memory allocated. the order of the number
										  should be kept the same as the ordering of `Np`.
		check_state_nosymm: CFunc or tuple(CFunc,ndarray(C-contiguous,dtype=basis_dtype)), optional
			This allows the user to specify a boolean function which checks a state before checking if a state is a 
			representative state. This allows the user to do things like,  enforce a local hilbert space constraint, 
			e.g. for spinful fermions never having a doubly occupied site. The ndarray are extra arguments which are 
			passed into this function.
		allowed_ops: list/set, optional
			A list of allowed charactors which can be passed in to the op_func. This will be used in the error handling
			so that python will throw a more detailed error message when the wrong operator string is passed into the basis.
		sps: int, optional
			The number of states per site, this is not required for the main functionality for this basis, however it is required
			for doing entanglement entropy calculations. 
		Ns_block_est: int, optional
			An estimate for the size of the symmetry reduced block, QuSpin does a simple estimate which is not always correct. 
		block_order: tuple/list, optional
			An ordered list which contains the order for which the symmetries will be applied on the state. 
		**blocks: optional
			keyword arguments which pass the symmetry generator arrays. For instance:

			>>> basis(...,kxblock=(QFunc,Tq,q),...)

			The keys of the symmetry sector, e.g. `kxblock`, can be defined arbitrarily by the user. The
			values are tuples where the first entry contains the numba.CFunc which generates the symmetry transformation :math:`Q` 
			acting on the state (see class example), the second entry is an integer :math:`Tq` which gives the periodicity
			of the symmetry sector, and q is the quantum number for the given sector. Note that if the periodicity is wrong
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
		self._made_basis = False # keeps track of whether the basis has been made

		self._N = N
		if basis_dtype not in [_np.uint32,_np.uint64]:
			raise ValueError("basis_dtype must be either uint32 or uint64 for the given basis representation.")

		use_32bit = (basis_dtype == _np.uint32)

		if not isinstance(op_func,CFunc):
			raise ValueError("op_func must be a numba.CFunc object.")

		if use_32bit:
			if op_func._sig != op_sig_32:
				raise ValueError("op_func does not have the correct signature, \
					try using op_sig_32 from quspin.basis.user module.")
		else:
			if op_func._sig != op_sig_64:
				raise ValueError("op_func does not have the correct signature, \
					try using op_sig_64 from quspin.basis.user module.")


		if pcon_args is not None:

			if type(pcon_args) is dict:
				next_state_args = pcon_args.get("next_state_args")
				if next_state_args is not None:
					if not isinstance(next_state_args,_np.ndarray):
						raise ValueError("next_state_args must be a C-contiguous numpy \
							array with dtype {}".format(basis_dtype))
					if not next_state_args.flags["CARRAY"]:
						raise ValueError("next_state_args must be a C-contiguous numpy \
							array with dtype {}".format(basis_dtype))
					if next_state_args.dtype != basis_dtype:
						raise ValueError("next_state_args must be a C-contiguous numpy \
							array with dtype {}".format(basis_dtype))

					pcon_args.pop("next_state_args")

				if len(pcon_args) == 4:
					Np = pcon_args["Np"]
					next_state = pcon_args["next_state"]
					get_Ns_pcon = pcon_args["get_Ns_pcon"]
					get_s0_pcon = pcon_args["get_s0_pcon"]
					n_sectors = None
					count_particles = None

				elif len(pcon_args) == 6:
					Np = pcon_args["Np"]
					next_state = pcon_args["next_state"]
					get_Ns_pcon = pcon_args["get_Ns_pcon"]
					get_s0_pcon = pcon_args["get_s0_pcon"]
					n_sectors = pcon_args["n_sectors"]
					count_particles = pcon_args["count_particles"]
				else:
					raise ValueError("pcon_args input not understood.")
			else:
				raise ValueError("pcon_args input not understood.")


			if Np is None:
				Ns = Ns_full
			elif type(Np) is tuple or type(Np) is int:
				self._get_proj_pcon = True
				if n_sectors is not None:
					if type(Np) is int and n_sectors!=1:
						raise ValueError("n_sectors is {} when the size \
							of the particle sector is 1".format(n_sectors))
					elif type(Np) is tuple and n_sectors!=len(Np):
						raise ValueError("n_sectors is {} when the size \
							of the particle sector is {}".format(n_sectors,len(np)))
					else:
						raise ValueError("Np must be tuple, int, or a list of tuples/integers.")
				else:
					if type(Np) is int:
						n_sectors=1
					elif type(Np) is tuple:
						n_sectors=len(Np)
					else:
						raise ValueError("Np must be tuple, int, or a list of tuples/integers.")


				Ns = get_Ns_pcon(N,Np)
			else:
				try:
					Np_iter = iter(Np)
				except TypeError:
					raise TypeError("Np must be integer or iteratable object.")

				Np = list(Np)
				
				for np in Np:
					if n_sectors is not None:
						if type(np) is int and n_sectors!=1:
							raise ValueError("n_sectors is {} when the size \
								of the particle sector is 1".format(n_sectors))
						elif type(np) is tuple and n_sectors!=len(np):
							raise ValueError("n_sectors is {} when the size \
								of the particle sector is {}".format(n_sectors,len(np)))
						else:
							raise ValueError("Np must be tuple, int, or a list of tuples/integers.")
					else:
						if type(np) is int:
							n_sectors=1
						elif type(np) is tuple:
							n_sectors=len(np)
						else:
							raise ValueError("Np must be tuple, int, or a list of tuples/integers.")

				Ns = sum(get_Ns_pcon(N,np) for np in Np)


		else:
			self._get_proj_pcon = False
			Ns = Ns_full
			Np = None
			next_state_args = None
			next_state = None
			count_particles = None
			get_s0_pcon = None
			get_Ns_pcon = None
			n_sectors = -1

		# check_state function BEFORE symmetry checks
		# this can be used to impose local hilbert space constraints.
		if check_state_nosymm is not None:
			try:
				check_state_nosymm,check_state_nosymm_args = check_state_nosymm
			except TypeError:
				check_state_nosymm_args = None

			if not isinstance(check_state_nosymm,CFunc):
				raise ValueError("check_state_nosymm must be a numba.CFunc object.")

			if use_32bit:
				if check_state_nosymm._sig != check_state_nosymm_sig_32:
					raise ValueError("check_state_nosymm does not have the correct signature, \
					try using check_state_nosymm_sig_32 from quspin.basis.user module.")
			else:
				if check_state_nosymm._sig != check_state_nosymm_sig_64:
					raise ValueError("check_state_nosymm does not have the correct signature, \
					try using check_state_nosymm_sig_64 from quspin.basis.user module.")


			if check_state_nosymm_args is not None:
				if not isinstance(check_state_nosymm_args,_np.ndarray):
					raise ValueError("next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(basis_dtype))
				if not check_state_nosymm_args.flags["CARRAY"]:
					raise ValueError("next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(basis_dtype))
				if check_state_nosymm_args.dtype != basis_dtype:
					raise ValueError("next_state_args must be a C-contiguous numpy \
						array with dtype {}".format(basis_dtype))
		else:
			check_state_nosymm,check_state_nosymm_args = None,None

		if next_state is not None:
			if not isinstance(next_state,CFunc):
				raise ValueError("next_state must be a numba.CFunc object.")

			if use_32bit:
				if next_state._sig != next_state_sig_32:
					raise ValueError("next_state does not have the correct signature, \
					try using next_state_sig_32 from quspin.basis.user module.")
			else:
				if next_state._sig != next_state_sig_64:
					raise ValueError("next_state does not have the correct signature, \
					try using next_state_sig_64 from quspin.basis.user module.")

		if count_particles is not None:
			if not isinstance(count_particles,CFunc):
				raise ValueError("next_state must be a numba.CFunc object.")

			if use_32bit:
				if count_particles._sig != count_particles_sig_32:
					raise ValueError("count_particles does not have the correct signature, \
					try using count_particles_sig_64 from quspin.basis.user module.")
			else:
				if count_particles._sig != count_particles_sig_64:
					raise ValueError("count_particles does not have the correct signature, \
					try using count_particles_sig_64 from quspin.basis.user module.")

		self._blocks,map_funcs,pers,qs = _process_user_blocks(use_32bit,blocks,block_order)

		self.map_funcs = map_funcs
		self._pers = _np.array(pers,dtype=_np.int)
		self._qs = _np.array(qs,dtype=_np.int)


		if Ns_block_est is None:
			if len(self._pers) > 0:
				Ns = int(float(Ns)/_np.multiply.reduce(self._pers))*2
		else:
			if type(Ns_block_est) is not int:
				raise TypeError("Ns_block_est must be integer value.")
				
			Ns = Ns_block_est

		self._basis_dtype = basis_dtype
		self._core = user_core_wrap(Ns_full, basis_dtype, N, map_funcs, pers, qs,
								n_sectors, get_Ns_pcon, get_s0_pcon, next_state,
								next_state_args,check_state_nosymm,check_state_nosymm_args,
								count_particles, op_func, sps)

		self._N = N
		self._Ns = Ns
		self._Np = Np

		nmax = _np.prod(self._pers)
		self._n_dtype = _np.min_scalar_type(nmax)

		if _make_basis:	
			self.make()
		else:
			self._Ns=1
			self._basis=basis_zeros(self._Ns,dtype=self._basis_dtype)
			self._n=_np.zeros(self._Ns,dtype=self._n_dtype)

		if not is_sorted_decending(self._basis):
			ind = _np.argsort(self._basis,kind="heapsort")[::-1]
			self._basis = self._basis[ind]
			self._n = self._n[ind]


		if allowed_ops is None:
			allowed_ops = set([chr(i) for i in range(256)]) # all characters allowed.

		self._sps=sps
		self._allowed_ops=set(allowed_ops)

	def __type__(self):
		return "<type 'qspin.basis.user.user_basis'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.user.user_basis' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.user.user_basis'>"


