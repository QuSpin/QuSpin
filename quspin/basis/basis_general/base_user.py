from .base_general import basis_general
from ._basis_general_core import user_core_wrap
import numpy as _np
from numba import cfunc, types
from numba.ccallback import CFunc

map_sig_32 = types.uint32(types.uint32,types.intc,types.CPointer(types.intc))
map_sig_64 = types.uint64(types.uint64,types.intc,types.CPointer(types.intc))

next_state_sig_32 = types.uint32(types.uint32,types.uint32,types.CPointer(types.uint32))
next_state_sig_64 = types.uint64(types.uint64,types.uint64,types.CPointer(types.uint64))

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

def _process_user_blocks(use_32bit,blocks_dict,block_order):

	if any((type(v) is not tuple) and (len(v)!=3) for v in blocks_dict.values()):
		raise ValueError

	if not all(isinstance(f,CFunc) for f,_,_ in blocks_dict.values()):
		raise ValueError

	if use_32bit:
		if not all(f._sig==map_sig_32 for f,_,_ in blocks_dict.values()):
			raise ValueError
	else:
		if not all(f._sig==map_sig_64 for f,_,_ in blocks_dict.values()):
			raise ValueError

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
	"""Constructs basis for spin operators for USER-DEFINED symmetries.

	Any unitary symmetry transformation :math:`Q` of multiplicity :math:`m_Q` (:math:`Q^{m_Q}=1`) has
	eigenvalues :math:`\\exp(-2\\pi i q/m_Q)`, labelled by an ingeter :math:`q\\in\\{0,1,\\dots,m_Q-1\\}`.
	These integers :math:`q` are used to define the symmetry blocks.

	For instance, if :math:`Q=P` is parity (reflection), then :math:`q=0,1`. If :math:`Q=T` is translation by one lattice site,
	then :math:`q` labels the mometum blocks in the same fashion as for the `..._basis_1d` classes. 

	User-defined symmetries with the `spin_basis_general` class can be programmed as follows. Suppose we have a system of
	L sites, enumerated :math:`s=(s_0,s_1,\\dots,s_{L-1})`. There are two types of operations one can perform on the sites:
		* exchange the labels of two sites: :math:`s_i \\leftrightarrow s_j` (e.g., translation, parity)
		* invert the population on a given site: :math:`s_i\\leftrightarrow -(s_j+1)` (e.g., spin inversion)

	These two operations already comprise a variety of symmetries, including translation, parity (reflection) and 
	spin inversion. For a specific example, see below.

	The supported operator strings for `spin_basis_general` are:

	.. math::
		\\begin{array}{cccc}
			\\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &	 \\texttt{"z"}   &   \\texttt{"x"}   &   \\texttt{"y"}  \\newline	
			\\texttt{spin_basis_general} &   \\hat{1}		&   \\hat\\sigma^+	   &   \\hat\\sigma^-	  &	 \\hat\\sigma^z	   &   (\\hat\\sigma^x)	 &   (\\hat\\sigma^y)  \\  \\newline
		\\end{array}

	**Notes:** 
		* The relation between spin and Pauli matrices is :math:`\\vec S = \\vec \\sigma/2`.
		* The default operators for spin-1/2 are the Pauli matrices, NOT the spin operators. To change this, see the argument `pauli` of the `spin_basis` class. Higher spins can only be defined using the spin operators, and do NOT support the operator strings "x" and "y". 
		* QuSpin raises a warning to alert the reader when non-commuting symmetries are passed. In such cases, we recommend the user to manually check the combined usage of symmetries by, e.g., comparing the eigenvalues.

 		
	Examples
	--------

	The code snippet below shows how to construct the two-dimensional transverse-field Ising model.
	
	.. math::
		H = J \\sum_{\\langle ij\\rangle} \\sigma^z_{i}\\sigma^z_j+ g\\sum_j\\sigma^x_j 

	Moreover, it demonstrates how to pass user-defined symmetries to the `spin_basis_general` constructor. In particular,
	we do translation invariance and parity (reflection) (along each lattice direction), and spin inversion. Note that parity 
	(reflection) and translation invariance are non-commuting symmetries, and QuSpin raises a warning when constructing the basis. 
	However, they do commute in the zero-momentum (also in the pi-momentum) symmetry sector; hence, one can ignore the warning and
	use the two symemtries together to reduce the Hilbert space dimension.


	.. literalinclude:: ../../doc_examples/spin_basis_general-example.py
		:linenos:
		:language: python
		:lines: 7-

	"""
	def __init__(self,basis_dtype,N,Ns_full,op_func,allowed_ops=None,sps=None,pcon_args=None,
		Ns_block_est=None,_make_basis=True,block_order=None,_Np=None,sort_basis=False,**blocks):
		"""Intializes the `user_basis_general` object (basis for user defined ED calculations).

		Parameters
		-----------
		N: int
			number of sites.
		
		"""
		if _Np is not None:
			raise ValueError("cannot use photon basis with user_basis_general.")

		self._check_pcon = None
		self._count_particles = False

		self._N = N
		if basis_dtype not in [_np.uint32,_np.uint64]:
			raise ValueError("basis_dtype must be either uint32 or uint64 for the given basis representation.")

		use_32bit = (basis_dtype == _np.uint32)

		if not isinstance(op_func,CFunc):
			raise ValueError

		if use_32bit:
			if op_func._sig != op_sig_32:
				raise ValueError
		else:
			if op_func._sig != op_sig_64:
				raise ValueError


		if pcon_args is not None:

			if type(pcon_args) is dict:
				next_state_args = pcon_args.get("next_state_args")
				if next_state_args is not None:
					if not isinstance(next_state_args,_np.ndarray):
						raise ValueError("next_state_args must be a C-contiguous numpy\
							array with dtype {}".format(basis_dtype))
					if not next_state_args.flags["CARRAY"]:
						raise ValueError("next_state_args must be a C-contiguous numpy\
							array with dtype {}".format(basis_dtype))
					if next_state_args.dtype != basis_dtype:
						raise ValueError("next_state_args must be a C-contiguous numpy\
							array with dtype {}".format(basis_dtype))

					pcon_args.pop("next_state_args")

				if len(pcon_args) == 4:
					Np = pcon_args["Np"]
					next_state = pcon_args["next_state"]
					get_Ns_pcon = pcon_args["get_Ns_pcon"]
					get_s0_pcon = pcon_args["get_s0_pcon"]
					n_sectors = 0
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
				Ns = get_Ns_pcon(N,Np)
			else:
				try:
					Np_iter = iter(Np)
				except TypeError:
					raise TypeError("Np must be integer or iteratable object.")

				Np = list(Np)
				Ns = sum(get_Ns_pcon(N,np) for np in Np)


		else:
			Ns = Ns_full
			Np = None
			next_state_args = None
			next_state = None
			count_particles = None
			get_s0_pcon = None
			get_Ns_pcon = None
			n_sectors = 0

		if next_state is not None:
			if not isinstance(next_state,CFunc):
				raise ValueError

			if use_32bit:
				if next_state._sig != next_state_sig_32:
					raise ValueError
			else:
				if next_state._sig != next_state_sig_64:
					raise ValueError

		if count_particles is not None:
			if not isinstance(count_particles,CFunc):
				raise ValueError

			if use_32bit:
				if count_particles._sig != count_particles_sig_32:
					raise ValueError
			else:
				if count_particles._sig != count_particles_sig_64:
					raise ValueError

		self._blocks,map_funcs,pers,qs = _process_user_blocks(use_32bit,blocks,block_order)

		self.map_funcs = map_funcs
		self._pers = _np.array(pers,dtype=_np.int)
		self._qs = _np.array(qs,dtype=_np.int)

		if len(self._pers)>0:
			if Ns_block_est is None:
				Ns = int(float(Ns)/_np.multiply.reduce(self._pers))*2
			else:
				if type(Ns_block_est) is not int:
					raise TypeError("Ns_block_est must be integer value.")
					
				Ns = Ns_block_est


		self._basis_dtype = basis_dtype
		self._core = user_core_wrap(Ns_full, basis_dtype, N, map_funcs, pers, qs,
								n_sectors, get_Ns_pcon, get_s0_pcon, next_state,
								next_state_args,count_particles, op_func, sps)

		self._N = N
		self._Ns = Ns
		self._Np = Np

		nmax = _np.prod(self._pers)
		self._n_dtype = _np.min_scalar_type(nmax)

		if _make_basis:	
			self.make(sort_basis=sort_basis)
		else:
			self._Ns=1
			self._basis=basis_zeros(self._Ns,dtype=self._basis_dtype)
			self._n=_np.zeros(self._Ns,dtype=self._n_dtype)

		if allowed_ops is None:
			allowed_ops = set([chr(i) for i in range(256)]) # all characters allowed.

		self._sps=sps
		self._allowed_ops=set(allowed_ops)




	def __type__(self):
		return "<type 'qspin.basis.general_hcb'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.general_hcb' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.general_hcb'>"

	# functions called in base class:

	def _sort_opstr(self,op):
		if op[0].count("|") > 0:
			raise ValueError("'|' character found in op: {0},{1}".format(op[0],op[1]))
		if len(op[0]) != len(op[1]):
			raise ValueError("number of operators in opstr: {0} not equal to length of indx {1}".format(op[0],op[1]))

		op = list(op)
		zipstr = list(zip(op[0],op[1]))
		if zipstr:
			zipstr.sort(key = lambda x:x[1])
			op1,op2 = zip(*zipstr)
			op[0] = "".join(op1)
			op[1] = tuple(op2)
		return tuple(op)

	def _non_zero(self,op):
		opstr = _np.array(list(op[0]))
		indx = _np.array(op[1])
		if _np.any(indx):
			indx_p = indx[opstr == "+"].tolist()
			p = not any(indx_p.count(x) > 1 for x in indx_p)
			indx_p = indx[opstr == "-"].tolist()
			m = not any(indx_p.count(x) > 1 for x in indx_p)
			return (p and m)
		else:
			return True
		
	def _hc_opstr(self,op):
		op = list(op)
		# take h.c. + <--> - , reverse operator order , and conjugate coupling
		op[0] = list(op[0].replace("+","%").replace("-","+").replace("%","-"))
		op[0].reverse()
		op[0] = "".join(op[0])
		op[1] = list(op[1])
		op[1].reverse()
		op[1] = tuple(op[1])
		op[2] = op[2].conjugate()
		return self._sort_opstr(op) # return the sorted op.

	def _expand_opstr(self,op,num):
		opstr = str(op[0])
		indx = list(op[1])
		J = op[2]
 
		if len(opstr) <= 1:
			if opstr == "x":
				op1 = list(op)
				op1[0] = op1[0].replace("x","+")
				if self._pauli in [0,1]:
					op1[2] *= 0.5
				op1.append(num)

				op2 = list(op)
				op2[0] = op2[0].replace("x","-")
				if self._pauli in [0,1]:
					op2[2] *= 0.5
				op2.append(num)

				return (tuple(op1),tuple(op2))
			elif opstr == "y":
				op1 = list(op)
				op1[0] = op1[0].replace("y","+")
				if self._pauli in [0,1]:
					op1[2] *= -0.5j
				else:
					op1[2] *= -1j
				op1.append(num)

				op2 = list(op)
				op2[0] = op2[0].replace("y","-")
				if self._pauli in [0,1]:
					op2[2] *= 0.5j
				else:
					op2[2] *= 1j
				op2.append(num)

				return (tuple(op1),tuple(op2))
			else:
				op = list(op)
				op.append(num)
				return [tuple(op)]	
		else:
	 
			i = len(opstr)//2
			op1 = list(op)
			op1[0] = opstr[:i]
			op1[1] = tuple(indx[:i])
			op1[2] = complex(J)
			op1 = tuple(op1)

			op2 = list(op)
			op2[0] = opstr[i:]
			op2[1] = tuple(indx[i:])
			op2[2] = complex(1)
			op2 = tuple(op2)

			l1 = self._expand_opstr(op1,num)
			l2 = self._expand_opstr(op2,num)

			l = []
			for op1 in l1:
				for op2 in l2:
					op = list(op1)
					op[0] += op2[0]
					op[1] += op2[1]
					op[2] *= op2[2]
					l.append(tuple(op))

			return tuple(l)

	def Op_bra_ket(self,opstr,indx,J,dtype,ket_states,reduce_output=True):

		if self._S == "1/2":
			ME,bra,ket = hcb_basis_general.Op_bra_ket(self,opstr,indx,J,dtype,ket_states,reduce_output=reduce_output)
			if self._pauli==1:
				n = len(opstr.replace("I",""))
				ME *= (1<<n)
			elif self._pauli==-1:
				n = len(opstr.replace("I","").replace("+","").replace("-",""))
				ME *= (1<<n)
		else:
			return higher_spin_basis_general.Op_bra_ket(self,opstr,indx,J,dtype,ket_states)

		return ME,bra,ket
