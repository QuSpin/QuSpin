from ._basis_1d_core import hcp_basis,hcp_ops,spf_basis,spf_ops
from .base_1d import basis_1d
from ..base import MAXPRINT
import numpy as _np



class spinless_fermion_basis_1d(basis_1d):
	"""Constructs basis for spinless fermionic operators in a specified 1-d symmetry sector.

	The supported operator strings for `spinless_fermion_basis_1d` are:

	.. math::
			\\begin{array}{cccc}
				\\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}    \\newline	
				\\texttt{spinless_fermion_basis_1d}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}      \\newline
			\\end{array}

	Examples
	--------

	The code snippet below shows how to use the `spinless_fermion_basis_1d` class to construct the basis in the zero momentum sector of positive parity for the fermion Hamiltonian 

	.. math::
		H(t)=-J\\sum_j c^\\dagger_{j+1}c_j + \\mathrm{h.c.} - \\mu\\sum_j n_j + U\\cos\\Omega t\\sum_j n_{j+1} n_j

	.. literalinclude:: ../../doc_examples/spinless_fermion_basis_1d-example.py
		:linenos:
		:language: python
		:lines: 7-

	"""	
	def __init__(self,L,Nf=None,nf=None,**blocks):
		"""Intializes the `fermion_basis_1d` object (basis for fermionic operators).

		Parameters
		-----------
		L: int
			Length of chain/number of sites.
		Nf: {int,list}, optional
			Number of fermions in chain. Can be integer or list to specify one or more particle sectors.
		nf: float, optional
			Density of fermions in chain (fermions per site).
		**blocks: optional
			extra keyword arguments which include:

				**a** (*int*) - specifies unit cell size for translation.

				**kblock** (*int*) - specifies momentum block.

				**pblock** (*int*) - specifies parity block.

		"""

		input_keys = set(blocks.keys())

		# Why can we NOT have a check_z_symm toggler just for one of those weirdass cases that worked?

		expected_keys = set(["_Np","kblock","pblock","a","L"])
		wrong_keys = input_keys - expected_keys 
		if wrong_keys:
			temp = ", ".join(["{}" for key in wrong_keys])
			raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))


		if blocks.get("a") is None: # by default a = 1
			blocks["a"] = 1

		if Nf is not None and nf is not None:
			raise ValueError("Cannot Nf and nf simultaineously.")
		elif Nf is None and nf is not None:
			if nf < 0 or nf > 1:
				raise ValueError("nf must be between 0 and 1")
			Nf = int(nf*L)

		if Nf is None:
			Nf_list = None
		elif type(Nf) is int:
			Nf_list = [Nf]
		else:
			try:
				Nf_list = list(Nf)
			except TypeError:
				raise TypeError("Nf must be iterable returning integers")

			if any((type(Nf) is not int) for Nf in Nf_list):
				TypeError("Nf must be iterable returning integers")

		count_particles = False
		if blocks.get("_Np") is not None:
			_Np = blocks.get("_Np")
			if Nf_list is not None:
				raise ValueError("do not use _Np and Nup/nb simultaineously.")
			blocks.pop("_Np")
			
			if _Np == -1:
				Nf_list = None
			else:
				count_particles = True
				_Np = min(L,_Np)
				Nf_list = list(range(_Np))

		if Nf_list is None:
			self._Np = None			
		else:
			self._Np = sum(Nf_list)

		self._blocks = blocks			

		self._sps = 2
		Imax = (1<<L)-1
		stag_A = sum(1<<i for i in range(0,L,2))
		stag_B = sum(1<<i for i in range(1,L,2))
		pars = _np.array([1,L,Imax,stag_A,stag_B]) # sign to be calculated
		self._operators = ("availible operators for ferion_basis_1d:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tn: number operator"+
							"\n\tz: c-symm number operator")

		self._allowed_ops = set(["I","+","-","n","z"])
		basis_1d.__init__(self,hcp_basis,hcp_ops,L,Np=Nf_list,pars=pars,count_particles=count_particles,**blocks)
		# self._check_symm=None


	def __type__(self):
		return "<type 'qspin.basis.fermion_basis_1d'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.fermion_basis_1d' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.fermion_basis_1d'>"



	# functions called in base class:

	def _sort_opstr(self,op):
		return _sort_opstr_spinless(op)

	def _hc_opstr(self,op):
		return _hc_opstr_spinless(op)
	
	def _non_zero(self,op):
		return _non_zero_spinless(op)

	def _expand_opstr(self,op,num):
		return _expand_opstr_spinless(op,num)	


class spinful_fermion_basis_1d(spinless_fermion_basis_1d,basis_1d):
	"""Constructs basis for spinful fermionic operators in a specified 1-d symmetry sector.

	The supported operator strings for `spinful_fermion_basis_1d` are:

	.. math::
			\\begin{array}{cccc}
				\\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}    \\newline	
				\\texttt{spinful_fermion_basis_1d}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}      \\newline
			\\end{array}

	
	Notes
	-----

	The `spinful_fermion_basis` operator strings are separated by a pipe symbol, '|', to distinguish the spin-up from 
	spin-down species. However, the index array has NO pipe symbol.

	Examples
	--------

	The code snippet below shows how to use the `spinful_fermion_basis_1d` class to construct the basis in the zero momentum sector of positive fermion spin for the Fermi-Hubbard Hamiltonian 

	.. math::
		H=-J\\sum_{j,\\sigma} c^\\dagger_{j+1\\sigma}c_{j\\sigma} + \\mathrm{h.c.} - \\mu\\sum_{j,\\sigma} n_{j\\sigma} + U \\sum_j n_{j\\uparrow} n_{j\\downarrow}

	Notice that the operator strings for constructing Hamiltonians with a `spinful_fermion_basis` object are separated by 
	a pipe symbol, '|', while the index array has no splitting pipe character.
		

	.. literalinclude:: ../../doc_examples/spinful_fermion_basis_1d-example.py
		:linenos:
		:language: python
		:lines: 7-

	"""	
	def __init__(self,L,Nf=None,nf=None,**blocks):
		"""Intializes the `fermion_basis_1d` object (basis for fermionic operators).

		Parameters
		-----------
		L: int
			Length of chain/number of sites.
		Nf: {tupe(int,list)}, optional
			Number of fermions in chain. First (left) entry refers to spin-up and second (right) entry refers
			to spin-down. Each of the two entries can be integer or list to specify one or more particle sectors.
		nf: tuple(float), optional
			Density of fermions in chain (fermions per site). First (left) entry refers to spin-up. Second (right)
			entry refers to spin-down.
		**blocks: optional
			extra keyword arguments which include:

				**a** (*int*) - specifies unit cell size for translation.

				**kblock** (*int*) - specifies momentum block.

				**pblock** (*int*) - specifies parity block.

				**sblock** (*int*) - specifies fermion spin inversion block.

				**psblock** (*int*) - specifies parity followed by fermion spin inversion symmetry block.

		"""

		input_keys = set(blocks.keys())

		# Why can we NOT have fermion spin (sblock) symm on sublat A and sulat B separately?

		expected_keys = set(["_Np","kblock","pblock","sblock","psblock","a","check_z_symm","L"])
		wrong_keys = input_keys - expected_keys 
		if wrong_keys:
			temp = ", ".join(["{}" for key in wrong_keys])
			raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))


		if blocks.get("a") is None: # by default a = 1
			blocks["a"] = 1

		if Nf is not None and nf is not None:
			raise ValueError("cannot use 'nf' and 'Nf' simultaineously.")
		if Nf is None and nf is not None:
			Nf = [(int(nf[0]*L),int(nf[1]*L))]

		self._sps = 2

		count_particles = False
		_Np = blocks.get("_Np")
		if _Np  is not None and Nf is None:
			count_particles = True
			if type(_Np) is not int:
				raise ValueError("_Np must be integer")
			if _Np >= -1:
				if _Np+1 > L: 
					Nf =  []
					for n in range(L+1):
						Nf.extend((n-i,i)for i in range(n+1))

					Nf = tuple(Nf)
				elif _Np==-1:
					Nf = None
				else:
					Nf=[]
					for n in range(_Np+1):
						Nf.extend((n-i,i)for i in range(n+1))

					Nf = tuple(Nf)
			else:
				raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")


		if Nf is None:
			Nf_list = None
			self._Np = None	
		else:
			if type(Nf) is tuple:
				if len(Nf)==2:
					Nup,Ndown = Nf
					self._Np = Nup+Ndown
					if (type(Nup) is not int) and (type(Ndown) is not int):
						raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")
					Nf_list = [Nf]
				else:
					Nf_list = list(Nf)
					N_up_list,N_down_list = zip(*Nf_list)
					self._Np = sum(N_up_list)
					self._Np += sum(N_down_list)
					if any((type(tup)is not tuple) and len(tup)!=2 for tup in Nf_list):
						raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")		

					if any((type(Nup) is not int) and (type(Ndown) is not int) for Nup,Ndown in Nf_list):
						raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")		

					if any(Nup > L or Nup < 0 or Ndown > L or Ndown < 0 for Nup,Ndown in Nf_list):
						raise ValueError("particle numbers in Nf must satisfy: 0 <= n <= L")

			else:
				try:
					Nf_iter = iter(Nf)
				except TypeError:
					raise ValueError("Nf must be tuple of integers or iterable object of tuples.")


				Nf_list = list(Nf)
				N_up_list,_N_down_list = zip(*Nf_list)
				self._Np = sum(N_up_list)
				self._Np += sum(N_down_list)

				if any((type(tup)is not tuple) and len(tup)!=2 for tup in Nf_list):
					raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")		

				if any((type(Nup) is not int) and (type(Ndown) is not int) for Nup,Ndown in Nf_list):
					raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")		

				if any(Nup > L or Nup < 0 or Ndown > L or Ndown < 0 for Nup,Ndown in Nf_list):
					raise ValueError("particle numbers in Nf must satisfy: 0 <= n <= L")

		if blocks.get("check_z_symm") is None or blocks.get("check_z_symm") is True:
			check_z_symm = True
		else:
			check_z_symm = False

		self._blocks = blocks	
		pblock = blocks.get("pblock")
		zblock = blocks.get("sblock")
		kblock = blocks.get("kblock")
		pzblock = blocks.get("psblock")
		a = blocks.get("a")

		if (type(pblock) is int) and (type(zblock) is int):
			blocks["pzblock"] = pblock*zblock
			self._blocks["pzblock"] = pblock*zblock
			pzblock = pblock*zblock		


		### Why do we have the check below for fermion spin symmetry?


		if check_z_symm:
			# checking if fermion spin inversion is compatible with Np and L
			if (Nf_list is not None) and ((type(zblock) is int) or (type(pzblock) is int)):
				if len(Nf_list) > 1:
					ValueError("fermion spin inversion symmetry only reduces the half-filled particle sector")

				Nup,Ndown = Nf_list[0]

				if (L*(self.sps-1) % 2) != 0:
					raise ValueError("fermion spin inversion symmetry with particle conservation must be used with chains at half filling")
				if Nup != L*(self.sps-1)//2 or Ndown != L*(self.sps-1)//2:
					raise ValueError("fermion spin inversion symmetry only reduces the half-filled particle sector")


		Imax = (1<<L)-1
		pars = _np.array([L,Imax,0,0]) # sign to be calculated
		self._operators = ("availible operators for ferion_basis_1d:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tn: number operator"+
							"\n\tz: c-symm number operator")

		self._allowed_ops = set(["I","+","-","n","z"])
		basis_1d.__init__(self,spf_basis,spf_ops,L,Np=Nf_list,pars=pars,count_particles=count_particles,**blocks)
		

	def _Op(self,opstr,indx,J,dtype):
		
		i = opstr.index("|")
		indx = _np.array(indx,dtype=_np.int32)
		indx[i:] += self.L
		opstr=opstr.replace("|","")

		return basis_1d._Op(self,opstr,indx,J,dtype)
		


	def __type__(self):
		return "<type 'qspin.basis.fermion_basis_1d'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.fermion_basis_1d' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.fermion_basis_1d'>"

	@property
	def N(self):
		return 2*self._L


	# functions called in base class:

	def _sort_opstr(self,op):
		return _sort_opstr_spinful(op)

	def _hc_opstr(self,op):
		return _hc_opstr_spinful(op)
	
	def _non_zero(self,op):
		return _non_zero_spinful(op)

	def _expand_opstr(self,op,num):
		return _expand_opstr_spinful(op,num) 

	def _get__str__(self):
		def get_state(b):
			bits_left = ((b>>(self.N-i-1))&1 for i in range(self.N//2))
			state_left = "|"+(" ".join(("{:1d}").format(bit) for bit in bits_left))+">"
			bits_right = ((b>>(self.N//2-i-1))&1 for i in range(self.N//2))
			state_right = "|"+(" ".join(("{:1d}").format(bit) for bit in bits_right))+">"
			return state_left+state_right


		temp1 = "     {0:"+str(len(str(self.Ns)))+"d}.  "
		if self._Ns > MAXPRINT:
			half = MAXPRINT // 2
			str_list = [(temp1.format(i))+get_state(b) for i,b in zip(range(half),self._basis[:half])]
			str_list.extend([(temp1.format(i))+get_state(b) for i,b in zip(range(self._Ns-half,self._Ns,1),self._basis[-half:])])
		else:
			str_list = [(temp1.format(i))+get_state(b) for i,b in enumerate(self._basis)]

		return tuple(str_list)




def _sort_opstr_spinless(op):
	if op[0].count("|") > 0:
		raise ValueError("'|' character found in op: {0},{1}".format(op[0],op[1]))
	if len(op[0]) != len(op[1]):
		raise ValueError("number of operators in opstr: {0} not equal to length of indx {1}".format(op[0],op[1]))

	op = list(op)
	zipstr = list(zip(op[0],op[1]))
	if zipstr:
		n = len(zipstr)
		swapped = True
		anticommutes = 0
		while swapped:
			swapped = False
			for i in range(n-1):
				if zipstr[i][1] > zipstr[i+1][1]:
					temp = zipstr[i]
					zipstr[i] = zipstr[i+1]
					zipstr[i+1] = temp
					swapped = True

					if zipstr[i][0] in ["+","-"] and zipstr[i+1][0] in ["+","-"]:
						anticommutes += 1

		op1,op2 = zip(*zipstr)
		op[0] = "".join(op1)
		op[1] = tuple(op2)
		op[2] *= (1 if anticommutes%2 == 0 else -1)
	return tuple(op)

def _sort_opstr_spinful(op):
	op = list(op)
	opstr = op[0]
	indx  = op[1]

	if opstr.count("|") == 0: 
		raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr,indx))

	# if opstr.count("|") > 1: 
	# 	raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

	if len(opstr)-opstr.count("|") != len(indx):
		raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

	i = opstr.index("|")
	indx_left = indx[:i]
	indx_right = indx[i:]

	opstr_left,opstr_right=opstr.split("|",1)

	op1 = list(op)
	op1[0] = opstr_left
	op1[1] = tuple(indx_left)

	op2 = list(op)
	op2[0] = opstr_right
	op2[1] = tuple(indx_right)

	op1 = _sort_opstr_spinless(op1)
	op2 = _sort_opstr_spinless(op2)

	op[0] = "|".join((op1[0],op2[0]))
	op[1] = op1[1] + op2[1]
	
	return tuple(op)


def _non_zero_spinless(op):
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

def _non_zero_spinful(op):
		op = list(op)
		opstr = op[0]
		indx  = op[1]
	
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx_left = indx[:i]
		indx_right = indx[i:]

		opstr_left,opstr_right=opstr.split("|",1)

		op1 = list(op)
		op1[0] = opstr_left
		op1[1] = indx_left

		op2 = list(op)
		op2[0] = opstr_right
		op2[1] = indx_right

		return (_non_zero_spinless(op1) and _non_zero_spinless(op2))


def _hc_opstr_spinless(op):
	op = list(op)
	# take h.c. + <--> - , reverse operator order , and conjugate coupling
	op[0] = list(op[0].replace("+","%").replace("-","+").replace("%","-"))
	op[0].reverse()
	op[0] = "".join(op[0])
	op[1] = list(op[1])
	op[1].reverse()
	op[1] = tuple(op[1])
	op[2] = op[2].conjugate()
	return _sort_opstr_spinless(op) # return the sorted op.

def _hc_opstr_spinful(op):
		op = list(op)
		opstr = op[0]
		indx  = op[1]
	
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx_left = indx[:i]
		indx_right = indx[i:]

		opstr_left,opstr_right=opstr.split("|",1)

		op1 = list(op)
		op1[0] = opstr_left
		op1[1] = indx_left
		op1[2] = op[2]

		op2 = list(op)
		op2[0] = opstr_right
		op2[1] = indx_right
		op2[2] = complex(1.0)
		
		op1 = _hc_opstr_spinless(op1)
		op2 = _hc_opstr_spinless(op2)

		op[0] = "|".join((op1[0],op2[0]))
		op[1] = op1[1] + op2[1]

		op[2] = op1[2]*op2[2]

		return tuple(op)


def _expand_opstr_spinless(op,num):
	op = list(op)
	op.append(num)
	return [tuple(op)]

def _expand_opstr_spinful(op,num):
		op = list(op)
		opstr = op[0]
		indx  = op[1]
	
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx_left = indx[:i]
		indx_right = indx[i:]

		opstr_left,opstr_right=opstr.split("|",1)

		op1 = list(op)
		op1[0] = opstr_left
		op1[1] = indx_left
		op1[2] = 1.0

		op2 = list(op)
		op2[0] = opstr_right
		op2[1] = indx_right

		op1_list = _expand_opstr_spinless(op1,num)
		op2_list = _expand_opstr_spinless(op2,num)

		op_list = []
		for new_op1 in op1_list:
			for new_op2 in op2_list:
				new_op = list(new_op1)
				new_op[0] = "|".join((new_op1[0],new_op2[0]))
				new_op[1] += tuple(new_op2[1])
				new_op[2] *= new_op2[2]


				op_list.append(tuple(new_op))

		return tuple(op_list)
