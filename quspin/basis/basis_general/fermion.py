from ._basis_general_core import spinful_fermion_basis_core_wrap_32,spinful_fermion_basis_core_wrap_64
from ._basis_general_core import spinless_fermion_basis_core_wrap_32,spinless_fermion_basis_core_wrap_64
from .base_general import basis_general,_check_symm_map
from ..base import MAXPRINT
import numpy as _np
from scipy.misc import comb


# general basis for hardcore bosons/spin-1/2
class spinless_fermion_basis_general(basis_general):
	"""Constructs basis for spinless fermion operators for USER-DEFINED symmetries.

	Any unitary symmetry transformation :math:`Q` of multiplicity :math:`m_Q` (:math:`Q^{m_Q}=1`) has
	eigenvalues :math:`\\exp(-2\\pi i q/m_Q)`, labelled by an ingeter :math:`q\\in\\{0,1,\\dots,m_Q-1\\}`.
	These integers :math:`q` are used to define the symmetry blocks.

	For instance, if :math:`Q=P` is parity (reflection), then :math:`q=0,1`. If :math:`Q=T` is translation by one lattice site,
	then :math:`q` labels the mometum blocks in the same fashion as for the `..._basis_1d` classes. 

	User-defined symmetries with the `spinless_fermion_basis_general` class can be programmed as follows. Suppose we have a system of
	L sites, enumerated :math:`s=(s_0,s_1,\\dots,s_{L-1})`. There are two types of operations one can perform on the sites:
		* exchange the labels of two sites: :math:`s_i \\leftrightarrow s_j` (e.g., translation, parity)
		* invert the **fermion population** on a given site with appropriate sign flip :math:`c_j^\\dagger\\to (-1)^j c_j`: :math:`s_i\\leftrightarrow -(s_j+1)` (e.g., particle-hole symmetry)
		
	These two operations already comprise a variety of symmetries, including translation, parity (reflection) and 
	population inversion. For a specific example, see below.

	The supported operator strings for `spinless_fermion_basis_general` are:

	.. math::
		\\begin{array}{cccc}
			\\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}    \\newline	
			\\texttt{spinless_fermion_basis_general}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}      \\newline
		\\end{array}

	Examples
	--------

	The code snippet below shows how to use the `spinless_fermion_basis_general` class to construct the basis in the zero momentum sector of positive parity for the fermion Hamiltonian 

	.. math::
		H=-J\\sum_{\\langle ij\\rangle} c^\\dagger_{i}c_j + \\mathrm{h.c.} - \\mu\\sum_j n_j + U \\sum_{\\langle ij\\rangle} n_{i} n_j

	Moreover, it demonstrates how to pass user-defined symmetries to the `boson_basis_general` constructor. In particular,
	we do translation invariance and parity (reflection) (along each lattice direction).

	.. literalinclude:: ../../doc_examples/spinless_fermion_basis_general-example.py
		:linenos:
		:language: python
		:lines: 7-


	"""
	

	def __init__(self,N,Nf=None,nf=None,Ns_block_est=None,**blocks):
		"""Intializes the `spinless_fermion_basis_general` object (basis for fermionic operators).

		Parameters
		-----------
		L: int
			Length of chain/number of sites.
		Nf: {int,list}, optional
			Number of fermions in chain. Can be integer or list to specify one or more particle sectors.
		nf: float, optional
			Density of fermions in chain (fermions per site).
		Ns_block_est: int, optional
			Overwrites the internal estimate of the size of the reduced Hilbert space for the given symmetries. This can be used to help conserve memory if the exact size of the H-space is known ahead of time. 	
		**blocks: optional
			keyword arguments which pass the symmetry generator arrays. For instance:

			>>> basis(...,kxblock=(Q,q),...)

			The keys of the symmetry sector, e.g. `kxblock`, can be chosen arbitrarily by the user. The
			values are tuples where the first entry contains the symmetry transformation :math:`Q` acting on the
			lattice sites (see class example), and the second entry is an integer :math:`q` to label the symmetry
			sector.

		"""

		_Np = blocks.get("_Np")
		if _Np is not None:
			blocks.pop("_Np")

		if Nf is not None and nf is not None:
			raise ValueError("cannot use 'nf' and 'Nf' simultaineously.")
		if Nf is None and nf is not None:
			Nf = int(nf*N)

		basis_general.__init__(self,N,**blocks)
		self._check_pcon = False
		count_particles = False
		if _Np is not None and Nf is None:
			count_particles = True
			if type(_Np) is not int:
				raise ValueError("_Np must be integer")
			if _Np >= -1:
				if _Np+1 > N: 
					Nf = list(range(N+1))
				elif _Np==-1:
					Nf = None
				else:
					Nf = list(range(_Np+1))
			else:
				raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")

		if Nf is None:
			Ns = (1<<N)	
		elif type(Nf) is int:
			self._check_pcon = True
			Ns = comb(N,Nf,exact=True)
		else:
			try:
				Np_iter = iter(Nf)
			except TypeError:
				raise TypeError("Nf must be integer or iteratable object.")
			Ns = 0
			for Nf in Np_iter:
				if Nf > N or Nf < 0:
					raise ValueError("particle number Nf must satisfy: 0 <= Nf <= N")
				Ns += comb(N,Nf,exact=True)

		if len(self._pers)>0:
			if Ns_block_est is None:
				Ns = int(float(Ns)/_np.multiply.reduce(self._pers))*4
			else:
				if type(Ns_block_est) is not int:
					raise TypeError("Ns_block_est must be integer value.")
				if Ns_block_est <= 0:
					raise ValueError("Ns_block_est must be an integer > 0")
					
				Ns = Ns_block_est


		Ns = max(Ns,1000)
		if N<=32:
			basis = _np.zeros(Ns,dtype=_np.uint32)
			n     = _np.zeros(Ns,dtype=self._n_dtype)
			self._core = spinless_fermion_basis_core_wrap_32(N,self._maps,self._pers,self._qs)
		elif N<=64:
			basis = _np.zeros(Ns,dtype=_np.uint64)
			n     = _np.zeros(Ns,dtype=self._n_dtype)
			self._core = spinless_fermion_basis_core_wrap_64(N,self._maps,self._pers,self._qs)
		else:
			raise ValueError("system size N must be <=64.")

		self._sps=2
		# if count_particles and (Nf is not None):
		# 	Np_list = _np.zeros_like(basis,dtype=_np.uint8)
		# 	self._Ns = self._core.make_basis(basis,n,Np=Nf,count=Np_list)
		# 	if self._Ns < 0:
		# 			raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

		# 	basis,ind = _np.unique(basis,return_index=True)
		# 	if self.Ns != basis.shape[0]:
		# 		basis = basis[1:]
		# 		ind = ind[1:]

		# 	self._basis = basis[::-1].copy()
		# 	self._n = n[ind[::-1]].copy()
		# 	self._Np_list = Np_list[ind[::-1]].copy()
		# else:
		# 	self._Ns = self._core.make_basis(basis,n,Np=Nf)
		# 	if self._Ns < 0:
		# 			raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

		# 	basis,ind = _np.unique(basis,return_index=True)
		# 	if self.Ns != basis.shape[0]:
		# 		basis = basis[1:]
		# 		ind = ind[1:]
				
		# 	self._basis = basis[::-1].copy()
		# 	self._n = n[ind[::-1]].copy()

		if count_particles and (Nf is not None):
			Np_list = _np.zeros_like(basis,dtype=_np.uint8)
			Ns = self._core.make_basis(basis,n,Np=Nf,count=Np_list)
		else:
			Np_list = None
			Ns = self._core.make_basis(basis,n,Np=Nf)

		if Ns < 0:
				raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

		if type(Nf) is int or Nf is None:
			if Ns > 0:
				self._basis = basis[Ns-1::-1].copy()
				self._n = n[Ns-1::-1].copy()
				if Np_list is not None: self._Np_list = Np_list[Ns-1::-1].copy()
			else:
				self._basis = _np.array([],dtype=basis.dtype)
				self._n = _np.array([],dtype=n.dtype)
				if Np_list is not None: self._Np_list = _np.array([],dtype=Np_list.dtype)
		else:
			ind = _np.argsort(basis[:Ns],kind="heapsort")[::-1]
			self._basis = basis[ind].copy()
			self._n = n[ind].copy()
			if Np_list is not None: self._Np_list = Np_list[ind].copy()



		self._Ns = Ns
		self._N = N
		self._index_type = _np.min_scalar_type(-self._Ns)
		self._operators = ("availible operators for ferion_basis_1d:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tn: number operator"+
							"\n\tz: c-symm number operator")

		self._allowed_ops=set(["I","n","+","-","z"])
		self._reduce_n_dtype()

	@property
	def _fermion_basis(self):
		return True 

	def _sort_opstr(self,op):
		if op[0].count("|") > 0:
			raise ValueError("'|' character found in op: {0},{1}".format(op[0],op[1]))
		if len(op[0]) != len(op[1]):
			raise ValueError("number of operators in opstr: {0} not equal to length of indx {1}".format(op[0],op[1]))

		op = list(op)
		zipstr = list(zip(op[0],op[1]))
		if zipstr:
			n = len(zipstr)
			swapped = True
			anticommutes = 1
			while swapped:
				swapped = False
				for i in range(n-1):
					if zipstr[i][1] > zipstr[i+1][1]:
						temp = zipstr[i]
						zipstr[i] = zipstr[i+1]
						zipstr[i+1] = temp
						swapped = True

						if zipstr[i][0] in ["+","-"] and zipstr[i+1][0] in ["+","-"]:
							anticommutes *= -1

			op1,op2 = zip(*zipstr)
			op[0] = "".join(op1)
			op[1] = tuple(op2)
			op[2] *= anticommutes
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
		return spinless_fermion_basis_general._sort_opstr(self,op) # return the sorted op.


	def _expand_opstr(self,op,num):
		op = list(op)
		op.append(num)
		return [tuple(op)]	



def process_spinful_map(N,map,q):
	map = _np.asarray(map,dtype=_np.int32)
	if len(map) == N:
		i_map = map.copy()
		i_map[map<0] = -(i_map[map<0] + 1) + N # site mapping
		adv_map = _np.hstack((i_map,(i_map+N)%(2*N)))
		return adv_map,q
	else:
		return map,q

# general basis for hardcore bosons/spin-1/2
class spinful_fermion_basis_general(spinless_fermion_basis_general):
	"""Constructs basis for spinful fermion operators for USER-DEFINED symmetries.

	Any unitary symmetry transformation :math:`Q` of multiplicity :math:`m_Q` (:math:`Q^{m_Q}=1`) has
	eigenvalues :math:`\\exp(-2\\pi i q/m_Q)`, labelled by an ingeter :math:`q\\in\\{0,1,\\dots,m_Q-1\\}`.
	These integers :math:`q` are used to define the symmetry blocks.

	For instance, if :math:`Q=P` is parity (reflection), then :math:`q=0,1`. If :math:`Q=T` is translation by one lattice site,
	then :math:`q` labels the mometum blocks in the same fashion as for the `..._basis_1d` classes. 

	User-defined symmetries with the `spinful_fermion_basis_general` class can be programmed in two equivalent ways: *simple* and *advanced*.
		* *simple* symmetry definition (see optional argument `simple_symm`) uses the pipe symbol, |, in the operator string (see site-coupling lists in example below) to distinguish between the spin-up and spin-down species. Suppose we have a system of L sites. In the *simple* case, the lattice sites are enumerated :math:`s=(s_0,s_1,\\dots,s_{L-1})` for **both** spin-up and spin-down. There are two types of operations one can perform on the sites:
			* exchange the labels of two sites: :math:`s_i \\leftrightarrow s_j` (e.g., translation, parity)
			* invert the **fermion spin** on a given site: :math:`s_i\\leftrightarrow -(s_j+1)` (e.g., fermion spin inversion)

		* *advanced* symmetry definition (see optional argument `simple_symm`) does NOT use any pipe symbol in the operator string to distinguish between the spin-up and spin-down species. In the *advanced* case, the sites are enumerated :math:`s=(s_0,s_1,\\dots,s_{L-1}; s_{L},s_{L+1},\\dots,s_{2L-1})`, where the first L sites label spin-up, and the last L sites -- spin-down. There are two types of operations one can perform on the sites:
			* exchange the labels of two sites: :math:`s_i \\leftrightarrow s_j` (e.g., translation, parity, fermion spin inversion)
			* invert the **fermion population** on a given site with appropriate sign flip :math:`c_j^\\dagger\\to (-1)^j c_j`: :math:`s_i\\leftrightarrow -(s_j+1)` (e.g., particle-hole symmetry)
	

	These two operations already comprise a variety of symmetries, including translation, parity (reflection), fermion-spin inversion
	and particle-hole like symmetries. For specific examples, see below.

	The supported operator strings for `spinful_fermion_basis_general` are:

	.. math::
		\\begin{array}{cccc}
			\\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}    \\newline	
			\\texttt{spinful_fermion_basis_general}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}      \\newline
		\\end{array}

	Notes
	-----

	The definition of the operation :math:`s_i\\leftrightarrow -(s_j+1)` **differs** for the *simple* and *advanced* cases. 


	Examples
	--------

	The code snippets below show how to construct the two-dimensional Fermi-Hubbard model with onsite interactions.
	
	.. math::
		H = -J \\sum_{\\langle ij\\rangle,\\sigma} c^\\dagger_{i\\sigma}c_{j\\sigma} + \\mathrm{h.c.} - \\mu\\sum_{j,\\sigma} n_{j\\sigma} + U\\sum_j n_{j\\uparrow} n_{j\\downarrow}

	The first code snippet demonstrates how to pass **simple** user-defined symmetries to the `spinful_fermion_basis_general` constructor. In particular,
	we do translation invariance and fermion spin inversion.

	.. literalinclude:: ../../doc_examples/spinful_fermion_basis_general-simple-example.py
		:linenos:
		:language: python
		:lines: 7-

	The second code snippet demonstrates how to pass **advanced** user-defined symmetries to the `spinful_fermion_basis_general` constructor. Like above,
	we do translation invariance and fermion spin inversion.

	.. literalinclude:: ../../doc_examples/spinful_fermion_basis_general-adv-example.py
		:linenos:
		:language: python
		:lines: 7-

	The third code snippet demonstrates how to pass **advanced** user-defined particle-hole symemtry to the `spinful_fermion_basis_general` constructor with translational invariance.

	.. literalinclude:: ../../doc_examples/spinful_fermion_basis_general-adv_ph-example.py
		:linenos:
		:language: python
		:lines: 7-

	"""
	def __init__(self,N,Nf=None,nf=None,Ns_block_est=None,simple_symm=True,**blocks):
		"""Intializes the `spinful_fermion_basis_general` object (basis for fermionic operators).

		Parameters
		-----------
		L: int
			Length of chain/number of sites.
		Nf: {int,list}, optional
			Number of fermions in chain. Can be integer or list to specify one or more particle sectors.
		nf: float, optional
			Density of fermions in chain (fermions per site).
		Ns_block_est: int, optional
			Overwrites the internal estimate of the size of the reduced Hilbert space for the given symmetries. This can be used to help conserve memory if the exact size of the H-space is known ahead of time. 
		simple_sym: bool, optional
			Flags whidh toggles the setting for the types of mappings and operator strings the basis will use. See this tutorial for more details. 
		**blocks: optional
			keyword arguments which pass the symmetry generator arrays. For instance:

			>>> basis(...,kxblock=(Q,q),...)

			The keys of the symmetry sector, e.g. `kxblock`, can be chosen arbitrarily by the user. The
			values are tuples where the first entry contains the symmetry transformation :math:`Q` acting on the
			lattice sites (see class example), and the second entry is an integer :math:`q` to label the symmetry
			sector.

		"""



		# Nf = [(Nup,Ndown),...]
		# Nup is left side of basis sites 0 - N-1
		# Ndown is right side of basis sites N - 2*N-1

		_Np = blocks.get("_Np")
		if _Np is not None:
			blocks.pop("_Np")

		if Nf is not None and nf is not None:
			raise ValueError("cannot use 'nf' and 'Nf' simultaineously.")
		if Nf is None and nf is not None:
			Nf = (int(nf[0]*N),int(nf[1]*N))

		if any((type(map) is not tuple) and (len(map)!=2) for map in blocks.values() if map is not None):
			raise ValueError("blocks must contain tuple: (map,q).")

		self._simple_symm = simple_symm

		if simple_symm:
			if any(len(item[0]) != N for item in blocks.values() if item is not None):
				raise ValueError("Simple symmetries must have mapping of length N.")
		else:
			if any(len(item[0]) != 2*N for item in blocks.values() if item is not None):
				raise ValueError("Simple symmetries must have mapping of length N.")


		new_blocks = {key:process_spinful_map(N,*item) for key,item in blocks.items() if item is not None}

		blocks.update(new_blocks)

		basis_general.__init__(self,2*N,**blocks)
		self._check_pcon = False
		count_particles = False
		if _Np is not None and Nf is None:
			count_particles = True
			if type(_Np) is not int:
				raise ValueError("_Np must be integer")
			if _Np >= -1:
				if _Np+1 > N: 
					Nf =  []
					for n in range(N+1):
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
			Ns = (1<<N)**2
		else:
			if type(Nf) is tuple:
				if len(Nf)==2:
					Nup,Ndown = Nf
					if (type(Nup) is int) and type(Ndown) is int:
						Ns = comb(N,Nup,exact=True)*comb(N,Ndown,exact=True)
					else:
						raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")
				else:
					Nf = list(Nf)
					if any((type(tup)is not tuple) and len(tup)!=2 for tup in Nf):
						raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")		

					Ns = 0
					for Nup,Ndown in Nf:
						if Nup > N or Nup < 0:
							raise ValueError("particle numbers in Nf must satisfy: 0 <= n <= N")
						if Ndown > N or Ndown < 0:
							raise ValueError("particle numbers in Nf must satisfy: 0 <= n <= N")
						Ns += comb(N,Nup,exact=True)*comb(N,Ndown,exact=True)

			else:
				try:
					Nf_iter = iter(Nf)
				except TypeError:
					raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")

				if any((type(tup)is not tuple) and len(tup)!=2 for tup in Nf):
					raise ValueError("Nf must be tuple of integers or iteratable object of tuples.")

				Nf = list(Nf)

				Ns = 0
				for Nup,Ndown in Nf:
					if Nup > N or Nup < 0:
						raise ValueError("particle numbers in Nf must satisfy: 0 <= n <= N")
					if Ndown > N or Ndown < 0:
						raise ValueError("particle numbers in Nf must satisfy: 0 <= n <= N")

					Ns += comb(N,Nup,exact=True)*comb(N,Ndown,exact=True)

		if len(self._pers)>0:
			if Ns_block_est is None:
				Ns = int(float(Ns)/_np.multiply.reduce(self._pers))*4
			else:
				if type(Ns_block_est) is not int:
					raise TypeError("Ns_block_est must be integer value.")
				if Ns_block_est <= 0:
					raise ValueError("Ns_block_est must be an integer > 0")
										
				Ns = Ns_block_est

		Ns = max(Ns,1000)
		if N<=16:
			basis = _np.zeros(Ns,dtype=_np.uint32)
			n     = _np.zeros(Ns,dtype=self._n_dtype)
			self._core = spinful_fermion_basis_core_wrap_32(N,self._maps,self._pers,self._qs)
		elif N<=32:
			basis = _np.zeros(Ns,dtype=_np.uint64)
			n     = _np.zeros(Ns,dtype=self._n_dtype)
			self._core = spinful_fermion_basis_core_wrap_64(N,self._maps,self._pers,self._qs)
		else:
			raise ValueError("system size N must be <=32.")

		self._sps=2
		if count_particles and (Nf is not None):
			Np_list = _np.zeros_like(basis,dtype=_np.uint8)
			Ns = self._core.make_basis(basis,n,Np=Nf,count=Np_list)
		else:
			Np_list = None
			Ns = self._core.make_basis(basis,n,Np=Nf)

		if Ns < 0:
				raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

		if type(Nf) is int or Nf is None:
			if Ns > 0:
				self._basis = basis[Ns-1::-1].copy()
				self._n = n[Ns-1::-1].copy()
				if Np_list is not None: self._Np_list = Np_list[Ns-1::-1].copy()
			else:
				self._basis = _np.array([],dtype=basis.dtype)
				self._n = _np.array([],dtype=n.dtype)
				if Np_list is not None: self._Np_list = _np.array([],dtype=Np_list.dtype)
		else:
			ind = _np.argsort(basis[:Ns],kind="heapsort")[::-1]
			self._basis = basis[ind].copy()
			self._n = n[ind].copy()
			if Np_list is not None: self._Np_list = Np_list[ind].copy()



		self._Ns = Ns
		self._N = 2*N
		self._index_type = _np.min_scalar_type(-self._Ns)
		self._operators = ("availible operators for ferion_basis_1d:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tn: number operator"+
							"\n\tz: c-symm number operator")
		self._allowed_ops=set(["I","n","+","-","z"])
		self._reduce_n_dtype()

	def _Op(self,opstr,indx,J,dtype):
		indx = _np.array(indx,dtype=_np.int32)
		if self._simple_symm:
			if opstr.count("|") == 0: 
				raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr,indx))

			i = opstr.index("|")
			
			indx[i:] += (self._N//2)
			opstr=opstr.replace("|","")

		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')

		if _np.any(indx >= 2*self._N) or _np.any(indx < 0):
			raise ValueError('values in indx falls outside of system')

		extra_ops = set(opstr) - self._allowed_ops
		if extra_ops:
			raise ValueError("unrecognized characters {} in operator string.".format(extra_ops))

		if self._Ns <= 0:
			return _np.array([],dtype=dtype),_np.array([],dtype=self._index_type),_np.array([],dtype=self._index_type)
	
		col = _np.zeros(self._Ns,dtype=self._index_type)
		row = _np.zeros(self._Ns,dtype=self._index_type)
		ME = _np.zeros(self._Ns,dtype=dtype)

		self._core.op(row,col,ME,opstr,indx,J,self._basis,self._n)

		mask = _np.logical_not(_np.logical_or(_np.isnan(ME),_np.abs(ME)==0.0))
		col = col[mask]
		row = row[mask]
		ME = ME[mask]

		return ME,row,col

	@property
	def _fermion_basis(self):
		return True 

	def _get__str__(self):
		def get_state(b):
			b = int(b)
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

	def _sort_opstr(self,op):

		if self._simple_symm:
			op = list(op)
			opstr = op[0]
			indx  = op[1]

			if opstr.count("|") == 0: 
				raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr,indx))
		
			if opstr.count("|") > 1: 
				raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

			if len(opstr)-opstr.count("|") != len(indx):
				raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

			i = opstr.index("|")
			indx_left = indx[:i]
			indx_right = indx[i:]

			opstr_left,opstr_right=opstr.split("|")

			op1 = list(op)
			op1[0] = opstr_left
			op1[1] = tuple(indx_left)
			op1[2] = 1.0

			op2 = list(op)
			op2[0] = opstr_right
			op2[1] = tuple(indx_right)
			op2[2] = 1.0
			
			op1 = spinless_fermion_basis_general._sort_opstr(self,op1)
			op2 = spinless_fermion_basis_general._sort_opstr(self,op2)

			op[0] = "|".join((op1[0],op2[0]))
			op[1] = op1[1] + op2[1]
			op[2] *= op1[2] * op2[2]
			return tuple(op)
		else:
			return spinless_fermion_basis_general._sort_opstr(self,op)

	def _hc_opstr(self,op):
		if self._simple_symm:
			op = list(op)
			opstr = op[0]
			indx  = op[1]

			if opstr.count("|") == 0: 
				raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr,indx))
		
			if opstr.count("|") > 1: 
				raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

			if len(opstr)-opstr.count("|") != len(indx):
				raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

			i = opstr.index("|")
			indx_left = indx[:i]
			indx_right = indx[i:]

			opstr_left,opstr_right=opstr.split("|")

			op1 = list(op)
			op1[0] = opstr_left
			op1[1] = tuple(indx_left)

			op2 = list(op)
			op2[0] = opstr_right
			op2[1] = tuple(indx_right)
			op2[2] = 1.0
			
			op1 = spinless_fermion_basis_general._hc_opstr(self,op1)
			op2 = spinless_fermion_basis_general._hc_opstr(self,op2)

			op[0] = "|".join((op1[0],op2[0]))
			op[1] = op1[1] + op2[1]
			op[2] = op1[2] * op2[2]

			return tuple(op)
		else:
			return spinless_fermion_basis_general._hc_opstr(self,op)

	def _non_zero(self,op):
		if self._simple_symm:
			op = list(op)
			opstr = op[0]
			indx  = op[1]

			if opstr.count("|") == 0: 
				raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr,indx))
		
			if opstr.count("|") > 1: 
				raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

			if len(opstr)-opstr.count("|") != len(indx):
				raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

			i = opstr.index("|")
			indx_left = indx[:i]
			indx_right = indx[i:]

			opstr_left,opstr_right=opstr.split("|")

			op1 = list(op)
			op1[0] = opstr_left
			op1[1] = indx_left

			op2 = list(op)
			op2[0] = opstr_right
			op2[1] = indx_right

			return (spinless_fermion_basis_general._non_zero(self,op1) and spinless_fermion_basis_general._non_zero(self,op2))
		else:
			return spinless_fermion_basis_general._non_zero(self,op)

	def _simple_to_adv(self,op):
			op = list(op)
			opstr = op[0]

			i = opstr.index("|")
			indx = list(op[1])
			indx_left = tuple(indx[:i])
			indx_right = tuple([j+self._N//2 for j in indx[i:]])

			opstr_left,opstr_right=opstr.split("|",1)

			op[0] = "".join([opstr_left,opstr_right])
			op[1] = indx_left+indx_right

			return tuple(op)

	def _expand_opstr(self,op,num):
		if self._simple_symm:
			op = list(op)
			opstr = op[0]
			indx  = op[1]
		
			if opstr.count("|") > 1: 
				raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

			if len(opstr)-opstr.count("|") != len(indx):
				raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

			i = opstr.index("|")
			indx_left = indx[:i]
			indx_right = indx[i:]

			opstr_left,opstr_right=opstr.split("|")

			op1 = list(op)
			op1[0] = opstr_left
			op1[1] = indx_left
			op1[2] = 1.0

			op2 = list(op)
			op2[0] = opstr_right
			op2[1] = indx_right

			op1_list = spinless_fermion_basis_general._expand_opstr(self,op1,num)
			op2_list = spinless_fermion_basis_general._expand_opstr(self,op2,num)

			op_list = []
			for new_op1 in op1_list:
				for new_op2 in op2_list:
					new_op = list(new_op1)
					new_op[0] = "|".join((new_op1[0],new_op2[0]))
					new_op[1] += tuple(new_op2[1])
					new_op[2] *= new_op2[2]


					op_list.append(tuple(new_op))

			return tuple(op_list)
		else:
			return spinless_fermion_basis_general._expand_opstr(self,op,num)



	def _check_symm(self,static,dynamic,photon_basis=None):
		if photon_basis is None:
			basis_sort_opstr = lambda op:spinless_fermion_basis_general._sort_opstr(self,op)
			static_list,dynamic_list = self._get_local_lists(static,dynamic)
		else:
			basis_sort_opstr = photon_basis._sort_opstr
			static_list,dynamic_list = photon_basis._get_local_lists(static,dynamic)

		if self._simple_symm:
			static_list = [self._simple_to_adv(op) for op in static_list]
			dynamic_list = [self._simple_to_adv(op) for op in dynamic_list]


		static_blocks = {}
		dynamic_blocks = {}
		for block,map in self._maps_dict.items():
			key = block+" symm"
			odd_ops,missing_ops = _check_symm_map(map,basis_sort_opstr,static_list)
			if odd_ops or missing_ops:
				static_blocks[key] = (tuple(odd_ops),tuple(missing_ops))

			odd_ops,missing_ops = _check_symm_map(map,basis_sort_opstr,dynamic_list)
			if odd_ops or missing_ops:
				dynamic_blocks[key] = (tuple(odd_ops),tuple(missing_ops))


		return static_blocks,dynamic_blocks
