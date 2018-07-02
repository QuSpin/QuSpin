from ._basis_general_core import boson_basis_core_wrap_32,boson_basis_core_wrap_64
from .base_hcb import hcb_basis_general
from .base_general import basis_general
import numpy as _np
from scipy.misc import comb


def H_dim(N,length,m_max):
    """
    Returns the total number of states in the bosonic Hilbert space

    --- arguments:

    N: total number of bosons in lattice
    length: total number of sites
    m_max+1: max number of states per site 
    """
    Ns = 0
    for r in range(N//(m_max+1)+1):
        r_2 = N - r*(m_max+1)
        if r % 2 == 0:
            Ns +=  comb(length,r,exact=True) * comb(length + r_2 - 1,r_2,exact=True)
        else:
            Ns += -comb(length,r,exact=True) * comb(length + r_2 - 1,r_2,exact=True)

    return Ns




def get_basis_type(N, Np, sps, **blocks):
    # calculates the datatype which will fit the largest representative state in basis
    if Np is None:
        # if no particle conservation the largest representative is sps**N
        dtype = _np.min_scalar_type(int(sps**N-1))
        return _np.result_type(dtype,_np.uint32)
    else:
        # if particles are conservated the largest representative is placing all particles as far left
        # as possible. 
        l=Np//(sps-1)
        s_max = sum((sps-1)*sps**(N-1-i)  for i in range(l))
        s_max += (Np%(sps-1))*sps**(N-l-1)
        dtype = _np.min_scalar_type(int(s_max))
        return _np.result_type(dtype,_np.uint32)


# general basis for hardcore bosons/spin-1/2
class boson_basis_general(hcb_basis_general,basis_general):
	"""Constructs basis for boson operators for USER-DEFINED symmetries.

	Any unitary symmetry transformation :math:`Q` of multiplicity :math:`m_Q` (:math:`Q^{m_Q}=1`) has
	eigenvalues :math:`\\exp(-2\\pi i q/m_Q)`, labelled by an ingeter :math:`q\\in\\{0,1,\\dots,m_Q-1\\}`.
	These integers :math:`q` are used to define the symmetry blocks.

	For instance, if :math:`Q=P` is parity (reflection), then :math:`q=0,1`. If :math:`Q=T` is translation by one lattice site,
	then :math:`q` labels the mometum blocks in the same fashion as for the `..._basis_1d` classes. 

	User-defined symmetries with the `boson_basis_general` class can be programmed as follows. Suppose we have a system of
	L sites, enumerated :math:`s=(s_0,s_1,\\dots,s_{L-1})`. There are two types of operations one can perform on the sites:
		* exchange the labels of two sites: :math:`s_i \\leftrightarrow s_j` (e.g., translation, parity)
		* invert the population on a given site: :math:`s_i\\leftrightarrow -(s_j+1)` (e.g., particle-hole symmetry, hardcore bosons only)

	These two operations already comprise a variety of symmetries, including translation, parity (reflection) and 
	spin inversion. For a specific example, see below.

	The supported operator strings for `boson_basis_general` are:

	.. math::
		\\begin{array}{cccc}
			\\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}     \\newline	
			\\texttt{boson_basis_general}&   \\hat{1}        &   \\hat b^\\dagger      &       \\hat b          & \\hat b^\\dagger b     &  \\hat b^\\dagger\\hat b - \\frac{\\mathrm{sps}-1}{2}  \\newline
		\\end{array}

	Notes
	-----
	* if `Nb` or `nb` are specified, by default `sps` is set to the number of bosons on the lattice.
	* if `sps` is specified, while `Nb` or `nb` are not, all particle sectors are filled up to the maximumal 
		occupation. 
	* if `Nb` or `nb` and `sps` are specified, the finite boson basis is constructed with the local Hilbert space 
		restrited by `sps`.

	Examples
	--------

	The code snippet below shows how to construct the two-dimensional Bose-Hubbard model.
	
	.. math::
		H = -J \\sum_{\\langle ij\\rangle} b^\dagger_i b_j + \\mathrm{h.c.} - \\mu\\sum_j n_j + \\frac{U}{2}\\sum_j n_j(n_j-1)

	Moreover, it demonstrates how to pass user-defined symmetries to the `boson_basis_general` constructor. In particular,
	we do translation invariance and parity (reflection) (along each lattice direction).

	.. literalinclude:: ../../doc_examples/boson_basis_general-example.py
		:linenos:
		:language: python
		:lines: 7-


	"""
	def __init__(self,N,Nb=None,nb=None,sps=None,Ns_block_est=None,**blocks):
		"""Intializes the `boson_basis_general` object (basis for bosonic operators).

		Parameters
		-----------
		N: int
			Number of sites.
		Nb: {int,list}, optional
			Number of bosons in chain. Can be integer or list to specify one or more particle sectors.
		nb: float, optional
			Density of bosons in chain (bosons per site).
		sps: int, optional
			Number of states per site (including zero bosons), or on-site Hilbert space dimension.
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

		if sps is None:

			if Nb is not None:
				if nb is not None:
					raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")

			elif nb is not None:
				if Nb is not None:
					raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")

				Nb = int(nb*N)
			else:
				raise ValueError("expecting value for 'Nb','nb' or 'sps'")
		else:
			if Nb is not None:
				if nb is not None:
					raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")

			elif nb is not None:
				Nb = int(nb*N)

		
		self._sps = sps

		self._allowed_ops=set(["I","z","n","+","-"])


		if self._sps == 2:

			self._operators = ("availible operators for boson_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tn: number operator"+
								"\n\tz: c-symm number operator")

			hcb_basis_general.__init__(self,N,Nb=Nb,_Np=_Np,**blocks)
		
		else:

			self._operators = ("availible operators for ferion_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tn: number operator"+
								"\n\tz: ph-symm number operator")


			basis_general.__init__(self,N,**blocks)
			self._check_pcon = False
			count_particles = False
			if _Np is not None and Nb is None:
				count_particles = True
				if type(_Np) is not int:
					raise ValueError("_Np must be integer")
				if _Np >= -1:
					if _Np+1 > N: 
						Nb = list(range(N+1))
					elif _Np==-1:
						Nb = None
					else:
						Nb = list(range(_Np+1))
				else:
					raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")

			if Nb is None:
				Ns = sps**N
				basis_type = get_basis_type(N,Nb,sps)
			elif type(Nb) is int:
				self._check_pcon = True

				if self._sps is None:
					self._sps = Nb

				Ns = H_dim(Nb,N,self._sps-1)
				basis_type = get_basis_type(N,Nb,self._sps)
			else:
				try:
					Np_iter = iter(Nb)
				except TypeError:
					raise TypeError("Nb must be integer or iteratable object.")

				if self._sps is None:
					self._sps = max(Nb)				

				Ns = 0
				for b in Nb:
					Ns += H_dim(b,N,self._sps-1)

				basis_type = get_basis_type(N,max(Nb),self._sps)

			if len(self._pers)>0:
				if Ns_block_est is None:
					Ns = int(float(Ns)/_np.multiply.reduce(self._pers))*self._sps
				else:
					if type(Ns_block_est) is not int:
						raise TypeError("Ns_block_est must be integer value.")
					if Ns_block_est <= 0:
						raise ValueError("Ns_block_est must be an integer > 0")						
					Ns = Ns_block_est

			Ns = max(Ns,1000)

			if basis_type==_np.uint32:
				basis = _np.zeros(Ns,dtype=_np.uint32)
				n     = _np.zeros(Ns,dtype=self._n_dtype)
				self._core = boson_basis_core_wrap_32(N,self._sps,self._maps,self._pers,self._qs)
			elif basis_type==_np.uint64:
				basis = _np.zeros(Ns,dtype=_np.uint64)
				n     = _np.zeros(Ns,dtype=self._n_dtype)
				self._core = boson_basis_core_wrap_64(N,self._sps,self._maps,self._pers,self._qs)
			else:
				raise ValueError("states can't be represented as 64-bit unsigned integer")

			# if count_particles and (Nb_list is not None):
			# 	Np_list = _np.zeros_like(basis,dtype=_np.uint8)
			# 	self._Ns = self._core.make_basis(basis,n,Np=Nb,count=Np_list)
			# 	if self._Ns < 0:
			# 		raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

			# 	basis,ind = _np.unique(basis,return_index=True)
			# 	if self.Ns != basis.shape[0]:
			# 		basis = basis[1:]
			# 		ind = ind[1:]

			# 	self._basis = basis[::-1].copy()
			# 	self._n = n[ind[::-1]].copy()
			# 	self._Np_list = Np_list[ind[::-1]].copy()
			# else:
			# 	self._Ns = self._core.make_basis(basis,n,Np=Nb)
			# 	if self._Ns < 0:
			# 		raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

			# 	basis,ind = _np.unique(basis,return_index=True)
			# 	if self.Ns != basis.shape[0]:
			# 		basis = basis[1:]
			# 		ind = ind[1:]
			# 	self._basis = basis[::-1].copy()
			# 	self._n = n[ind[::-1]].copy()


			if count_particles and (Nb is not None):
				Np_list = _np.zeros_like(basis,dtype=_np.uint8)
				Ns = self._core.make_basis(basis,n,Np=Nb,count=Np_list)
			else:
				Np_list = None
				Ns = self._core.make_basis(basis,n,Np=Nb)

			if Ns < 0:
					raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

			if type(Nb) is int or Nb is None:
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

			self._reduce_n_dtype()

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
		op = list(op)
		op.append(num)
		return [tuple(op)]	




