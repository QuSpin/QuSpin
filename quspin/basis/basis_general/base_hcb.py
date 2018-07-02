from ._basis_general_core import hcb_basis_core_wrap_32,hcb_basis_core_wrap_64
from .base_general import basis_general
import numpy as _np
from scipy.misc import comb
import cProfile

# general basis for hardcore bosons/spin-1/2
class hcb_basis_general(basis_general):
	def __init__(self,N,Nb=None,Ns_block_est=None,_Np=None,**kwargs):
		basis_general.__init__(self,N,**kwargs)
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
			Ns = (1<<N)	
		elif type(Nb) is int:
			self._check_pcon = True
			Ns = comb(N,Nb,exact=True)
		else:
			try:
				Np_iter = iter(Nb)
			except TypeError:
				raise TypeError("Nb must be integer or iteratable object.")
			Nb = list(Nb)
			Ns = 0
			for b in Nb:
				if b > N or b < 0:
					raise ValueError("particle number Nb must satisfy: 0 <= Nb <= N")
				Ns += comb(N,b,exact=True)

		if len(self._pers)>0:
			if Ns_block_est is None:
				Ns = int(float(Ns)/_np.multiply.reduce(self._pers))*2
			else:
				if type(Ns_block_est) is not int:
					raise TypeError("Ns_block_est must be integer value.")
					
				Ns = Ns_block_est

		Ns = max(Ns,1000)

		if N<=32:
			basis = _np.zeros(Ns,dtype=_np.uint32)
			n     = _np.zeros(Ns,dtype=self._n_dtype)
			self._core = hcb_basis_core_wrap_32(N,self._maps,self._pers,self._qs)
		elif N<=64:
			basis = _np.zeros(Ns,dtype=_np.uint64)
			n     = _np.zeros(Ns,dtype=self._n_dtype)
			self._core = hcb_basis_core_wrap_64(N,self._maps,self._pers,self._qs)
		else:
			raise ValueError("system size N must be <=64.")

		self._sps=2
		
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
		self._allowed_ops=set(["I","x","y","z","+","-","n"])
		self._reduce_n_dtype()





