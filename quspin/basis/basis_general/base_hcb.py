from ._basis_general_core import hcb_basis_core_wrap_32,hcb_basis_core_wrap_64
from .base_general import basis_general
import numpy as _np
from scipy.misc import comb

# general basis for hardcore bosons/spin-1/2
class hcb_basis_general(basis_general):
	def __init__(self,N,Nb=None,_Np=None,**kwargs):
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
			Ns = 0
			for Nb in Np_iter:
				if Nb > N or Nb < 0:
					raise ValueError("particle number Nb must satisfy: 0 <= Nb <= N")
				Ns += comb(N,Nb,exact=True)

		if len(self._pers)>0:
			Ns = max(int(float(Ns)/_np.multiply.reduce(self._pers))*2,1000)


		if N<=32:
			self._basis = _np.zeros(Ns,dtype=_np.uint32)
			self._n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = hcb_basis_core_wrap_32(N,self._maps,self._pers,self._qs)
		elif N<=64:
			self._basis = _np.zeros(Ns,dtype=_np.uint64)
			self._n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = hcb_basis_core_wrap_64(N,self._maps,self._pers,self._qs)
		else:
			raise ValueError("system size N must be <=64.")

		self._sps=2
		if count_particles and (Nb is not None):
			self._Np_list = _np.zeros_like(self._basis,dtype=_np.uint8)
			self._Ns = self._core.make_basis(self._basis,self._n,Np=Nb,count=self._Np_list)
			self._basis = self._basis[:self._Ns]
			arg = self._basis.argsort()[::-1]
			self._basis = self._basis[arg].copy()
			self._n = self._n[arg].copy()
			self._Np_list = self._Np_list[arg].copy()
		else:
			self._Ns = self._core.make_basis(self._basis,self._n,Np=Nb)
			self._basis = self._basis[:self._Ns]
			arg = self._basis.argsort()[::-1]
			self._basis = self._basis[arg].copy()
			self._n = self._n[arg].copy()

		self._N = N
		self._index_type = _np.min_scalar_type(-self._Ns)
		self._allowed_ops=set(["I","x","y","z","+","-"])

		self._check_symm = None





