from ._basis_general_core import higher_spin_basis_core_wrap_32,higher_spin_basis_core_wrap_64
from .base_general import basis_general
from .boson import H_dim,get_basis_type
import numpy as _np
from scipy.misc import comb



# general basis for hardcore bosons/spin-1/2
class higher_spin_basis_general(basis_general):
	def __init__(self,N,Nup=None,sps=None,_Np=None,**kwargs):
		basis_general.__init__(self,N,**kwargs)
		self._check_pcon = False
		count_particles = False
		if _Np is not None and Nup is None:
			count_particles = True
			if type(_Np) is not int:
				raise ValueError("_Np must be integer")
			if _Np >= -1:
				if _Np+1 > N: 
					Nup = list(range(N+1))
				elif _Np==-1:
					Nup = None
				else:
					Nup = list(range(_Np+1))
			else:
				raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")

		if Nup is None and sps is None:
			raise ValueError("must specify number of boons or sps")

		if Nup is not None and sps is None:
			sps = Nup+1

		if Nup is None:
			Ns = sps**N
			basis_type = get_basis_type(N,Nup,sps)
		elif type(Nup) is int:
			self._check_pcon = True
			Ns = H_dim(Nup,N,sps-1)
			basis_type = get_basis_type(N,Nup,sps)
		else:
			try:
				Np_iter = iter(Nup)
			except TypeError:
				raise TypeError("Nup must be integer or iteratable object.")
			Ns = 0
			for Nup in Np_iter:
				Ns += H_dim(Nup,N,sps-1)

			basis_type = get_basis_type(N,max(iter(Nup)),sps)

		if len(self._pers)>0:
			Ns = max(int(float(Ns)/_np.multiply.reduce(self._pers))*2,1000)

		if basis_type==_np.uint32:
			self._basis = _np.zeros(Ns,dtype=_np.uint32)
			self._n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = higher_spin_basis_core_wrap_32(N,sps,self._maps,self._pers,self._qs)
		elif basis_type==_np.uint64:
			self._basis = _np.zeros(Ns,dtype=_np.uint64)
			self._n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = higher_spin_basis_core_wrap_64(N,sps,self._maps,self._pers,self._qs)
		else:
			raise ValueError("states can't be represented as 64-bit unsigned integer")

		self._sps=sps
		if count_particles and (Nup is not None):
			self._Np_list = _np.zeros_like(self._basis,dtype=_np.uint8)
			self._Ns = self._core.make_basis(self._basis,self._n,Np=Nup,count=self._Np_list)
			self._basis = self._basis[:self._Ns]
			arg = self._basis.argsort()[::-1]
			self._basis = self._basis[arg].copy()
			self._n = self._n[arg].copy()
			self._Np_list = self._Np_list[arg].copy()
		else:
			self._Ns = self._core.make_basis(self._basis,self._n,Np=Nup)
			self._basis = self._basis[:self._Ns]
			arg = self._basis.argsort()[::-1]
			self._basis = self._basis[arg].copy()
			self._n = self._n[arg].copy()

		self._N = N
		self._index_type = _np.min_scalar_type(-self._Ns)
		self._allowed_ops=set(["I","z","+","-"])

		self._check_symm = None

