from ._basis_general_core import higher_spin_basis_core_wrap
from .base_general import basis_general
from .boson import H_dim,get_basis_type
import numpy as _np
from scipy.misc import comb



# general basis for higher spin representations
class higher_spin_basis_general(basis_general):
	def __init__(self,N,Nup=None,sps=None,Ns_block_est=None,_Np=None,_make_basis=True,**kwargs):
		basis_general.__init__(self,N,**kwargs)
		self._check_pcon = False
		self._count_particles = False
		if _Np is not None and Nup is None:
			self._count_particles = True
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
			self._basis_dtype = get_basis_type(N,Nup,sps)
		elif type(Nup) is int:
			self._check_pcon = True
			self._get_proj_pcon = True
			Ns = H_dim(Nup,N,sps-1)
			self._basis_dtype = get_basis_type(N,Nup,sps)
		else:
			try:
				Np_iter = iter(Nup)
			except TypeError:
				raise TypeError("Nup must be integer or iteratable object.")
			Ns = 0
			for Nup in Np_iter:
				Ns += H_dim(Nup,N,sps-1)

			self._basis_dtype = get_basis_type(N,max(iter(Nup)),sps)

		if len(self._pers)>0:
			if Ns_block_est is None:
				Ns = int(float(Ns)/_np.multiply.reduce(self._pers))*sps
			else:
				if type(Ns_block_est) is not int:
					raise TypeError("Ns_block_est must be integer value.")
				if Ns_block_est <= 0:
					raise ValueError("Ns_block_est must be an integer > 0")

				Ns = Ns_block_est

		if self._basis_dtype not in [_np.uint32,_np.uint64]:
			raise ValueError("basis type is not representable with uint32 or uint64.")

		self._core = higher_spin_basis_core_wrap(self._basis_dtype,N,sps,self._maps,self._pers,self._qs)

		self._N = N
		self._Ns = Ns
		self._Np = Nup
		

		# make the basisl; make() is function method of base_general
		if _make_basis:		
			self.make()
		else:
			self._Ns=1
			self._basis=_np.zeros(self._Ns,dtype=basis_type)
			self._n=_np.zeros(self._Ns,dtype=basis_type)

		self._sps=sps
		self._allowed_ops=set(["I","z","+","-"])
		
