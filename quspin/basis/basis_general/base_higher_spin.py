from ._basis_general_core import higher_spin_basis_core_wrap_32,higher_spin_basis_core_wrap_64
from .base_general import basis_general
from .boson import H_dim,get_basis_type
import numpy as _np
from scipy.misc import comb



# general basis for higher spin representations
class higher_spin_basis_general(basis_general):
	def __init__(self,N,Nup=None,sps=None,Ns_block_est=None,_Np=None,**kwargs):
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
			if Ns_block_est is None:
				Ns = int(float(Ns)/_np.multiply.reduce(self._pers))*sps
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
			self._core = higher_spin_basis_core_wrap_32(N,sps,self._maps,self._pers,self._qs)
		elif basis_type==_np.uint64:
			basis = _np.zeros(Ns,dtype=_np.uint64)
			n     = _np.zeros(Ns,dtype=self._n_dtype)
			self._core = higher_spin_basis_core_wrap_64(N,sps,self._maps,self._pers,self._qs)
		else:
			raise ValueError("states can't be represented as 64-bit unsigned integer")

		self._sps=sps
		# if count_particles and (Nup is not None):
		# 	Np_list = _np.zeros_like(basis,dtype=_np.uint8)
		# 	self._Ns = self._core.make_basis(basis,n,Np=Nup,count=Np_list)
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

		# 	self._Ns = self._core.make_basis(basis,n,Np=Nup)
		# 	if self._Ns < 0:
		# 			raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

		# 	basis,ind = _np.unique(basis,return_index=True)
		# 	if self.Ns != basis.shape[0]:
		# 		basis = basis[1:]
		# 		ind = ind[1:]
				
		# 	self._basis = basis[::-1].copy()
		# 	self._n = n[ind[::-1]].copy()

		if count_particles and (Nup is not None):
			Np_list = _np.zeros_like(basis,dtype=_np.uint8)
			Ns = self._core.make_basis(basis,n,Np=Nup,count=Np_list)
		else:
			Np_list = None
			Ns = self._core.make_basis(basis,n,Np=Nup)

		if Ns < 0:
				raise ValueError("estimate for size of reduced Hilbert-space is too low, please double check that transformation mappings are correct or use 'Ns_block_est' argument to give an upper bound of the block size.")

		if type(Nup) is int or Nup is None:
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
		self._allowed_ops=set(["I","z","+","-"])
		self._reduce_n_dtype()

