from ._basis_general_core import spinful_fermion_basis_core_wrap_32,spinful_fermion_basis_core_wrap_64
from ._basis_general_core import spinless_fermion_basis_core_wrap_32,spinless_fermion_basis_core_wrap_64
from .base_general import basis_general
from ..base import MAXPRINT
import numpy as _np
from scipy.misc import comb


# general basis for hardcore bosons/spin-1/2
class spinful_fermion_basis_general(basis_general):
	def __init__(self,N,Nf=None,_Np=None,**kwargs):
		# Nf = [(Nup,Ndown),...]
		# Nup is left side of basis sites 0 - N-1
		# Ndown is right side of basis sites N - 2*N-1

		basis_general.__init__(self,2*N,**kwargs)
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
			Ns = max(int(float(Ns)/_np.multiply.reduce(self._pers))*2,1000)

		if N<=16:
			basis = _np.zeros(Ns,dtype=_np.uint32)
			n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = spinful_fermion_basis_core_wrap_32(N,self._maps,self._pers,self._qs)
		elif N<=32:
			basis = _np.zeros(Ns,dtype=_np.uint64)
			n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = spinful_fermion_basis_core_wrap_64(N,self._maps,self._pers,self._qs)
		else:
			raise ValueError("system size N must be <=32.")

		self._sps=2
		if count_particles and (Nf is not None):
			Np_list = _np.zeros_like(basis,dtype=_np.uint8)
			self._Ns = self._core.make_basis(basis,n,Np=Nf,count=Np_list)
			if self.Ns != basis.shape[0]:
				basis = basis[1:]
				ind = ind[1:]
			self._basis = basis[::-1].copy()
			self._n = n[ind[::-1]].copy()
			self._Np_list = Np_list[ind[::-1]].copy()
		else:
			self._Ns = self._core.make_basis(basis,n,Np=Nf)
			basis,ind = _np.unique(basis,return_index=True)
			if self.Ns != basis.shape[0]:
				basis = basis[1:]
				ind = ind[1:]
			self._basis = basis[::-1].copy()
			self._n = n[ind[::-1]].copy()

		self._N = N
		self._index_type = _np.min_scalar_type(-self._Ns)
		self._allowed_ops=set(["I","n","+","-"])

		self._check_symm = None

	def _get__str__(self):
		def get_state(b):
			bits_left = ((b>>(2*self.N-i-1))&1 for i in range(self.N))
			state_left = "|"+(" ".join(("{:1d}").format(bit) for bit in bits_left))+">"
			bits_right = ((b>>(self.N-i-1))&1 for i in range(self.N))
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



# general basis for hardcore bosons/spin-1/2
class spinless_fermion_basis_general(basis_general):
	def __init__(self,N,Nf=None,_Np=None,**kwargs):
		basis_general.__init__(self,N,**kwargs)
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
			Ns = max(int(float(Ns)/_np.multiply.reduce(self._pers))*2,1000)


		if N<=32:
			basis = _np.zeros(Ns,dtype=_np.uint32)
			n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = spinless_fermion_basis_core_wrap_32(N,self._maps,self._pers,self._qs)
		elif N<=64:
			basis = _np.zeros(Ns,dtype=_np.uint64)
			n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = spinless_fermion_basis_core_wrap_64(N,self._maps,self._pers,self._qs)
		else:
			raise ValueError("system size N must be <=64.")

		self._sps=2
		if count_particles and (Nf is not None):
			Np_list = _np.zeros_like(basis,dtype=_np.uint8)
			self._Ns = self._core.make_basis(basis,n,Np=Nf,count=Np_list)
			basis,ind = _np.unique(basis,return_index=True)
			if self.Ns != basis.shape[0]:
				basis = basis[1:]
				ind = ind[1:]

			self._basis = basis[::-1].copy()
			self._n = n[ind[::-1]].copy()
			self._Np_list = Np_list[ind[::-1]].copy()
		else:
			self._Ns = self._core.make_basis(basis,n,Np=Nf)
			basis,ind = _np.unique(basis,return_index=True)
			if self.Ns != basis.shape[0]:
				basis = basis[1:]
				ind = ind[1:]
				
			self._basis = basis[::-1].copy()
			self._n = n[ind[::-1]].copy()

		self._N = N
		self._index_type = _np.min_scalar_type(-self._Ns)
		self._allowed_ops=set(["I","n","+","-"])

		self._check_symm = None






