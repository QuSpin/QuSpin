import cython
from scipy.misc import comb
from general_basis_core cimport dtype,index_type
import scipy.sparse as _sp

include "source/general_basis_core.pyx"

# specialized code 
cdef extern from "spinful_fermion_basis_core.h":
	cdef cppclass spinful_fermion_basis_core[I](general_basis_core[I]):
		spinful_fermion_basis_core(const int,const int,const int[],const int[],const int[])
		spinful_fermion_basis_core(const int) 


cdef class spinful_fermion_basis_core_wrap_32(general_basis_core_wrap_32):
	def __cinit__(self,object N,int[:,::1] maps, int[:] pers, int[:] qs):

		if N > 16:
			raise ValueError("for 32-bit code N must be <= 16.")
		self._N = N
		self._nt = pers.shape[0]
		self._sps = 2

		if self._nt>0:
			self._basis_core = new spinful_fermion_basis_core[uint32_t](N,self._nt,&maps[0,0],&pers[0],&qs[0])
		else:
			self._basis_core = new spinful_fermion_basis_core[uint32_t](N)

	@cython.boundscheck(False)
	def make_basis(self,uint32_t[:] basis,uint16_t[:] n,object Np=None,uint8_t[:] count=None):
		cdef int Ns_1 = 0
		cdef int Ns_2 = 0
		cdef int np_1,np_2
		cdef npy_intp i = 0


		if Np is None:
			Ns_1 = self.make_basis_full(basis,n)
		elif type(Np) is tuple:
			np_1,np_2 = Np
			Ns_1 = self.make_basis_pcon(np_1,np_2,basis,n)
		elif type(Np) is list:
			if count is None:
				for np_1,np_2 in Np:
					Ns_1 += self.make_basis_pcon(np_1,np_2,basis[Ns_1:],n[Ns_1:])
			else:
				for np_1,np_2 in Np:
					Ns_2 = Ns_1 + self.make_basis_pcon(np_1,np_2,basis[Ns_1:],n[Ns_1:])
					for i in range(Ns_1,Ns_2,1):
						count[i] = np_1+np_2

					Ns_1 = Ns_2

		return Ns_1


	@cython.boundscheck(False)
	cdef npy_intp make_basis_full(self,uint32_t[:] basis,uint16_t[:] n):
		cdef npy_intp Ns = (1<<self._N)**2
		with nogil:
			Ns = make_basis(self._basis_core,Ns,&basis[0],&n[0])

		return Ns

	@cython.boundscheck(False)
	cdef npy_intp make_basis_pcon(self,int Np_1,int Np_2,uint32_t[:] basis,uint16_t[:] n):
		cdef npy_intp Ns = comb(self._N,Np_1,exact=True)*comb(self._N,Np_2,exact=True)
		cdef uint32_t s = sum(1<<i for i in range(Np_2))
		s += (sum(1<<i for i in range(Np_1))) << self._N
		with nogil:
			Ns =  make_basis_pcon(self._basis_core,Ns,s,&basis[0],&n[0])

		return Ns


cdef class spinful_fermion_basis_core_wrap_64(general_basis_core_wrap_64):
	def __cinit__(self,object N,int[:,::1] maps, int[:] pers, int[:] qs):
		if N > 64:
			raise ValueError("for 64-bit code N must be <= 32.")
		self._N = N
		self._nt = pers.shape[0]
		self._sps = 2
		if self._nt>0:
			self._basis_core = new spinful_fermion_basis_core[uint64_t](N,self._nt,&maps[0,0],&pers[0],&qs[0])
		else:
			self._basis_core = new spinful_fermion_basis_core[uint64_t](N)


	def make_basis(self,uint64_t[:] basis,uint16_t[:] n,object Np=None,object count=None):
		cdef int Ns_1 = 0
		cdef int Ns_2 = 0
		cdef int np_1,np_2
		cdef npy_intp i = 0


		if Np is None:
			Ns_1 = self.make_basis_full(basis,n)
		elif type(Np) is tuple:
			np_1,np_2 = Np
			Ns_1 = self.make_basis_pcon(np_1,np_2,basis,n)
		elif type(Np) is list:
			if count is None:
				for np_1,np_2 in Np:
					Ns_1 += self.make_basis_pcon(np_1,np_2,basis[Ns_1:],n[Ns_1:])
			else:
				for np_1,np_2 in Np:
					Ns_2 = Ns_1 + self.make_basis_pcon(np_1,np_2,basis[Ns_1:],n[Ns_1:])
					for i in range(Ns_1,Ns_2,1):
						count[i] = np_1+np_2

					Ns_1 = Ns_2

		return Ns_1


	@cython.boundscheck(False)
	cdef npy_intp make_basis_full(self,uint64_t[:] basis,uint16_t[:] n):
		cdef npy_intp Ns = (1ull<<self._N)
		with nogil:
			Ns = make_basis(self._basis_core,Ns,&basis[0],&n[0])

		return Ns

	@cython.boundscheck(False)
	cdef npy_intp make_basis_pcon(self,int Np_1,int Np_2,uint64_t[:] basis,uint16_t[:] n):
		cdef npy_intp Ns = comb(self._N,Np_1,exact=True)*comb(self._N,Np_2,exact=True)
		cdef uint64_t s = sum(1<<i for i in range(Np_2))
		s += (sum(1<<i for i in range(Np_1))) << self._N
		with nogil:
			Ns =  make_basis_pcon(self._basis_core,Ns,s,&basis[0],&n[0])

		return Ns

