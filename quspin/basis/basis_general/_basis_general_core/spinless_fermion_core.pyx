import cython
from scipy.misc import comb
from general_basis_core cimport dtype,index_type,norm_type
import scipy.sparse as _sp

include "source/general_basis_core.pyx"

cdef extern from "glibc_fix.h":
	pass

# specialized code 
cdef extern from "spinless_fermion_basis_core.h":
	cdef cppclass spinless_fermion_basis_core[I](general_basis_core[I]):
		spinless_fermion_basis_core(const int,const int,const int[],const int[],const int[])
		spinless_fermion_basis_core(const int) 


cdef class spinless_fermion_basis_core_wrap_32(general_basis_core_wrap_32):
	def __cinit__(self,object N,int[:,::1] maps, int[:] pers, int[:] qs):

		if N > 32:
			raise ValueError("for 32-bit code N must be <= 32.")
		self._N = N
		self._nt = pers.shape[0]
		self._sps = 2

		if self._nt>0:
			self._basis_core = new spinless_fermion_basis_core[uint32_t](N,self._nt,&maps[0,0],&pers[0],&qs[0])
		else:
			self._basis_core = new spinless_fermion_basis_core[uint32_t](N)


	@cython.boundscheck(False)
	def make_basis(self,uint32_t[:] basis,norm_type[:] n,object Np=None,uint8_t[:] count=None):
		cdef int Ns_1 = 0
		cdef int Ns_2 = 0
		cdef int Ns_3 = 0
		cdef uint8_t np = 0
		cdef npy_intp i = 0
		cdef mem_MAX = basis.shape[0]

		if Np is None:
			Ns_2 = self.make_basis_full[norm_type](basis,n)
		elif type(Np) is int:
			Ns_2 = self.make_basis_pcon[norm_type](Np,basis,n)
		else:
			Np_iter = iter(Np)
			if count is None:
				for np in Np_iter:
					Ns_1 = self.make_basis_pcon[norm_type](np,basis[Ns_2:],n[Ns_2:])
					if Ns_1 < 0:
						return Ns_1
					else:
						Ns_2 += Ns_1

					if Ns_2 > mem_MAX:
						return -1
			else:

				for np in Np_iter:
					Ns_1 = self.make_basis_pcon[norm_type](np,basis[Ns_2:],n[Ns_2:])
					if Ns_1 < 0:
						return Ns_1
					else:
						Ns_3 = Ns_2 + Ns_1
						for i in range(Ns_2,Ns_3,1):
							count[i] = np

						Ns_2 = Ns_3

					if Ns_2 > mem_MAX:
						return -1
		return Ns_2

	@cython.boundscheck(False)
	cdef npy_intp make_basis_full(self,uint32_t[:] basis,norm_type[:] n):
		cdef npy_intp Ns = (1ull<<self._N)
		cdef npy_intp mem_MAX = basis.shape[0]
		with nogil:
			Ns = make_basis(self._basis_core,Ns,mem_MAX,&basis[0],&n[0])

		return Ns

	@cython.boundscheck(False)
	cdef npy_intp make_basis_pcon(self,int Np,uint32_t[:] basis,norm_type[:] n):
		cdef npy_intp Ns = comb(self._N,Np,exact=True)
		cdef npy_intp mem_MAX = basis.shape[0]
		cdef uint32_t s  = sum(1<<i for i in range(Np))
		with nogil:
			Ns =  make_basis_pcon(self._basis_core,Ns,mem_MAX,s,&basis[0],&n[0])

		return Ns


cdef class spinless_fermion_basis_core_wrap_64(general_basis_core_wrap_64):
	def __cinit__(self,object N,int[:,::1] maps, int[:] pers, int[:] qs):
		if N > 64:
			raise ValueError("for 64-bit code N must be <= 64.")
		self._N = N
		self._nt = pers.shape[0]
		self._sps = 2
		if self._nt>0:
			self._basis_core = new spinless_fermion_basis_core[uint64_t](N,self._nt,&maps[0,0],&pers[0],&qs[0])
		else:
			self._basis_core = new spinless_fermion_basis_core[uint64_t](N)


	@cython.boundscheck(False)
	def make_basis(self,uint64_t[:] basis,norm_type[:] n,object Np=None,uint8_t[:] count=None):
		cdef int Ns_1 = 0
		cdef int Ns_2 = 0
		cdef int Ns_3 = 0
		cdef uint8_t np = 0
		cdef npy_intp i = 0
		cdef mem_MAX = basis.shape[0]

		if Np is None:
			Ns_2 = self.make_basis_full[norm_type](basis,n)
		elif type(Np) is int:
			Ns_2 = self.make_basis_pcon[norm_type](Np,basis,n)
		else:
			Np_iter = iter(Np)
			if count is None:
				for np in Np_iter:
					Ns_1 = self.make_basis_pcon[norm_type](np,basis[Ns_2:],n[Ns_2:])
					if Ns_1 < 0:
						return Ns_1
					else:
						Ns_2 += Ns_1

					if Ns_2 > mem_MAX:
						return -1
			else:

				for np in Np_iter:
					Ns_1 = self.make_basis_pcon[norm_type](np,basis[Ns_2:],n[Ns_2:])
					if Ns_1 < 0:
						return Ns_1
					else:
						Ns_3 = Ns_2 + Ns_1
						for i in range(Ns_2,Ns_3,1):
							count[i] = np

						Ns_2 = Ns_3

					if Ns_2 > mem_MAX:
						return -1
		return Ns_2



	@cython.boundscheck(False)
	cdef npy_intp make_basis_full(self,uint64_t[:] basis,norm_type[:] n):
		cdef npy_intp Ns = (1ull<<self._N)
		cdef npy_intp mem_MAX = basis.shape[0]
		with nogil:
			Ns = make_basis(self._basis_core,Ns,mem_MAX,&basis[0],&n[0])

		return Ns

	@cython.boundscheck(False)
	cdef npy_intp make_basis_pcon(self,int Np,uint64_t[:] basis,norm_type[:] n):
		cdef npy_intp Ns = comb(self._N,Np,exact=True)
		cdef npy_intp mem_MAX = basis.shape[0]
		cdef uint64_t s = sum(1<<i for i in range(Np))
		with nogil:
			Ns =  make_basis_pcon(self._basis_core,Ns,mem_MAX,s,&basis[0],&n[0])

		return Ns

