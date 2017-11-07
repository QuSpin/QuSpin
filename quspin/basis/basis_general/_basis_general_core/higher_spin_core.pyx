import cython
from scipy.misc import comb
from general_basis_core cimport dtype,index_type,norm_type
import scipy.sparse as _sp

include "source/general_basis_core.pyx"

cdef extern from "glibc_fix.h":
	pass

# specialized code 
cdef extern from "higher_spin_basis_core.h":
	cdef cppclass higher_spin_basis_core[I](general_basis_core[I]):
		higher_spin_basis_core(const int,const int,const int,const int[],const int[],const int[])
		higher_spin_basis_core(const int,const int) 

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

cdef class higher_spin_basis_core_wrap_32(general_basis_core_wrap_32):
	def __cinit__(self,object N,int sps,int[:,::1] maps, int[:] pers, int[:] qs):

		self._N = N
		self._nt = pers.shape[0]
		self._sps = sps

		if self._nt>0:
			self._basis_core = new higher_spin_basis_core[uint32_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
		else:
			self._basis_core = new higher_spin_basis_core[uint32_t](N,sps)

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
		cdef npy_intp Ns = (self._sps**self._N)
		cdef npy_intp mem_MAX = basis.shape[0]
		with nogil:
			Ns = make_basis(self._basis_core,Ns,mem_MAX,&basis[0],&n[0])

		return Ns

	@cython.boundscheck(False)
	cdef npy_intp make_basis_pcon(self,int Np,uint32_t[:] basis,norm_type[:] n):
		cdef npy_intp Ns = H_dim(Np,self._N,self._sps-1)
		cdef npy_intp mem_MAX = basis.shape[0]
		cdef int l = Np/(self._sps-1)
		cdef uint32_t s  = sum((self._sps-1)*self._sps**i for i in range(l))
		s += (Np%(self._sps-1))*self._sps**l

		with nogil:
			Ns =  make_basis_pcon(self._basis_core,Ns,mem_MAX,s,&basis[0],&n[0])

		return Ns


cdef class higher_spin_basis_core_wrap_64(general_basis_core_wrap_64):
	def __cinit__(self,object N,int sps,int[:,::1] maps, int[:] pers, int[:] qs):

		self._N = N
		self._nt = pers.shape[0]
		self._sps = sps

		if self._nt>0:
			self._basis_core = new higher_spin_basis_core[uint64_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
		else:
			self._basis_core = new higher_spin_basis_core[uint64_t](N,sps)

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
		cdef npy_intp Ns = (self._sps**self._N)
		cdef npy_intp mem_MAX = basis.shape[0]
		with nogil:
			Ns = make_basis(self._basis_core,Ns,mem_MAX,&basis[0],&n[0])

		return Ns

	@cython.boundscheck(False)
	cdef npy_intp make_basis_pcon(self,int Np,uint64_t[:] basis,norm_type[:] n):
		cdef npy_intp Ns = H_dim(Np,self._N,self._sps-1)
		cdef npy_intp mem_MAX = basis.shape[0]
		cdef int l = Np/(self._sps-1)
		cdef uint64_t s  = sum((self._sps-1)*self._sps**i for i in range(l))
		s += (Np%(self._sps-1))*self._sps**l

		with nogil:
			Ns =  make_basis_pcon(self._basis_core,Ns,mem_MAX,s,&basis[0],&n[0])

		return Ns


