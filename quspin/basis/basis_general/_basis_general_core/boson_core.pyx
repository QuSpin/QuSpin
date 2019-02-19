# cython: language_level=2
# distutils: language=c++
import cython
from scipy.misc import comb
from general_basis_core cimport dtype,index_type,norm_type
import scipy.sparse as _sp
cimport numpy as _np
import numpy as _np

include "source/general_basis_core.pyx"

cdef extern from "glibc_fix.h":
    pass

# specialized code 
cdef extern from "boson_basis_core.h":
    cdef cppclass boson_basis_core[I](general_basis_core[I]):
        boson_basis_core(const int,const int,const int,const int[],const int[],const int[])
        boson_basis_core(const int,const int) 


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


cdef class boson_basis_core_wrap(general_basis_core_wrap):
    def __cinit__(self,object dtype,object N,int sps,int[:,::1] maps, int[:] pers, int[:] qs):

        self._N = N
        self._nt = pers.shape[0]
        self._sps = sps
        self._Ns_full = (sps**N)

        if self._sps < 2:
            raise ValueError("must have sps > 2")

        if dtype == uint32:
            if self._nt>0:
                self._basis_core = <void *> new boson_basis_core[uint32_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new boson_basis_core[uint32_t](N,sps)
        elif dtype == uint64:
            if self._nt>0:
                self._basis_core = <void *> new boson_basis_core[uint64_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new boson_basis_core[uint64_t](N,sps)
        elif dtype == uint128:
            if self._nt>0:
                self._basis_core = <void *> new boson_basis_core[uint128_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new boson_basis_core[uint128_t](N,sps)
        else:
            raise ValueError("general basis supports integer sizes <= 64 bits.")

    def get_s0_pcon(self,int Np):
        l = Np//(self._sps-1)
        s  = sum((self._sps-1)*self._sps**i for i in range(l))
        s += (Np%(self._sps-1))*self._sps**l
        return s

    def get_Ns_pcon(self,int Np):
        return H_dim(Np,self._N,self._sps-1)

    @cython.boundscheck(False)
    def make_basis(self,_np.ndarray basis,norm_type[:] n,object Np=None,uint8_t[:] count=None):
        cdef int Ns_1 = 0
        cdef int Ns_2 = 0
        cdef int Ns_3 = 0
        cdef uint8_t np = 0
        cdef npy_intp i = 0
        cdef mem_MAX = basis.shape[0]

        if Np is None:
            Ns_2 = general_basis_core_wrap._make_basis_full(self,basis,n)
        elif type(Np) is int:
            Ns_2 = general_basis_core_wrap._make_basis_pcon(self,Np,basis,n)
        else:
            Np_iter = iter(Np)
            if count is None:
                for np in Np_iter:
                    Ns_1 = general_basis_core_wrap._make_basis_pcon(self,np,basis[Ns_2:],n[Ns_2:])
                    if Ns_1 < 0:
                        return Ns_1
                    else:
                        Ns_2 += Ns_1

                    if Ns_2 > mem_MAX:
                        return -1
            else:

                for np in Np_iter:
                    Ns_1 = general_basis_core_wrap._make_basis_pcon(self,np,basis[Ns_2:],n[Ns_2:])
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

