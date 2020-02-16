# cython: language_level=2
# distutils: language=c++
import cython
from scipy.special import comb
import scipy.sparse as _sp
cimport numpy as _np
import numpy as _np

include "source/general_basis_core.pyx"

cdef extern from "glibc_fix.h":
    pass

# specialized code 
cdef extern from "higher_spin_basis_core.h" namespace "basis_general":
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



cdef class higher_spin_basis_core_wrap(general_basis_core_wrap):
    def __cinit__(self,object dtype,object N,object sps,int[:,::1] maps, int[:] pers, int[:] qs):

        self._N = N
        self._nt = pers.shape[0]
        self._sps = sps
        self._Ns_full = (sps**N)

        if self._sps < 3:
            if sps == 2:
                raise ValueError("for sps==2 use hcb_core for internals.")

            raise ValueError("must have sps > 2")

        if dtype == uint32:
            if self._nt>0:
                self._basis_core = <void *> new higher_spin_basis_core[uint32_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new higher_spin_basis_core[uint32_t](N,sps)
        elif dtype == uint64:
            if self._nt>0:
                self._basis_core = <void *> new higher_spin_basis_core[uint64_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new higher_spin_basis_core[uint64_t](N,sps)
        elif dtype == uint256:
            if self._nt>0:
                self._basis_core = <void *> new higher_spin_basis_core[uint256_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new higher_spin_basis_core[uint256_t](N,sps)
        elif dtype == uint1024:
            if self._nt>0:
                self._basis_core = <void *> new higher_spin_basis_core[uint1024_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new higher_spin_basis_core[uint1024_t](N,sps)
        elif dtype == uint4096:
            if self._nt>0:
                self._basis_core = <void *> new higher_spin_basis_core[uint4096_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new higher_spin_basis_core[uint4096_t](N,sps)
        elif dtype == uint16384:
            if self._nt>0:
                self._basis_core = <void *> new higher_spin_basis_core[uint16384_t](N,sps,self._nt,&maps[0,0],&pers[0],&qs[0])
            else:
                self._basis_core = <void *> new higher_spin_basis_core[uint16384_t](N,sps)
        else:
            raise ValueError("general basis supports integer sizes <= 64 bits.")

    def get_s0_pcon(self,object Np):
        sps = <object>(self._sps)
        l = Np//(sps-1)
        s  = sum((sps-1) * sps**i for i in range(l))
        s += (Np%(sps-1)) * sps**l
        return s

    def get_Ns_pcon(self,object Np):
        return H_dim(Np,self._N,self._sps-1)

    @cython.boundscheck(False)
    def make_basis(self,_np.ndarray basis,norm_type[:] n,object Np=None,uint8_t[:] count=None):
        cdef npy_intp Ns_1 = 0
        cdef npy_intp Ns_2 = 0
        cdef npy_intp Ns_3 = 0
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
