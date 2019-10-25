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
cdef extern from "spinful_fermion_basis_core.h" namespace "basis_general":
    cdef cppclass spinful_fermion_basis_core[I](general_basis_core[I]):
        spinful_fermion_basis_core(const int,const int,const int[],const int[],const int[],const bool)
        spinful_fermion_basis_core(const int,const bool) 

cdef class spinful_fermion_basis_core_wrap(general_basis_core_wrap):
    def __cinit__(self,object dtype,object N,int[:,::1] maps, int[:] pers, int[:] qs,bool dble_occ):

        self._N = N
        self._nt = pers.shape[0]
        self._sps = 2
        self._Ns_full = (<object>(1)<<2*N)

        if dtype == uint32:
            if self._nt>0:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint32_t](N,self._nt,&maps[0,0],&pers[0],&qs[0],dble_occ)
            else:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint32_t](N,dble_occ)
        elif dtype == uint64:
            if self._nt>0:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint64_t](N,self._nt,&maps[0,0],&pers[0],&qs[0],dble_occ)
            else:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint64_t](N,dble_occ)
        elif dtype == uint256:
            if self._nt>0:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint256_t](N,self._nt,&maps[0,0],&pers[0],&qs[0],dble_occ)
            else:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint256_t](N,dble_occ)
        elif dtype == uint1024:
            if self._nt>0:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint1024_t](N,self._nt,&maps[0,0],&pers[0],&qs[0],dble_occ)
            else:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint1024_t](N,dble_occ)
        elif dtype == uint4096:
            if self._nt>0:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint4096_t](N,self._nt,&maps[0,0],&pers[0],&qs[0],dble_occ)
            else:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint4096_t](N,dble_occ)
        elif dtype == uint16384:
            if self._nt>0:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint16384_t](N,self._nt,&maps[0,0],&pers[0],&qs[0],dble_occ)
            else:
                self._basis_core = <void *> new spinful_fermion_basis_core[uint16384_t](N,dble_occ)
        else:
            raise ValueError("general basis supports system sizes <= 16384.")

    def get_s0_pcon(self,object Np):
        s = sum(<object>(1)<<i for i in range(Np[1]))
        s += (sum(<object>(1)<<i for i in range(Np[0]))) << self._N
        return s

    def get_Ns_pcon(self,object Np):
        return comb(self._N,Np[0],exact=True)*comb(self._N,Np[1],exact=True)

    @cython.boundscheck(False)
    def make_basis(self,_np.ndarray basis,norm_type[:] n,object Np=None,uint8_t[:] count=None):
        cdef npy_intp Ns_1 = 0
        cdef npy_intp Ns_2 = 0
        cdef npy_intp Ns_3 = 0
        cdef npy_intp i = 0
        cdef mem_MAX = basis.shape[0]


        if Np is None:
            Ns_2 = general_basis_core_wrap._make_basis_full(self,basis,n)
        elif type(Np) is tuple:
            np_1,np_2 = Np
            Ns_2 = general_basis_core_wrap._make_basis_pcon(self,Np,basis,n)
        elif type(Np) is list:
            if count is None:
                for np in Np:
                    Ns_1 =general_basis_core_wrap._make_basis_pcon(self,np,basis[Ns_2:],n[Ns_2:])
                    if Ns_1 < 0:
                        return Ns_1
                    else:
                        Ns_2 += Ns_1

                    if Ns_2 > mem_MAX:
                        return -1
            else:
                for np in Np:
                    Ns_1 = general_basis_core_wrap._make_basis_pcon(self,np,basis[Ns_2:],n[Ns_2:])
                    if Ns_1 < 0:
                        return Ns_1
                    else:
                        Ns_3 = Ns_2 + Ns_1
                        for i in range(Ns_2,Ns_3,1):
                            count[i] = np[0]+np[1]

                        Ns_2 = Ns_3

                    if Ns_2 > mem_MAX:
                        return -1

        return Ns_2

