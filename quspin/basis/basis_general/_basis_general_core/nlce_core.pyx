# cython: language_level=2
# distutils: language=c++
import cython
cimport numpy as _npc
import numpy as _np
import scipy.sparse as _sp
from general_basis_types cimport uint64_t,uint128_t,uint256_t,uint512_t,uint1024_t,npy_intp




# specialized code 
cdef extern from "nlce_basis_core.h" namespace "nlce":
    cdef cppclass nlce_basis_core[I]:
        nlce_basis_core(const int,const int,const int,const int[],
            const int,const int,const int[],const int[],const int[])
        void clusters_calc()
        void calc_subclusters()
        void get_Y_matrix_dims(int&,int&)
        void get_Y_matrix(npy_intp[],npy_intp[],npy_intp[])
        void cluster_copy(I[],int[],int[])


cdef class nlce_core_wrap:
    cdef int _Ncl
    cdef int _N
    cdef void * _nlce_core

    def __cinit__(self,int Ncl
                      ,int nt_point
                      ,int nt_tran
                      ,int[:,::1] maps
                      ,int[::1] pers
                      ,int[::1] qs
                      ,int[:,::1] nn_list):
        
        cdef int N = maps.shape[1]
        cdef int Nnn = nn_list.shape[1]
        self._Ncl = Ncl
        self._N = N

        if N <= 64:
            self._nlce_core = <void*> new nlce_basis_core[uint64_t](Ncl,N,Nnn,&nn_list[0,0],nt_point,nt_tran,&maps[0,0],&pers[0],&qs[0])        
        elif N <= 128:
            self._nlce_core = <void*> new nlce_basis_core[uint128_t](Ncl,N,Nnn,&nn_list[0,0],nt_point,nt_tran,&maps[0,0],&pers[0],&qs[0])
        elif N <= 256:
            self._nlce_core = <void*> new nlce_basis_core[uint256_t](Ncl,N,Nnn,&nn_list[0,0],nt_point,nt_tran,&maps[0,0],&pers[0],&qs[0])
        elif N <= 512:
            self._nlce_core = <void*> new nlce_basis_core[uint512_t](Ncl,N,Nnn,&nn_list[0,0],nt_point,nt_tran,&maps[0,0],&pers[0],&qs[0])
        elif N <= 1024:
            self._nlce_core = <void*> new nlce_basis_core[uint1024_t](Ncl,N,Nnn,&nn_list[0,0],nt_point,nt_tran,&maps[0,0],&pers[0],&qs[0])
        else:
            raise ValueError

    def calc_clusters(self):
        cdef void * nlce_core = self._nlce_core
        cdef nlce_basis_core[uint64_t]* nlce_64 = <nlce_basis_core[uint64_t]*>nlce_core
        cdef nlce_basis_core[uint128_t]* nlce_128 = <nlce_basis_core[uint128_t]*>nlce_core
        cdef nlce_basis_core[uint256_t]* nlce_256 = <nlce_basis_core[uint256_t]*>nlce_core
        cdef nlce_basis_core[uint512_t]* nlce_512 = <nlce_basis_core[uint512_t]*>nlce_core
        cdef nlce_basis_core[uint1024_t]* nlce_1024 = <nlce_basis_core[uint1024_t]*>nlce_core

        cdef _npc.ndarray L
        cdef _npc.ndarray clusters
        cdef _npc.ndarray data
        cdef _npc.ndarray indices
        cdef _npc.ndarray indptr

        cdef void * L_ptr
        cdef void * clusters_ptr
        cdef void * data_ptr
        cdef void * indices_ptr
        cdef void * indptr_ptr

        cdef npy_intp row=0
        cdef npy_intp nnz=0

        if self._N <= 64:
            nlce_64.clusters_calc()
            nlce_64.calc_subclusters()
            nlce_64.get_Y_matrix_dims(row,nnz)
            dtype = _np.uint64

        elif self._N <= 128:
            nlce_128.clusters_calc()
            nlce_128.calc_subclusters()
            nlce_128.get_Y_matrix_dims(row,nnz)
            dtype = _np.dtype((_np.void,sizeof(uint128_t)))

        elif self._N <= 256:
            nlce_256.clusters_calc()
            nlce_256.calc_subclusters()
            nlce_256.get_Y_matrix_dims(row,nnz)
            dtype = _np.dtype((_np.void,sizeof(uint256_t)))

        elif self._N <= 512:
            nlce_512.clusters_calc()
            nlce_512.calc_subclusters()
            nlce_512.get_Y_matrix_dims(row,nnz)
            dtype = _np.dtype((_np.void,sizeof(uint512_t)))

        elif self._N <= 1024:
            nlce_1024.clusters_calc()
            nlce_1024.calc_subclusters()
            nlce_1024.get_Y_matrix_dims(row,nnz)
            dtype = _np.dtype((_np.void,sizeof(uint1024_t)))


        L        = _np.zeros(row  ,dtype=_np.int32)
        ncl      = _np.zeros(row  ,dtype=_np.int32)
        clusters = _np.zeros(row  ,dtype=dtype)
        data     = _np.zeros(nnz  ,dtype=_np.intp)
        indptr   = _np.zeros(row+1,dtype=_np.intp)
        indices  = _np.zeros(nnz  ,dtype=_np.intp)

        L_ptr        = _npc.PyArray_DATA(L)
        ncl_ptr      = _npc.PyArray_DATA(ncl)
        clusters_ptr = _npc.PyArray_DATA(clusters)
        data_ptr     = _npc.PyArray_DATA(data)
        indices_ptr  = _npc.PyArray_DATA(indices)
        indptr_ptr   = _npc.PyArray_DATA(indptr)


        if self._N <= 64:
            nlce_64.get_Y_matrix(<npy_intp*>data_ptr,<npy_intp*>indices_ptr,<npy_intp*>indptr_ptr)
            nlce_64.cluster_copy(<uint64_t*>clusters_ptr,<int*>L_ptr,<int*>ncl_ptr)

        elif self._N <= 128:
            nlce_128.get_Y_matrix(<npy_intp*>data_ptr,<npy_intp*>indices_ptr,<npy_intp*>indptr_ptr)
            nlce_128.cluster_copy(<uint128_t*>clusters_ptr,<int*>L_ptr,<int*>ncl_ptr)

        elif self._N <= 256:
            nlce_256.get_Y_matrix(<npy_intp*>data_ptr,<npy_intp*>indices_ptr,<npy_intp*>indptr_ptr)
            nlce_256.cluster_copy(<uint256_t*>clusters_ptr,<int*>L_ptr,<int*>ncl_ptr)

        elif self._N <= 512:
            nlce_512.get_Y_matrix(<npy_intp*>data_ptr,<npy_intp*>indices_ptr,<npy_intp*>indptr_ptr)
            nlce_512.cluster_copy(<uint512_t*>clusters_ptr,<int*>L_ptr,<int*>ncl_ptr)

        elif self._N <= 1024:
            nlce_1024.get_Y_matrix(<npy_intp*>data_ptr,<npy_intp*>indices_ptr,<npy_intp*>indptr_ptr)
            nlce_1024.cluster_copy(<uint1024_t*>clusters_ptr,<int*>L_ptr,<int*>ncl_ptr)

        Y = _sp.csr_matrix((data,indices,indptr),shape=(row,row))
        return clusters,L,ncl,Y



