from numpy cimport npy_intp,int8_t,int16_t,int32_t,int64_t,uint8_t,uint16_t,uint32_t,uint64_t
from numpy cimport float32_t,float64_t,complex64_t,complex128_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set


cdef extern from "general_basis_core.h":
    cdef cppclass general_basis_core[I]:
        const int pers[]
        const int qs[]
        void map_state(I[],npy_intp,int,signed char[]) nogil

cdef extern from "make_general_basis.h":
    npy_intp make_basis[I,J](general_basis_core[I]*,npy_intp,npy_intp,I[], J[]) nogil
    npy_intp make_basis_pcon[I,J](general_basis_core[I]*,npy_intp,npy_intp,I,I[], J[]) nogil

cdef extern from "general_basis_op.h":
    int general_op[I,J,K,T](general_basis_core[I] *B,const int,const char[], const int[],
                          const double complex, const bool, const npy_intp, const I[], const J[], K[], K[], T[]) nogil
    int general_inplace_op[I,J,K](general_basis_core[I] *B,const bool,const bool,const int,const char[], const int[],
                          const double complex, const bool, const npy_intp,const npy_intp, const I[], const J[],const K[], K[]) nogil
    int general_op_bra_ket[I,T](general_basis_core[I] *B,const int,const char[], const int[],
                          const double complex, const npy_intp, const I[], I[], T[]) nogil
    int general_op_bra_ket_pcon[I,T](general_basis_core[I] *B,const int,const char[], const int[],
                          const double complex, const npy_intp, const int, const set[vector[int]], const I[], I[], T[]) nogil

cdef extern from "general_basis_get_vec.h":
    bool get_vec_general_dense[I,J,T](general_basis_core[I] *B,const I[],const J[],const npy_intp,
                                    const npy_intp,const npy_intp,const I[],const T[],T[]) nogil

cdef extern from "misc.h":
    K binary_search[K,I](const K,const I[],const I) nogil
    void map_state_wrapper(void*,void*,npy_intp,int,uint8_t*) nogil


cdef extern from "general_basis_rep.h":
    void general_representative[I](general_basis_core[I] *B, const I[], I[], int[], int8_t[], const npy_intp) nogil

    int general_normalization[I,J](general_basis_core[I] *B, I[], J[], const npy_intp) nogil

# cdef extern from "boost/multiprecision/cpp_int.hpp" namespace "boost::multiprecision": 
#     ctypedef uint128_t

ctypedef fused index_type:
    int32_t
    int64_t

ctypedef fused dtype:
    float32_t
    float64_t
    complex64_t
    complex128_t

ctypedef fused norm_type:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

ctypedef fused state_type:
    uint32_t
    uint64_t

