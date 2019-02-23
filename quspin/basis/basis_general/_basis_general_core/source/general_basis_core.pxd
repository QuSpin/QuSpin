# distutils: language=c++
from numpy cimport npy_intp,int8_t,int16_t,int32_t,int64_t,uint8_t,uint16_t,uint32_t,uint64_t
from numpy cimport float32_t,float64_t,complex64_t,complex128_t
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.set cimport set


cdef extern from "general_basis_core.h" namespace "basis_general":
    cdef cppclass general_basis_core[I]:
        const int pers[]
        const int qs[]
        void map_state(I[],npy_intp,int,signed char[]) nogil

cdef extern from "make_general_basis.h" namespace "basis_general":
    npy_intp make_basis[I,J](general_basis_core[I]*,npy_intp,npy_intp,I[], J[]) nogil
    npy_intp make_basis_pcon[I,J](general_basis_core[I]*,npy_intp,npy_intp,I,I[], J[]) nogil

cdef extern from "general_basis_op.h" namespace "basis_general":
    int general_op[I,J,K,T](general_basis_core[I] *B,const int,const char[], const int[],
                          const double complex, const bool, const npy_intp, const I[], const J[], K[], K[], T[]) nogil
    int general_inplace_op[I,J,K](general_basis_core[I] *B,const bool,const bool,const int,const char[], const int[],
                          const double complex, const bool, const npy_intp,const npy_intp, const I[], const J[],const K[], K[]) nogil
    int general_op_bra_ket[I,T](general_basis_core[I] *B,const int,const char[], const int[],
                          const double complex, const npy_intp, const I[], I[], T[]) nogil
    int general_op_bra_ket_pcon[I,T](general_basis_core[I] *B,const int,const char[], const int[],
                          const double complex, const npy_intp, const set[vector[int]], const I[], I[], T[]) nogil

cdef extern from "general_basis_get_vec.h" namespace "basis_general":
    bool get_vec_general_dense[I,J,T](general_basis_core[I] *B,const I[],const J[],const npy_intp,
                                    const npy_intp,const npy_intp,const I[],const T[],T[]) nogil

cdef extern from "misc.h" namespace "basis_general":
    K binary_search[K,I](const K,const I[],const I) nogil
    void map_state_wrapper(void*,void*,npy_intp,int,uint8_t*) nogil


cdef extern from "general_basis_rep.h" namespace "basis_general":
    void general_representative[I](general_basis_core[I] *B, const I[], I[], int[], int8_t[], const npy_intp) nogil

    int general_normalization[I,J](general_basis_core[I] *B, I[], J[], const npy_intp) nogil

cdef extern from "bits_info.h" namespace "basis_general":
    cdef cppclass uint256_t:
        uint256_t operator&(int)
        uint256_t operator>>(int)
        uint256_t operator<<(int)
        uint256_t operator^(uint256_t)
        bool operator==(uint256_t)
        bool operator!=(uint256_t)
        bool operator!=(int)

    cdef cppclass uint1024_t:
        uint1024_t operator&(int)
        uint1024_t operator>>(int)
        uint1024_t operator<<(int)
        uint1024_t operator^(uint1024_t)
        uint1024_t& operator[](int)
        bool operator==(uint1024_t)
        bool operator!=(uint1024_t)
        bool operator!=(int)

    cdef cppclass uint4096_t:
        uint4096_t operator&(int)
        uint4096_t operator>>(int)
        uint4096_t operator<<(int)
        uint4096_t operator^(uint4096_t)
        bool operator==(uint4096_t)
        bool operator!=(uint4096_t)
        bool operator!=(int)

    cdef cppclass uint16384_t:
        uint16384_t operator&(int)
        uint16384_t operator>>(int)
        uint16384_t operator<<(int)
        uint16384_t operator^(uint16384_t)
        bool operator==(uint16384_t)
        bool operator!=(uint16384_t)
        bool operator!=(int)

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

ctypedef fused npy_uint:
    uint32_t
    uint64_t    

ctypedef fused state_type:
    uint32_t
    uint64_t
    uint256_t
    uint1024_t
    uint4096_t
    uint16384_t



cdef inline state_type python_to_basis(object python_val, state_type val):
    cdef int i = 0
    val = <state_type>(0)

    while(python_val!=0):
        val = val ^ ((<state_type>(<int>(python_val&1))) << i)
        i += 1
        python_val = python_val >> 1

    return val

cdef inline object basis_to_python(state_type *ptr):
    cdef state_type val = ptr[0]
    cdef object python_val = 0
    cdef object i = 0

    while(val!=0):
        python_val ^= (<object>(<int>(val&1))) << i
        i += 1
        val = val >> 1

    return python_val


cdef extern from "general_basis_bitops.h" namespace "basis_general":
    void bitwise_and[I](const I[], const I[], bool[], I[], const npy_intp) nogil

