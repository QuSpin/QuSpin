# distutils: language=c++
from general_basis_types cimport *
from libcpp.vector cimport vector
from libcpp.set cimport set
from libcpp.utility cimport pair
from numpy cimport PyArray_Descr,PyArray_DESCR,ndarray
import numpy as _np

_np.import_array()

cdef extern from "general_basis_core.h" namespace "basis_general":
    cdef cppclass general_basis_core[I]:
        const int pers[]
        const int qs[]
        void map_state(I[],npy_intp,int,signed char[]) nogil

cdef extern from "make_general_basis.h" namespace "basis_general":
    npy_intp make_basis[I,J](general_basis_core[I]*,npy_intp,npy_intp,I[], J[]) nogil
    npy_intp make_basis_pcon[I,J](general_basis_core[I]*,npy_intp,npy_intp,I,I[], J[]) nogil
    int general_make_basis_blocks[I](general_basis_core[I] *B,const int,const npy_intp,const I[],npy_intp[],npy_intp[]) nogil

cdef extern from "general_basis_op.h" namespace "basis_general":
    pair[int,int] general_op[I,J,K,T](general_basis_core[I] *B,const int,const char[], const int[],
                                      const double complex, const bool, const npy_intp, const I[], const J[],
                                      const npy_intp[],const npy_intp[],const int, K[], K[], T[]) nogil

    int general_inplace_op_impl[I,J](general_basis_core[I] *B,const bool,const bool,const int,const char[], 
                          const int[],void*, const bool, const npy_intp,const npy_intp, const I[], const J[],
                          const npy_intp[],const npy_intp[],const int,PyArray_Descr*,void*,void*) nogil
    int general_op_bra_ket[I,T](general_basis_core[I] *B,const int,const char[], const int[],
                          const double complex, const npy_intp, const I[], I[], T[]) nogil
    int general_op_bra_ket_pcon[I,T](general_basis_core[I] *B,const int,const char[], const int[],
                          const double complex, const npy_intp, const set[vector[int]], const I[], I[], T[]) nogil

    int general_op_shift_sectors[I1,J1,I2,J2,K](general_basis_core[I1]*,const int,const char[],const int[],
                                 const double complex,const npy_intp,const I1[],const J1[],const npy_intp,
                                 const I2[],const J2[],const npy_intp,const K[],K[]) nogil

cdef extern from "general_basis_get_vec.h" namespace "basis_general":
    bool project_from_general_dense[I,J,T](general_basis_core[I] *B,const I[],const J[],const npy_intp,
                                    const npy_intp,const npy_intp,const T[],T[]) nogil
    bool project_from_general_pcon_dense[I,J,T](general_basis_core[I] *B,const I[],const J[],const npy_intp,
                                    const npy_intp,const npy_intp,const I[],const T[],T[]) nogil
    bool project_to_general_dense[I,J,T](general_basis_core[I] *B,const I[],const J[],const npy_intp,
                                    const npy_intp,const npy_intp,const T[],T[]) nogil
    bool project_to_general_pcon_dense[I,J,T](general_basis_core[I] *B,const I[],const J[],const npy_intp,
                                    const npy_intp,const npy_intp,const I[],const T[],T[]) nogil


cdef extern from "general_basis_get_amp.h" namespace "basis_general":
    int get_amp_general[I,J](general_basis_core[I] *B, const I [], J [], const npy_intp ) nogil
    int get_amp_general_light[I,J](general_basis_core[I] *B, const I [], J [], const npy_intp ) nogil



cdef extern from "misc.h" namespace "basis_general":
    K rep_position[K,I](const K,const I[],const I) nogil


cdef extern from "general_basis_rep.h" namespace "basis_general":
    void general_representative[I](general_basis_core[I] *B, const I[], I[], int[], int8_t[], const npy_intp) nogil

    int general_normalization[I,J](general_basis_core[I] *B, I[], J[], const npy_intp) nogil



cdef inline void python_to_basis_inplace(object python_val, state_type * val):
    cdef int i = 0
    val[0] = <state_type>(0)

    while(python_val!=0):
        val[0] = val[0] ^ ((<state_type>(<int>(python_val&1))) << i)
        i += 1
        python_val = python_val >> 1

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

cdef inline void set_zeros(state_type * ptr, npy_intp N) nogil:
    cdef npy_intp i
    for i in range(N):
        ptr[i] = <state_type>(0)

cdef inline void set_ones(state_type * ptr, npy_intp N) nogil:
    cdef npy_intp i
    for i in range(N):
        ptr[i] = <state_type>(1)

cdef extern from "general_basis_bitops.h" namespace "basis_general":

    void bitwise_op[I,binary_operator](const I[], const I[], bool[], I[], const npy_intp, binary_operator) nogil
    void bitwise_shift_op[I,J,binary_operator](const I[], const J[], bool[], I[], const npy_intp, binary_operator) nogil

    void bitwise_not_op_core[I](const I[], bool[], I[], const npy_intp) nogil

    cdef cppclass bitwise_and_op[I]:
        bitwise_and_op()
    cdef cppclass bitwise_or_op[I]:
        bitwise_or_op()
    cdef cppclass bitwise_xor_op[I]:
        bitwise_xor_op()
    cdef cppclass bitwise_left_shift_op[I,J]:
        bitwise_left_shift_op()
    cdef cppclass bitwise_right_shift_op[I,J]:
        bitwise_right_shift_op()


cdef inline void bitwise_and_op_core(state_type * x1_ptr, state_type * x2_ptr, bool * where_ptr, state_type * out_ptr, npy_intp Ns):
    cdef bitwise_and_op[state_type] * op = new bitwise_and_op[state_type]()
    with nogil:
        bitwise_op(x1_ptr,x2_ptr, where_ptr,out_ptr, Ns, op[0])

cdef inline void bitwise_or_op_core(state_type * x1_ptr, state_type * x2_ptr, bool * where_ptr, state_type * out_ptr, npy_intp Ns):
    cdef bitwise_or_op[state_type] * op = new bitwise_or_op[state_type]()
    with nogil:
        bitwise_op(x1_ptr,x2_ptr, where_ptr,out_ptr, Ns, op[0])

cdef inline void bitwise_xor_op_core(state_type * x1_ptr, state_type * x2_ptr, bool * where_ptr, state_type * out_ptr, npy_intp Ns):
    cdef bitwise_xor_op[state_type] * op = new bitwise_xor_op[state_type]()
    with nogil:
        bitwise_op(x1_ptr,x2_ptr, where_ptr,out_ptr, Ns, op[0])



cdef inline void bitwise_left_shift_op_core(state_type * x1_ptr, shift_type * x2_ptr, bool * where_ptr, state_type * out_ptr, npy_intp Ns):
    cdef bitwise_left_shift_op[state_type, shift_type] * op = new bitwise_left_shift_op[state_type, shift_type]()
    with nogil:
        bitwise_shift_op(x1_ptr,x2_ptr, where_ptr,out_ptr, Ns, op[0])

cdef inline void bitwise_right_shift_op_core(state_type * x1_ptr, shift_type * x2_ptr, bool * where_ptr, state_type * out_ptr, npy_intp Ns):
    cdef bitwise_right_shift_op[state_type, shift_type] * op = new bitwise_right_shift_op[state_type, shift_type]()
    with nogil:
        bitwise_shift_op(x1_ptr,x2_ptr, where_ptr,out_ptr, Ns, op[0])


cdef inline PyArray_Descr * get_Descr(ndarray arr):
    return PyArray_DESCR(arr)