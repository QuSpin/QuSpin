# cython: language_level=2
# distutils: language=c++
# cython imports
cimport numpy as _np
cimport cython
from numpy cimport npy_intp,npy_uintp
from numpy cimport float32_t,float64_t,complex64_t,complex128_t
from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp cimport bool
from general_basis_core cimport state_type,uint32_t,uint64_t,uint256_t,uint1024_t,uint4096_t,uint16384_t,python_to_basis,basis_to_python
# python imports
import numpy as _np



_uint32 = _np.uint32
_uint64 = _np.uint64
_uint256 = _np.dtype((_np.void,sizeof(uint256_t)))
_uint1024 = _np.dtype((_np.void,sizeof(uint1024_t)))
_uint4096 = _np.dtype((_np.void,sizeof(uint4096_t)))
_uint16384 = _np.dtype((_np.void,sizeof(uint16384_t)))


ctypedef fused npy_type:
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    float32_t
    float64_t
    complex64_t
    complex128_t

__all__ = ["basis_int_to_python_int","get_basis_index","shuffle_sites"]


cdef extern from "shuffle_sites.h":
    void shuffle_sites_core[T](const int32_t,const npy_intp,const int32_t*,const npy_intp,const npy_intp,const T*,T*) nogil
    void shuffle_sites_core_base_2[T](const int32_t,const int32_t*,const npy_intp,const npy_intp,const T*,T*) nogil
    void shuffle_sites_strid[T](const int32_t,const npy_intp*,const int32_t*,const npy_intp,const npy_intp,const T*,T*) nogil

# @cython.boundscheck(False)
# def _shuffle_sites_core(npy_intp sps,int32_t[::1] T_tup,npy_type[:,::1] A, npy_type[:,::1] A_T):

#     cdef npy_intp n_row = A.shape[0]
#     cdef npy_intp n_col = A.shape[1]
#     cdef int32_t nd = T_tup.size
#     cdef int32_t * T_tup_ptr = &T_tup[0]
#     cdef npy_type * A_ptr = &A[0,0]
#     cdef npy_type * A_T_ptr = &A_T[0,0]

#     if nd > 64:
#         raise ValueError("can't transpose more than 64 dimensions")

#     with nogil:
#         if sps > 2:
#             shuffle_sites_core(nd,sps,T_tup_ptr,n_row,n_col,A_ptr,A_T_ptr)
#         else:
#             shuffle_sites_core_base_2(nd,T_tup_ptr,n_row,n_col,A_ptr,A_T_ptr)


@cython.boundscheck(False)
def _shuffle_sites_core(npy_intp[::1] R_tup,int32_t[::1] T_tup,npy_type[:,::1] A, npy_type[:,::1] A_T):

    cdef npy_intp n_row = A.shape[0]
    cdef npy_intp n_col = A.shape[1]
    cdef int32_t nd = T_tup.size
    cdef int32_t * T_tup_ptr = &T_tup[0]
    cdef npy_intp * R_tup_ptr = &R_tup[0]
    cdef npy_type * A_ptr = &A[0,0]
    cdef npy_type * A_T_ptr = &A_T[0,0]

    if nd > 64:
        raise ValueError("can't transpose more than 64 dimensions")

    with nogil:
        shuffle_sites_strid(nd,R_tup_ptr,T_tup_ptr,n_row,n_col,A_ptr,A_T_ptr)






def _reduce_transpose(T_tup,sps=2):
    """
    This function is used to reduce the size of the reshaping and transposing tuples.
    It does this by finding consecutive sites in T_tup and then grouping those indices together
    replacing it with a single index which has a range of values given by sps**(num_sites) where
    'num_sites' is the number of sites within that grouping.

    for example when the subsystem is just a consecutive section starting at site 0, this function suggests that there is no 
    need to do a transpose returning T_tup: (0,) and R_tup: (2**N,).
    """
    T_tup = list(T_tup)

    sub_lists = []
    while(len(T_tup)>0):
        n = len(T_tup)
        sub_list = (T_tup[0],)
        for i in range(n-1):
            if T_tup[i]+1 == T_tup[i+1]:
                sub_list = sub_list + (T_tup[i+1],)
            else:
                break

        for s in sub_list:
            T_tup.remove(s)

        sub_lists.append(sub_list)

    sub_lists_sorted = sorted(sub_lists,key=lambda x:x[0])

    T_tup = []
    R_tup = []
    for sub_list in sub_lists_sorted:
        R_tup.append(sps**len(sub_list))

    for i,sub_list in enumerate(sub_lists):
        j = sub_lists_sorted.index(sub_list)
        T_tup.append(j)

    return tuple(T_tup),tuple(R_tup)



def shuffle_sites(npy_intp sps,T_tup,A):
    A = _np.ascontiguousarray(A)

    T_tup_reduced,R_tup_reduced = _reduce_transpose(T_tup,sps)

    if len(T_tup_reduced) > 1:

        extra_dim = A.shape[:-1]
        last_dim = A.shape[-1:]
        new_shape = (-1,)+A.shape[-1:]
        A = A.reshape(new_shape,order="C")
        T_tup = _np.array(T_tup_reduced,dtype=_np.int32)
        R_tup = _np.array(R_tup_reduced,dtype=_np.intp)

        if len(T_tup_reduced) < 32: # use numpy as it is much faster
            reshape_tup = A.shape[:1] + tuple(R_tup)
            A = A.reshape(reshape_tup,order="C")
            transpose_tup = (0,)+tuple(T_tup+1)
            A = A.transpose(transpose_tup)
            A_T = A.reshape(extra_dim+last_dim,order="C")
        else: # spill into less efficient C++ code
            A_T = _np.zeros(A.shape,dtype=A.dtype,order="C")
            _shuffle_sites_core(R_tup,T_tup,A,A_T)

        return A_T
    else:
        return A


def basis_int_to_python_int(a):

    cdef _np.ndarray a_wrapper = _np.array(a)
    cdef object python_int = 0
    cdef object i = 0
    cdef void * ptr = _np.PyArray_GETPTR1(a_wrapper,0)

    if a_wrapper.dtype in [_np.int32,_np.int64]:
        a_wrapper = a_wrapper.astype(_np.object)

    if a_wrapper.dtype == _np.object:
        return a
    elif a_wrapper.dtype == _uint32:
        return basis_to_python[uint32_t](<uint32_t*>ptr)
    elif a_wrapper.dtype == _uint64:
        return basis_to_python[uint64_t](<uint64_t*>ptr)
    elif a_wrapper.dtype == _uint256:
        return basis_to_python[uint256_t](<uint256_t*>ptr)
    elif a_wrapper.dtype == _uint1024:
        return basis_to_python[uint1024_t](<uint1024_t*>ptr)
    elif a_wrapper.dtype == _uint4096:
        return basis_to_python[uint4096_t](<uint4096_t*>ptr)
    elif a_wrapper.dtype == _uint16384:
        return basis_to_python[uint16384_t](<uint16384_t*>ptr)
    else:
        raise ValueError("dtype {} is not recognized, must be python integer or basis type".format(a_wrapper.dtype))


cdef search_array(state_type * ptr,npy_intp n, object value):
    cdef state_type val = <state_type>(0)
    cdef npy_intp i = 0

    val = python_to_basis[state_type](value,val)

    for i in range(n):
        if(ptr[i]==val):
            return i

    return -1

def get_basis_index(_np.ndarray basis,object val):
    cdef object value = basis_int_to_python_int(val)
    cdef void * ptr = _np.PyArray_GETPTR1(basis,0)
    cdef npy_intp n = basis.size
    cdef npy_intp j = 0
    cdef npy_intp i = -1

    if basis.dtype == _uint32:
        i = search_array[uint32_t](<uint32_t*>ptr,n,value)
    elif basis.dtype == _uint64:
        i = search_array[uint64_t](<uint64_t*>ptr,n,value)
    elif basis.dtype == _uint256:
        i = search_array[uint256_t](<uint256_t*>ptr,n,value)
    elif basis.dtype == _uint1024:
        i = search_array[uint1024_t](<uint1024_t*>ptr,n,value)
    elif basis.dtype == _uint4096:
        i = search_array[uint4096_t](<uint4096_t*>ptr,n,value)
    elif basis.dtype == _uint16384:
        i = search_array[uint16384_t](<uint16384_t*>ptr,n,value)
    elif basis.dtype == _np.dtype('O'):
        for j in range(n):
            if(basis[i]==value):
                i = j
                break

    if i < 0:
        raise ValueError("s must be representive state in basis. ")

    return i