# cython: language_level=2
# distutils: language=c++
from general_basis_core cimport uint32_t,uint64_t,uint128_t,uint256_t,uint512_t,uint1024_t
from libcpp.vector cimport vector
import numpy as _np
cimport numpy as _np
from numpy cimport npy_intp

# cdef extern from "source/general_basis_utils.h":
#     void boost_init_array[T](T *,npy_intp) nogil
#     vector[int] int_to_str_core[T](T*,int,int) nogil


uint32 = _np.uint32
uint64 = _np.uint64
uint128 = _np.dtype((_np.void,sizeof(uint128_t)))
uint256 = _np.dtype((_np.void,sizeof(uint256_t)))
uint512 = _np.dtype((_np.void,sizeof(uint512_t)))
uint1024 = _np.dtype((_np.void,sizeof(uint1024_t)))



def boost_zeros(shape,dtype=uint32):
    if dtype not in [uint32,uint64,uint128,uint256,uint512,uint1024]:
        raise TypeError("dtype must be one of the possible dtypes used as the representatives for the general basis class.")

    cdef _np.ndarray array = _np.zeros(shape,dtype=dtype)
    cdef void * ptr = _np.PyArray_GETPTR1(array,0)
    cdef uint32_t * uint32_ptr = NULL
    cdef uint64_t * uint64_ptr = NULL
    cdef uint128_t * uint128_ptr = NULL
    cdef uint256_t * uint256_ptr = NULL
    cdef uint512_t * uint512_ptr = NULL
    cdef uint1024_t * uint1024_ptr = NULL

    cdef npy_intp i,N

    N = array.size

    if array.dtype == uint32:
        with nogil:
            uint32_ptr = <uint32_t*>ptr
            for i in range(N):
                uint32_ptr[i] = <uint32_t>(0)
    elif array.dtype == uint64:
        with nogil:
            uint64_ptr = <uint64_t*>ptr
            for i in range(N):
                uint64_ptr[i] = <uint64_t>(0)
    elif array.dtype == uint128:
        with nogil:
            uint128_ptr = <uint128_t*>ptr
            for i in range(N):
                uint128_ptr[i] = <uint128_t>(0)
    elif array.dtype == uint256:
        with nogil:
            uint256_ptr = <uint256_t*>ptr
            for i in range(N):
                uint256_ptr[i] = <uint256_t>(0)
    elif array.dtype == uint512:
        with nogil:
            uint512_ptr = <uint512_t*>ptr
            for i in range(N):
                uint512_ptr[i] = <uint512_t>(0)
    elif array.dtype == uint1024:
        with nogil:
            uint1024_ptr = <uint1024_t*>ptr
            for i in range(N):
                uint1024_ptr[i] = <uint1024_t>(0)

    return array



# def get_site_list(a,int sps,int N):
#     cdef vector[int] sites

#     if type(a) is int:
#         if sps>2:
#             for i in range(N):
#                 sites.push_back(a%sps)
#                 a = a // sps
#         else:
#             for i in range(N):
#                 sites.push_back(a&1)
#                 a = a >> 1

#         return sites
#     else:
#         a = _np.array(a)
#         if a.ndim > 0:
#             raise ValueError

#     cdef _np.ndarray arr = a
#     cdef void * ptr = _np.PyArray_GETPTR1(arr,0)

#     if arr.dtype == uint32:
#         with nogil:
#             sites = int_to_str_core[uint32_t](<uint32_t*>ptr,sps,N)
#     elif arr.dtype == uint64:
#         with nogil:
#             sites = int_to_str_core[uint64_t](<uint64_t*>ptr,sps,N)
#     elif arr.dtype == uint128:
#         with nogil:
#             sites = int_to_str_core[uint128_t](<uint128_t*>ptr,sps,N)
#     elif arr.dtype == uint256:
#         with nogil:
#             sites = int_to_str_core[uint256_t](<uint256_t*>ptr,sps,N)
#     elif arr.dtype == uint512:
#         with nogil:
#             sites = int_to_str_core[uint512_t](<uint512_t*>ptr,sps,N)
#     elif arr.dtype == uint1024:
#         with nogil:
#             sites = int_to_str_core[uint1024_t](<uint1024_t*>ptr,sps,N)
#     else:
#         raise TypeError

#     return sites

