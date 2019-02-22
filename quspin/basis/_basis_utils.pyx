# cython: language_level=2
# distutils: language=c++
# cython imports
cimport numpy as _np
cimport cython
from numpy cimport npy_intp
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


@cython.boundscheck(False)
@cython.cdivision(True)
def _shuffle_sites_core(const npy_intp sps,npy_intp[::1] T_tup,npy_type[:,::1] A, npy_type[:,::1] A_T):
    cdef npy_intp i_new,i_old,i,j,r
    cdef npy_intp N_extra_dim = A.shape[0]
    cdef npy_intp Ns = A.shape[1]
    cdef npy_intp M = max(T_tup)+1
    cdef npy_intp nd = T_tup.size
    cdef npy_intp[::1] sps_pow = sps**(_np.arange(M,dtype=_np.intp)[::-1])

    with nogil:
        if sps > 2:
            for i in range(Ns):
                j = 0
                for i_old in range(nd):
                    i_new = T_tup[i_old]
                    j += ((i/(sps_pow[i_new])) % sps)*(sps_pow[i_old])

                for r in range(N_extra_dim):
                    A_T[r,j] = A[r,i]

        else:
            for i in range(Ns):
                j = 0
                for i_old in range(nd):
                    i_new = T_tup[i_old]
                    j += ((i>>(M-i_new-1))&1)<<(M-i_old-1)


                for r in range(N_extra_dim):
                    A_T[r,j] = A[r,i]


def shuffle_sites(npy_intp sps,T_tup,A):
    A = _np.asanyarray(A)
    T_tup = _np.array(T_tup,dtype=_np.intp)


    extra_dim = A.shape[:-1]
    last_dim = A.shape[-1:]
    new_shape = (-1,)+A.shape[-1:]

    A = _np.ascontiguousarray(A)
    A = _np.reshape(A,new_shape,order="C")
    A_T = _np.zeros(A.shape,dtype=A.dtype,order="C")

    _shuffle_sites_core(sps,T_tup,A,A_T)
    A_T = _np.reshape(A_T,extra_dim+last_dim,order="C")

    return A_T


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