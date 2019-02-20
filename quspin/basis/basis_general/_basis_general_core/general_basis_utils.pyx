# cython: language_level=2
# distutils: language=c++
from general_basis_core cimport uint32_t,uint64_t,uint128_t,uint256_t,uint512_t,uint1024_t
import numpy as _np
cimport numpy as _np
from numpy cimport npy_intp



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



def get_basis_type(N, Np, sps):

    if sps>2:
        try:
            Np = max(list(Np))
        except TypeError: 
            pass

        # calculates the datatype which will fit the largest representative state in basis
        if Np is None:
            # if no particle conservation the largest representative is sps**N-1
            s_max = sps**N-1
        else:
            # if particles are conservated the largest representative is placing all particles as far left
            # as possible. 
            l=Np//(sps-1)
            s_max = sum((sps-1)*sps**(N-1-i)  for i in range(l))
            s_max += (Np%(sps-1))*sps**(N-l-1)

        s_max = int(s_max)

        nbits = 0
        while(s_max>0):
            s_max >>= 1
            nbits += 1

        if nbits <= 32:
            return uint32
        elif nbits <= 64:
            return uint64
        elif nbits <= 128:
            return uint128
        elif nbits <= 256:
            return uint256
        elif nbits <= 512:
            return uint512
        elif nbits <= 1024:
            return uint1024
        else:
            raise ValueError("basis is not representable with general_basis integer imeplmentations.")

    else:
        if N <= 32:
            return uint32
        elif N <= 64:
            return uint64
        elif N <= 128:
            return uint128
        elif N <= 256:
            return uint256
        elif N <= 512:
            return uint512
        elif N <= 1024:
            return uint1024
        else:
            raise ValueError("basis is not representable with general_basis integer imeplmentations.")

