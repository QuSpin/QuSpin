# cython: language_level=2
# distutils: language=c++
from general_basis_core cimport *
import numpy as _np
cimport numpy as _np
from numpy cimport npy_intp



uint32 = _np.uint32
uint64 = _np.uint64
uint256 = _np.dtype((_np.void,sizeof(uint256_t)))
uint1024 = _np.dtype((_np.void,sizeof(uint1024_t)))
uint4096 = _np.dtype((_np.void,sizeof(uint4096_t)))
uint16384 = _np.dtype((_np.void,sizeof(uint16384_t)))



def boost_zeros(shape,dtype=uint32):
    if dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("dtype must be one of the possible dtypes used as the representatives for the general basis class.")

    cdef _np.ndarray array = _np.zeros(shape,dtype=dtype)
    cdef void * ptr = _np.PyArray_GETPTR1(array,0)
    cdef uint32_t * uint32_ptr = NULL
    cdef uint64_t * uint64_ptr = NULL
    cdef uint256_t * uint256_ptr = NULL
    cdef uint1024_t * uint1024_ptr = NULL
    cdef uint4096_t * uint4096_ptr = NULL
    cdef uint16384_t * uint16384_ptr = NULL

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
    elif array.dtype == uint256:
        with nogil:
            uint256_ptr = <uint256_t*>ptr
            for i in range(N):
                uint256_ptr[i] = <uint256_t>(0)
    elif array.dtype == uint1024:
        with nogil:
            uint1024_ptr = <uint1024_t*>ptr
            for i in range(N):
                uint1024_ptr[i] = <uint1024_t>(0)
    elif array.dtype == uint4096:
        with nogil:
            uint4096_ptr = <uint4096_t*>ptr
            for i in range(N):
                uint4096_ptr[i] = <uint4096_t>(0)
    elif array.dtype == uint16384:
        with nogil:
            uint16384_ptr = <uint16384_t*>ptr
            for i in range(N):
                uint16384_ptr[i] = <uint16384_t>(0)

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
            
    else:
        nbits = N

    if nbits <= 32:
        return uint32
    elif nbits <= 64:
        return uint64
    elif nbits <= 256:
        return uint256
    elif nbits <= 1024:
        return uint1024
    elif nbits <= 4096:
        return uint4096
    elif nbits <= 16384:
        return uint16384
    else:
        raise ValueError("basis is not representable with general_basis integer imeplmentations.")

cdef inline void bitwise_and_op_core(state_type * x1_ptr, state_type * x2_ptr, bool * where_ptr, state_type * out_ptr, npy_intp Ns):
    cdef bitwise_and_op[state_type] * op = new bitwise_and_op[state_type]()
    with nogil:
        bitwise_op(x1_ptr,x2_ptr, where_ptr,out_ptr, Ns, op[0])


def bitwise_and(_np.ndarray x1, _np.ndarray x2, _np.ndarray[::1] out=None, bool[::1] where=None):
    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(x2,0)
    cdef bool * where_ptr = NULL

    if x1.shape != x2.shape:
        raise TypeError("expecting same shape for x1 and x2 arrays.")
    
    if x1.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("dtype must be one of the possible dtypes used as the representatives for the general basis class.")
    if x2.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("dtype must be one of the possible dtypes used as the representatives for the general basis class.")

    if out is None:
        out = _np.zeros(x1.shape,dtype=x1.dtype,order="C")
    else:
        if out.shape!=x1.shape:
            raise TypeError("expecting same shape for out and x1 arrays.")

        if out.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
           raise TypeError("unsupported dtype for variable out. Expecting array of unsigned integers.")

    out_ptr = _np.PyArray_GETPTR1(out,0)

    if where is not None:
        where_ptr = &where[0]

    if x1.dtype == uint32:
        bitwise_and_op_core(<uint32_t*>x1_ptr, <uint32_t*>x2_ptr, where_ptr, <uint32_t*>out_ptr, Ns)
    elif x1.dtype == uint64:
        bitwise_and_op_core(<uint64_t*>x1_ptr, <uint64_t*>x2_ptr, where_ptr, <uint64_t*>out_ptr, Ns )
    elif x1.dtype == uint256:
        bitwise_and_op_core(<uint256_t*>x1_ptr, <uint256_t*>x2_ptr, where_ptr, <uint256_t*>out_ptr, Ns )
    elif x1.dtype == uint1024:
        bitwise_and_op_core(<uint1024_t*>x1_ptr, <uint1024_t*>x2_ptr, where_ptr, <uint1024_t*>out_ptr, Ns )
    elif x1.dtype == uint4096:
        bitwise_and_op_core(<uint4096_t*>x1_ptr, <uint4096_t*>x2_ptr, where_ptr, <uint4096_t*>out_ptr, Ns )
    elif x1.dtype == uint16384:
        bitwise_and_op_core(<uint16384_t*>x1_ptr, <uint16384_t*>x2_ptr, where_ptr, <uint16384_t*>out_ptr, Ns )    
    else:
        raise TypeError("basis dtype {} not recognized.".format(x1.dtype))
    
    return out

