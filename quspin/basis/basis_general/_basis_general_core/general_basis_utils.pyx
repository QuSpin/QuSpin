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
        raise ValueError("basis is not representable with general_basis integer implementations.")


#############################################
########  bitwise operations  ###############
#############################################


def bitwise_not(x, out=None, where=None):
    """ Bitwise NOT operation `~x`.

    Performs pair-wise bitwise NOT operation basis states in integer representation.

    Parameters
    -----------
    x : array-like
        a collection of integers which contain basis states to apply pair-wise NOT on.
    out : array-like, optional
        an array to hold the resulting `~x` integer.
    where: array_like[bool], optional
        an array of booleans (mask) to define which elements of `x`` to apply the bitwise NOT operation on.

    Returns
    -------
    numpy.ndarray[uint]
        array of unsigned integers containing the result of the bitwise operation.

    Examples
    --------

    >>> basis=spin_basis_general(4) # 16 states Hilbert space
    >>> x=basis.states[:8]
    >>> not_x1=bitwise_not(x)
    >>> print(not_x)

    """
    cdef _np.ndarray[bool] py_array_where = _np.asarray(where)
    cdef _np.ndarray py_array_out
    
    x=_np.asarray(x)
    cdef _np.ndarray py_array_x = _np.asarray(x,order='C')
 

    cdef npy_intp Ns = x.shape[0]
    cdef void * x_ptr = _np.PyArray_GETPTR1(py_array_x,0)
    cdef bool * where_ptr = NULL
    
   
    if x.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("dtype must be one of the possible dtypes used as the representatives for the general basis class.")

    if out is None:
        out = _np.zeros(x.shape,dtype=x.dtype,order="C")
    else:
        if out.shape!=x.shape:
            raise TypeError("expecting same shape for out and x arrays.")

        if out.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
           raise TypeError("unsupported dtype for variable out. Expecting array of unsigned integers.")

    py_array_out = out
    out_ptr = _np.PyArray_GETPTR1(out,0)

    if where is not None:
        where_ptr = &py_array_where[0] #_np.PyArray_GETPTR1(where,0)

    if x.dtype == uint32:
        bitwise_not_op_core(<uint32_t*>x_ptr, where_ptr, <uint32_t*>out_ptr, Ns)
    elif x.dtype == uint64:
        bitwise_not_op_core(<uint64_t*>x_ptr, where_ptr, <uint64_t*>out_ptr, Ns)
    elif x.dtype == uint256:
        bitwise_not_op_core(<uint256_t*>x_ptr, where_ptr, <uint256_t*>out_ptr, Ns)
    elif x.dtype == uint1024:
        bitwise_not_op_core(<uint1024_t*>x_ptr, where_ptr, <uint1024_t*>out_ptr, Ns)
    elif x.dtype == uint4096:
        bitwise_not_op_core(<uint4096_t*>x_ptr, where_ptr, <uint4096_t*>out_ptr, Ns)
    elif x.dtype == uint16384:
        bitwise_not_op_core(<uint16384_t*>x_ptr, where_ptr, <uint16384_t*>out_ptr, Ns)    
    else:
        raise TypeError("basis dtype {} not recognized.".format(x.dtype))
    
    return out


def bitwise_and(x1, x2, out=None, where=None):
    """ Bitwise AND operation `x1 & x2`.

    Performs pair-wise bitwise AND operation on pairs of basis states in integer representation.

    Parameters
    -----------
    x1 : array-like
        a collection of integers which contain basis states to apply pair-wise AND on.
    x2 : array-like
        a collection of integers which contain basis states to apply pair-wise AND on.   
    out : array-like, optional
        an array to hold the resulting `x1 & x2` integer.
    where: array_like[bool], optional
        an array of booleans (mask) to define which pairs of elements of `(x1,x2)` to apply the bitwise AND operation on.

    Returns
    -------
    numpy.ndarray[uint]
        array of unsigned integers containing the result of the bitwise operation.

    Examples
    --------

    >>> basis=spin_basis_general(4) # 16 states Hilbert space
    >>> x1=basis.states[:8]
    >>> x2=basis.states[8:]
    >>> x1_and_x2=bitwise_and(x1,x2)
    >>> print(x1_and_x2)

    """
    cdef _np.ndarray[bool] py_array_where = _np.asarray(where)
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    x2=_np.asarray(x2)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')
    cdef _np.ndarray py_array_x2 = _np.asarray(x2,order='C')


    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(py_array_x2,0)
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

    py_array_out = out
    out_ptr = _np.PyArray_GETPTR1(out,0)

    if where is not None:
        where_ptr = &py_array_where[0] #_np.PyArray_GETPTR1(where,0)

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


def bitwise_or(x1, x2, out=None, where=None):
    """ Bitwise OR operation `x1 | x2`.

    Performs pair-wise bitwise OR operation on pairs of basis states in integer representation.

    Parameters
    -----------
    x1 : array-like
        a collection of integers which contain basis states to apply pair-wise OR on.
    x2 : array-like
        a collection of integers which contain basis states to apply pair-wise OR on.   
    out : array-like, optional
        an array to hold the resulting `x1 | x2` integer.
    where: array_like[bool], optional
        an array of booleans (mask) to define which pairs of elements of `(x1,x2)` to apply the bitwise OR operation on.

    Returns
    -------
    numpy.ndarray[uint]
        array of unsigned integers containing the result of the bitwise operation.

    Examples
    --------

    >>> basis=spin_basis_general(4) # 16 states Hilbert space
    >>> x1=basis.states[:8]
    >>> x2=basis.states[8:]
    >>> x1_or_x2=bitwise_or(x1,x2)
    >>> print(x1_or_x2)

    """
    cdef _np.ndarray[bool] py_array_where = _np.asarray(where)
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    x2=_np.asarray(x2)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')
    cdef _np.ndarray py_array_x2 = _np.asarray(x2,order='C')


    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(py_array_x2,0)
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

    py_array_out = out
    out_ptr = _np.PyArray_GETPTR1(out,0)

    if where is not None:
        where_ptr = &py_array_where[0] #_np.PyArray_GETPTR1(where,0)

    if x1.dtype == uint32:
        bitwise_or_op_core(<uint32_t*>x1_ptr, <uint32_t*>x2_ptr, where_ptr, <uint32_t*>out_ptr, Ns)
    elif x1.dtype == uint64:
        bitwise_or_op_core(<uint64_t*>x1_ptr, <uint64_t*>x2_ptr, where_ptr, <uint64_t*>out_ptr, Ns )
    elif x1.dtype == uint256:
        bitwise_or_op_core(<uint256_t*>x1_ptr, <uint256_t*>x2_ptr, where_ptr, <uint256_t*>out_ptr, Ns )
    elif x1.dtype == uint1024:
        bitwise_or_op_core(<uint1024_t*>x1_ptr, <uint1024_t*>x2_ptr, where_ptr, <uint1024_t*>out_ptr, Ns )
    elif x1.dtype == uint4096:
        bitwise_or_op_core(<uint4096_t*>x1_ptr, <uint4096_t*>x2_ptr, where_ptr, <uint4096_t*>out_ptr, Ns )
    elif x1.dtype == uint16384:
        bitwise_or_op_core(<uint16384_t*>x1_ptr, <uint16384_t*>x2_ptr, where_ptr, <uint16384_t*>out_ptr, Ns )    
    else:
        raise TypeError("basis dtype {} not recognized.".format(x1.dtype))
    
    return out


def bitwise_xor(x1, x2, out=None, where=None):
    """ Bitwise XOR operation `x1 ^ x2`.

    Performs pair-wise bitwise OR operation on pairs of basis states in integer representation.

    Parameters
    -----------
    x1 : array-like
        a collection of integers which contain basis states to apply pair-wise XOR on.
    x2 : array-like
        a collection of integers which contain basis states to apply pair-wise XOR on.   
    out : array-like, optional
        an array to hold the resulting `x1 ^ x2` integer.
    where: array_like[bool], optional
        an array of booleans (mask) to define which pairs of elements of `(x1,x2)` to apply the bitwise XOR operation on.

    Returns
    -------
    numpy.ndarray[uint]
        array of unsigned integers containing the result of the bitwise operation.

    Examples
    --------

    >>> basis=spin_basis_general(4) # 16 states Hilbert space
    >>> x1=basis.states[:8]
    >>> x2=basis.states[8:]
    >>> x1_xor_x2=bitwise_xor(x1,x2)
    >>> print(x1_xor_x2)

    """
    cdef _np.ndarray[bool] py_array_where = _np.asarray(where)
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    x2=_np.asarray(x2)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')
    cdef _np.ndarray py_array_x2 = _np.asarray(x2,order='C')


    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(py_array_x2,0)
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

    py_array_out = out
    out_ptr = _np.PyArray_GETPTR1(out,0)

    if where is not None:
        where_ptr = &py_array_where[0] #_np.PyArray_GETPTR1(where,0)

    if x1.dtype == uint32:
        bitwise_xor_op_core(<uint32_t*>x1_ptr, <uint32_t*>x2_ptr, where_ptr, <uint32_t*>out_ptr, Ns)
    elif x1.dtype == uint64:
        bitwise_xor_op_core(<uint64_t*>x1_ptr, <uint64_t*>x2_ptr, where_ptr, <uint64_t*>out_ptr, Ns )
    elif x1.dtype == uint256:
        bitwise_xor_op_core(<uint256_t*>x1_ptr, <uint256_t*>x2_ptr, where_ptr, <uint256_t*>out_ptr, Ns )
    elif x1.dtype == uint1024:
        bitwise_xor_op_core(<uint1024_t*>x1_ptr, <uint1024_t*>x2_ptr, where_ptr, <uint1024_t*>out_ptr, Ns )
    elif x1.dtype == uint4096:
        bitwise_xor_op_core(<uint4096_t*>x1_ptr, <uint4096_t*>x2_ptr, where_ptr, <uint4096_t*>out_ptr, Ns )
    elif x1.dtype == uint16384:
        bitwise_xor_op_core(<uint16384_t*>x1_ptr, <uint16384_t*>x2_ptr, where_ptr, <uint16384_t*>out_ptr, Ns )    
    else:
        raise TypeError("basis dtype {} not recognized.".format(x1.dtype))
    
    return out


def bitwise_left_shift(x1, x2, out=None, where=None):
    """ Bitwise LEFT SHIFT operation `x1 << x2`.

    Performs pair-wise bitwise OR operation on pairs of basis states in integer representation.

    Parameters
    -----------
    x1 : array-like
        a collection of integers which contain basis states to apply pair-wise LEFT SHIFT on.
    x2 : array-like
        a collection of integers to apply pair-wise LEFT SHIFT of `x1` with.  
    out : array-like, optional
        an array to hold the resulting `x1 << x2` integer.
    where: array_like[bool], optional
        an array of booleans (mask) to define which pairs of elements of `(x1,x2)` to apply the bitwise LEFT SHIFT operation on.

    Returns
    -------
    numpy.ndarray[uint]
        array of unsigned integers containing the result of the bitwise operation.

    Examples
    --------

    >>> basis=spin_basis_general(4) # 16 states Hilbert space
    >>> x1=basis.states[:8]
    >>> x2=basis.states[8:]
    >>> x1_leftshift_x2=bitwise_left_shift(x1,x2)
    >>> print(x1_leftshift_x2)

    """
    cdef _np.ndarray[bool] py_array_where = _np.asarray(where)
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    x2=_np.asarray(x2)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')
    cdef _np.ndarray py_array_x2 = _np.asarray(x2,order='C')


    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(py_array_x2,0)
    cdef bool * where_ptr = NULL
    

    if x1.shape != x2.shape:
        raise TypeError("expecting same shape for x1 and x2 arrays.")
    
    if x1.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("x1 dtype must be one of the possible dtypes used as the representatives for the general basis class.")
    if not _np.issubdtype(x2.dtype, _np.unsignedinteger): #x2.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("x2 dtype must be unsigned integer.")

    if out is None:
        out = _np.zeros(x1.shape,dtype=x1.dtype,order="C")
    else:
        if out.shape!=x1.shape:
            raise TypeError("expecting same shape for out and x1 arrays.")

        if out.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
           raise TypeError("unsupported dtype for variable out. Expecting array of unsigned integers.")

    py_array_out = out
    out_ptr = _np.PyArray_GETPTR1(out,0)

    if where is not None:
        where_ptr = &py_array_where[0] #_np.PyArray_GETPTR1(where,0)

    if x1.dtype == uint32:
        bitwise_left_shift_op_core(<uint32_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint32_t*>out_ptr, Ns)
    elif x1.dtype == uint64:
        bitwise_left_shift_op_core(<uint64_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint64_t*>out_ptr, Ns )
    elif x1.dtype == uint256:
        bitwise_left_shift_op_core(<uint256_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint256_t*>out_ptr, Ns )
    elif x1.dtype == uint1024:
        bitwise_left_shift_op_core(<uint1024_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint1024_t*>out_ptr, Ns )
    elif x1.dtype == uint4096:
        bitwise_left_shift_op_core(<uint4096_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint4096_t*>out_ptr, Ns )
    elif x1.dtype == uint16384:
        bitwise_left_shift_op_core(<uint16384_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint16384_t*>out_ptr, Ns )    
    else:
        raise TypeError("basis dtype {} not recognized.".format(x1.dtype))
    
    return out



def bitwise_right_shift(x1, x2, out=None, where=None):
    """ Bitwise RIGHT SHIFT operation `x1 >> x2`.

    Performs pair-wise bitwise OR operation on pairs of basis states in integer representation.

    Parameters
    -----------
    x1 : array-like
        a collection of integers which contain basis states to apply pair-wise RIGHT SHIFT on.
    x2 : array-like
        a collection of integers to apply pair-wise RIGHT SHIFT of `x1` with.   
    out : array-like, optional
        an array to hold the resulting `x1 >> x2` integer.
    where: array_like[bool], optional
        an array of booleans (mask) to define which pairs of elements of `(x1,x2)` to apply the bitwise RIGHT SHIFT operation on.

    Returns
    -------
    numpy.ndarray[uint]
        array of unsigned integers containing the result of the bitwise operation.

    Examples
    --------

    >>> basis=spin_basis_general(4) # 16 states Hilbert space
    >>> x1=basis.states[:8]
    >>> x2=basis.states[8:]
    >>> x1_rightshift_x2=bitwise_right_shift(x1,x2)
    >>> print(x1_rightshift_x2)

    """
    cdef _np.ndarray[bool] py_array_where = _np.asarray(where)
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    x2=_np.asarray(x2)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')
    cdef _np.ndarray py_array_x2 = _np.asarray(x2,order='C')


    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(py_array_x2,0)
    cdef bool * where_ptr = NULL
    

    if x1.shape != x2.shape:
        raise TypeError("expecting same shape for x1 and x2 arrays.")
    
    if x1.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("x1 dtype must be one of the possible dtypes used as the representatives for the general basis class.")
    if not _np.issubdtype(x2.dtype, _np.unsignedinteger): #x2.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("x2 dtype must be unsigned integer.")

    if out is None:
        out = _np.zeros(x1.shape,dtype=x1.dtype,order="C")
    else:
        if out.shape!=x1.shape:
            raise TypeError("expecting same shape for out and x1 arrays.")

        if out.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
           raise TypeError("unsupported dtype for variable out. Expecting array of unsigned integers.")

    py_array_out = out
    out_ptr = _np.PyArray_GETPTR1(out,0)

    if where is not None:
        where_ptr = &py_array_where[0] #_np.PyArray_GETPTR1(where,0)

    if x1.dtype == uint32:
        bitwise_right_shift_op_core(<uint32_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint32_t*>out_ptr, Ns)
    elif x1.dtype == uint64:
        bitwise_right_shift_op_core(<uint64_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint64_t*>out_ptr, Ns )
    elif x1.dtype == uint256:
        bitwise_right_shift_op_core(<uint256_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint256_t*>out_ptr, Ns )
    elif x1.dtype == uint1024:
        bitwise_right_shift_op_core(<uint1024_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint1024_t*>out_ptr, Ns )
    elif x1.dtype == uint4096:
        bitwise_right_shift_op_core(<uint4096_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint4096_t*>out_ptr, Ns )
    elif x1.dtype == uint16384:
        bitwise_right_shift_op_core(<uint16384_t*>x1_ptr, <unsigned long int *> x2_ptr, where_ptr, <uint16384_t*>out_ptr, Ns )    
    else:
        raise TypeError("basis dtype {} not recognized.".format(x1.dtype))
    
    return out

