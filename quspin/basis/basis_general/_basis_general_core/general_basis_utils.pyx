# cython: embedsignature=True
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



def basis_zeros(shape,dtype=uint32):
    """ Allocates initialized array using the QuSpin basis integers.

    QuSpin uses non-standard numpy datatypes which wrap boost multiprecision integers. 
    This function creates an array of integers which are properly initialized to 0. 

    Parameters
    -----------
    shape : tuple
        shape of the numpy array.
    dtype : numpy.dtype, optional
        numpy dtype used to create the array, one can use QuSpin defined dtypes here: `uint32`,`uint64`,`uint256`,`uint1024`,`uint4096`, or `uint16384`.


    Returns
    -------
    numpy.array object
        a properly initialized array of type `dtype` initialized to 0.

    Examples
    --------

    >>> a = basis_zeros((100,))  

    """
    if dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("dtype must be one of the possible dtypes used as the representatives for the general basis class.")

    cdef _np.ndarray array = _np.zeros(shape,dtype=dtype)
    cdef void * ptr = _np.PyArray_GETPTR1(array,0)
    cdef npy_intp i,N

    N = array.size

    if array.dtype == uint32:
        with nogil:
            set_zeros[uint32_t](<uint32_t*>ptr,N)
    elif array.dtype == uint64:
        with nogil:
            set_zeros[uint64_t](<uint64_t*>ptr,N)
    elif array.dtype == uint256:
        with nogil:
            set_zeros[uint256_t](<uint256_t*>ptr,N)
    elif array.dtype == uint1024:
        with nogil:
            set_zeros[uint1024_t](<uint1024_t*>ptr,N)
    elif array.dtype == uint4096:
        with nogil:
            set_zeros[uint4096_t](<uint4096_t*>ptr,N)
    elif array.dtype == uint16384:
        with nogil:
            set_zeros[uint16384_t](<uint16384_t*>ptr,N)

    return array


def basis_ones(shape,dtype=uint32):
    """ Allocates initialized array using the QuSpin basis integers.

    QuSpin uses non-standard numpy datatypes which wrap boost multiprecision integers. 
    This function creates an array of integers which are properly initialized to 1. 

    Parameters
    -----------
    shape : tuple
        shape of the numpy array.
    dtype : numpy.dtype, optional
        numpy dtype used to create the array, one can use QuSpin defined dtypes here: `uint32`,`uint64`,`uint256`,`uint1024`,`uint4096`, or `uint16384`.


    Returns
    -------
    numpy.array object
        a properly initialized array of type `dtype` initialized to 1.

    Examples
    --------

    >>> a = basis_ones((100,))  

    """
    if dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("dtype must be one of the possible dtypes used as the representatives for the general basis class.")

    cdef _np.ndarray array = _np.zeros(shape,dtype=dtype)
    cdef void * ptr = _np.PyArray_GETPTR1(array,0)
    cdef npy_intp i,N

    N = array.size

    if array.dtype == uint32:
        with nogil:
            set_ones[uint32_t](<uint32_t*>ptr,N)
    elif array.dtype == uint64:
        with nogil:
            set_ones[uint64_t](<uint64_t*>ptr,N)
    elif array.dtype == uint256:
        with nogil:
            set_ones[uint256_t](<uint256_t*>ptr,N)
    elif array.dtype == uint1024:
        with nogil:
            set_ones[uint1024_t](<uint1024_t*>ptr,N)
    elif array.dtype == uint4096:
        with nogil:
            set_ones[uint4096_t](<uint4096_t*>ptr,N)
    elif array.dtype == uint16384:
        with nogil:
            set_ones[uint16384_t](<uint16384_t*>ptr,N)
            
    return array


def get_basis_type(N, Np, sps):
    """ Returns minimum dtype to represent manybody state. 

    Given the system size, number of particles and states per site this function calculates the minimum type required to represent those states.

    Parameters
    -----------
    N : int
        total number of sites on lattice
    Np : None, int, or list of ints
        number of particles on the lattice. For a list, the largest particle sector is chosen for the representation. 
    sps: int
        number of possible states allowed per site. e.g. spin-1/2 has 2 possible states per site. 

    Returns
    -------
    numpy.dtype object
        the appropriate dtype size to represent the system. will be one of:  `uint32`,`uint64`,`uint256`,`uint1024`,`uint4096`, or `uint16384`.

    Examples
    --------

    >>> dtype = get_basis_type(10,5,2)  

    """
    sps = int(sps)
    N = int(N)

    try:
        Np = max(list(Np))
    except TypeError: 
        if Np is not None and type(Np) is not int:
            raise ValueError("{} is not None, an integer, or list of integers".format(Np))

    if Np is not None and (Np > N*(sps-1)):
        raise ValueError("{} particle(s) will not fit into system size {} with sps={}".format(Np,N,sps))

    if sps>2:
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
            
    elif sps == 2:
        nbits = N
    else:
        raise ValueError("sps must be larger than 1.")

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
        an array of booleans (mask) to define which elements of `x` to apply the bitwise NOT operation on.

    Returns
    -------
    numpy.ndarray
        array of unsigned integers containing the result of the bitwise operation. Output is of same data type as `x`.

    Examples
    --------
    
    >>> from quspin.basis import spin_basis_general, basis_int_to_python_int
    >>> N=100 # sites
    >>> basis=spin_basis_general(N,Nup=1) # 1 particle 
    >>> x=[basis.states[0]] # large integer stored as a byte
    >>> not_x=bitwise_not(x)
    >>> print('original integer stored as a byte:') 
    >>> print(x)
    >>> x_int=basis_int_to_python_int(x) # cast byte as python integer
    >>> print('original integer in integer form:')
    >>> print(x_int)
    >>> print('result in integer form:')
    >>> print(basis_int_to_python_int(not_x))

    """
    cdef _np.ndarray[uint8_t] py_array_where 
    cdef _np.ndarray py_array_out
    
    x=_np.asarray(x)
    cdef _np.ndarray py_array_x = _np.asarray(x,order='C')
 
    cdef npy_intp Ns = x.shape[0]
    cdef void * x_ptr = _np.PyArray_GETPTR1(py_array_x,0)
    
    cdef bool * where_ptr = NULL
    if where is not None:
        py_array_where = _np.asarray(where,dtype=_np.uint8)
       
        if x.size!=py_array_where.size:
            raise TypeError("expecting same size for x and where input arrays.")
       
        if py_array_where.ndim!=1:
            raise TypeError("where must be a 1d array.")
       
        where_ptr =<bool*> &py_array_where[0]

    if x.ndim!=1:
        raise TypeError("x must be a 1d array.")
    
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

    if py_array_out.ndim!=1:
            raise TypeError("out must be a 1d array.")

    if x.size != py_array_out.size:
        raise TypeError("expecting same size for x and out input arrays.")

    out_ptr = _np.PyArray_GETPTR1(py_array_out,0)

   
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
    
    return py_array_out


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
    numpy.ndarray
        array of unsigned integers containing the result of the bitwise operation. Output is of same data type as `x`.

    Examples
    --------

    >>> from quspin.basis import spin_basis_general, basis_int_to_python_int
    >>> N=100 # sites
    >>> basis=spin_basis_general(N,Nup=1) # 1 particle 
    >>> x1=[basis.states[0]] # large integer stored as a byte
    >>> x2=[basis.states[1]] # large integer stored as a byte
    >>> x1_and_x2=bitwise_and(x1,x2)
    >>> print('x1 integer stored as a byte:') 
    >>> print(x1)
    >>> x1_int=basis_int_to_python_int(x1) # cast byte as python integer
    >>> print('x1 integer in integer form:')
    >>> print(x1_int)
    >>> print('result in integer form:')
    >>> print(basis_int_to_python_int(x1_and_x2))

    """
    cdef _np.ndarray[uint8_t] py_array_where 
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    x2=_np.asarray(x2)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')
    cdef _np.ndarray py_array_x2 = _np.asarray(x2,order='C')


    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(py_array_x2,0)
    
    cdef bool * where_ptr = NULL
    if where is not None:
        py_array_where = _np.asarray(where,dtype=_np.uint8)
       
        if x1.size!=py_array_where.size:
            raise TypeError("expecting same size for x1 and where input arrays.")
       
        if py_array_where.ndim!=1:
            raise TypeError("where must be a 1d array.")
       
        where_ptr =<bool*> &py_array_where[0]
    

    if x1.size != x2.size:
        raise TypeError("expecting same size for x1 and x2 input arrays.")
    if x1.shape != x2.shape:
        raise TypeError("expecting same shape for x1 and x2 arrays.")
    if x1.ndim!=1:
        raise TypeError("x1 must be a 1d array.")
    if x2.ndim!=1:
        raise TypeError("x2 must be a 1d array.")

    
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

    if py_array_out.ndim!=1:
            raise TypeError("out must be a 1d array.")

    if x1.size != py_array_out.size:
        raise TypeError("expecting same size for x1 and out input arrays.")

    out_ptr = _np.PyArray_GETPTR1(py_array_out,0)

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
    
    return py_array_out


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
    numpy.ndarray
        array of unsigned integers containing the result of the bitwise operation. Output is of same data type as `x`.

    Examples
    --------

    >>> from quspin.basis import spin_basis_general, basis_int_to_python_int
    >>> N=100 # sites
    >>> basis=spin_basis_general(N,Nup=1) # 1 particle 
    >>> x1=[basis.states[0]] # large integer stored as a byte
    >>> x2=[basis.states[1]] # large integer stored as a byte
    >>> x1_or_x2=bitwise_or(x1,x2)
    >>> print('x1 integer stored as a byte:') 
    >>> print(x1)
    >>> x1_int=basis_int_to_python_int(x1) # cast byte as python integer
    >>> print('x1 integer in integer form:')
    >>> print(x1_int)
    >>> print('result in integer form:')
    >>> print(basis_int_to_python_int(x1_or_x2))

    """
    cdef _np.ndarray[uint8_t] py_array_where 
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    x2=_np.asarray(x2)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')
    cdef _np.ndarray py_array_x2 = _np.asarray(x2,order='C')


    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(py_array_x2,0)
    
    cdef bool * where_ptr = NULL
    if where is not None:
        py_array_where = _np.asarray(where,dtype=_np.uint8)
       
        if x1.size!=py_array_where.size:
            raise TypeError("expecting same size for x1 and where input arrays.")
       
        if py_array_where.ndim!=1:
            raise TypeError("where must be a 1d array.")
       
        where_ptr =<bool*> &py_array_where[0] 
    

    if x1.size != x2.size:
        raise TypeError("expecting same size for x1 and x2 input arrays.")
    if x1.shape != x2.shape:
        raise TypeError("expecting same shape for x1 and x2 arrays.")
    if x1.ndim!=1:
        raise TypeError("x1 must be a 1d array.")
    if x2.ndim!=1:
        raise TypeError("x2 must be a 1d array.")

    
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

    if py_array_out.ndim!=1:
            raise TypeError("out must be a 1d array.")

    if x1.size != py_array_out.size:
        raise TypeError("expecting same size for x1 and out input arrays.")

    out_ptr = _np.PyArray_GETPTR1(py_array_out,0)

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
    
    return py_array_out


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
    numpy.ndarray
        array of unsigned integers containing the result of the bitwise operation. Output is of same data type as `x`.

    Examples
    --------

    >>> from quspin.basis import spin_basis_general, basis_int_to_python_int
    >>> N=100 # sites
    >>> basis=spin_basis_general(N,Nup=1) # 1 particle 
    >>> x1=[basis.states[0]] # large integer stored as a byte
    >>> x2=[basis.states[1]] # large integer stored as a byte
    >>> x1_xor_x2=bitwise_xor(x1,x2)
    >>> print('x1 integer stored as a byte:') 
    >>> print(x1)
    >>> x1_int=basis_int_to_python_int(x1) # cast byte as python integer
    >>> print('x1 integer in integer form:')
    >>> print(x1_int)
    >>> print('result in integer form:')
    >>> print(basis_int_to_python_int(x1_xor_x2))

    """
    cdef _np.ndarray[uint8_t] py_array_where 
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    x2=_np.asarray(x2)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')
    cdef _np.ndarray py_array_x2 = _np.asarray(x2,order='C')


    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    cdef void * x2_ptr = _np.PyArray_GETPTR1(py_array_x2,0)
    
    cdef bool * where_ptr = NULL
    if where is not None:
        py_array_where = _np.asarray(where,dtype=_np.uint8)
       
        if x1.size!=py_array_where.size:
            raise TypeError("expecting same size for x1 and where input arrays.")
       
        if py_array_where.ndim!=1:
            raise TypeError("where must be a 1d array.")
       
        where_ptr =<bool*> &py_array_where[0]
    

    if x1.size != x2.size:
        raise TypeError("expecting same size for x1 and x2 input arrays.")
    if x1.shape != x2.shape:
        raise TypeError("expecting same shape for x1 and x2 arrays.")
    if x1.ndim!=1:
        raise TypeError("x1 must be a 1d array.")
    if x2.ndim!=1:
        raise TypeError("x2 must be a 1d array.")

    
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

    if py_array_out.ndim!=1:
            raise TypeError("out must be a 1d array.")

    if x1.size != py_array_out.size:
        raise TypeError("expecting same size for x1 and out input arrays.")

    out_ptr = _np.PyArray_GETPTR1(py_array_out,0)

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
    
    return py_array_out




def bitwise_leftshift(x1, x2, out=None, where=None):
    """ Bitwise LEFT SHIFT operation `x1 << x2`.

    Performs pair-wise bitwise OR operation on pairs of basis states in integer representation.

    Parameters
    ----------
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
    numpy.ndarray
        array of unsigned integers containing the result of the bitwise operation. Output is of same data type as `x`.

    Examples
    --------

    >>> from quspin.basis import spin_basis_general, basis_int_to_python_int
    >>> N=100 # sites
    >>> basis=spin_basis_general(N,Nup=1) # 1 particle 
    >>> x1=[basis.states[0]] # large integer stored as a byte
    >>> x2=[basis.states[1]] # large integer stored as a byte
    >>> x1_leftshift_x2=bitwise_leftshift(x1,x2)
    >>> print('x1 integer stored as a byte:') 
    >>> print(x1)
    >>> x1_int=basis_int_to_python_int(x1) # cast byte as python integer
    >>> print('x1 integer in integer form:')
    >>> print(x1_int)
    >>> print('result in integer form:')
    >>> print(basis_int_to_python_int(x1_leftshift_x2))

    """

    return _bitwise_left_shift(x1, x2, out=out, where=where)


def bitwise_rightshift(x1, x2, out=None, where=None):
    """ Bitwise RIGHT SHIFT operation `x1 >> x2`.

    Performs pair-wise bitwise OR operation on pairs of basis states in integer representation.

    Parameters
    ----------
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
    numpy.ndarray
        array of unsigned integers containing the result of the bitwise operation. Output is of same data type as `x`.

    Examples
    --------

    >>> from quspin.basis import spin_basis_general, basis_int_to_python_int
    >>> N=100 # sites
    >>> basis=spin_basis_general(N,Nup=1) # 1 particle 
    >>> x1=[basis.states[0]] # large integer stored as a byte
    >>> x2=[basis.states[1]] # large integer stored as a byte
    >>> x1_rightshift_x2=bitwise_rightshift(x1,x2)
    >>> print('x1 integer stored as a byte:') 
    >>> print(x1)
    >>> x1_int=basis_int_to_python_int(x1) # cast byte as python integer
    >>> print('x1 integer in integer form:')
    >>> print(x1_int)
    >>> print('result in integer form:')
    >>> print(basis_int_to_python_int(x1_rightshift_x2)) 

    """

    return _bitwise_right_shift(x1, x2, out=out, where=where)


def _bitwise_left_shift(x1, shift_type[:] x2, out=None, where=None):
    
    cdef _np.ndarray[uint8_t] py_array_where 
    cdef _np.ndarray py_array_out
    
    x1=_np.asarray(x1)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')

    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    
    cdef bool * where_ptr = NULL
    if where is not None:
        py_array_where = _np.asarray(where,dtype=_np.uint8)
       
        if x1.size!=py_array_where.size:
            raise TypeError("expecting same size for x1 and where input arrays.")
       
        if py_array_where.ndim!=1:
            raise TypeError("where must be a 1d array.")
       
        where_ptr =<bool*> &py_array_where[0] 
    

    if x1.size != x2.size:
        raise TypeError("expecting same size for x1 and x2 input arrays.")
    
    if x1.ndim!=1:
        raise TypeError("x1 must be a 1d array.")

    if x1.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("x1 dtype must be one of the possible dtypes used as the representatives for the general basis class.")
   
    if out is None:
        out = _np.zeros(x1.shape,dtype=x1.dtype,order="C")
    else:
        if out.shape!=x1.shape:
            raise TypeError("expecting same shape for out and x1 arrays.")

        if out.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
           raise TypeError("unsupported dtype for variable out. Expecting array of unsigned integers.")

    py_array_out = out

    if py_array_out.ndim!=1:
            raise TypeError("out must be a 1d array.")

    if x1.size != py_array_out.size:
        raise TypeError("expecting same size for x1 and out input arrays.")

    out_ptr = _np.PyArray_GETPTR1(py_array_out,0)


    if x1.dtype == uint32:
        bitwise_left_shift_op_core(<uint32_t*>x1_ptr, &x2[0], where_ptr, <uint32_t*>out_ptr, Ns)
    elif x1.dtype == uint64:
        bitwise_left_shift_op_core(<uint64_t*>x1_ptr, &x2[0], where_ptr, <uint64_t*>out_ptr, Ns )
    elif x1.dtype == uint256:
        bitwise_left_shift_op_core(<uint256_t*>x1_ptr, &x2[0], where_ptr, <uint256_t*>out_ptr, Ns )
    elif x1.dtype == uint1024:
        bitwise_left_shift_op_core(<uint1024_t*>x1_ptr, &x2[0], where_ptr, <uint1024_t*>out_ptr, Ns )
    elif x1.dtype == uint4096:
        bitwise_left_shift_op_core(<uint4096_t*>x1_ptr, &x2[0], where_ptr, <uint4096_t*>out_ptr, Ns )
    elif x1.dtype == uint16384:
        bitwise_left_shift_op_core(<uint16384_t*>x1_ptr, &x2[0], where_ptr, <uint16384_t*>out_ptr, Ns )    
    else:
        raise TypeError("basis dtype {} not recognized.".format(x1.dtype))
    
    return py_array_out


def _bitwise_right_shift(x1, shift_type[:] x2, out=None, where=None):
    
    cdef _np.ndarray[uint8_t] py_array_where 
    cdef _np.ndarray py_array_out
   
    x1=_np.asarray(x1)
    cdef _np.ndarray py_array_x1 = _np.asarray(x1,order='C')

    cdef npy_intp Ns = x1.shape[0]
    cdef void * x1_ptr = _np.PyArray_GETPTR1(py_array_x1,0)
    
    cdef bool * where_ptr = NULL
    if where is not None:
        py_array_where = _np.asarray(where,dtype=_np.uint8)
       
        if x1.size!=py_array_where.size:
            raise TypeError("expecting same size for x1 and where input arrays.")
       
        if py_array_where.ndim!=1:
            raise TypeError("where must be a 1d array.")
       
        where_ptr =<bool*> &py_array_where[0] 
    

    if x1.size != x2.size:
        raise TypeError("expecting same size for x1 and x2 input arrays.")
    
    if x1.ndim!=1:
        raise TypeError("x1 must be a 1d array.")

    if x1.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
        raise TypeError("x1 dtype must be one of the possible dtypes used as the representatives for the general basis class.")
   
    if out is None:
        out = _np.zeros(x1.shape,dtype=x1.dtype,order="C")
    else:
        if out.shape!=x1.shape:
            raise TypeError("expecting same shape for out and x1 arrays.")

        if out.dtype not in [uint32,uint64,uint256,uint1024,uint4096,uint16384]:
           raise TypeError("unsupported dtype for variable out. Expecting array of unsigned integers.")

    py_array_out = out

    if py_array_out.ndim!=1:
            raise TypeError("out must be a 1d array.")

    if x1.size != py_array_out.size:
        raise TypeError("expecting same size for x1 and out input arrays.")

    out_ptr = _np.PyArray_GETPTR1(py_array_out,0)

    if x1.dtype == uint32:
        bitwise_right_shift_op_core(<uint32_t*>x1_ptr, &x2[0], where_ptr, <uint32_t*>out_ptr, Ns)
    elif x1.dtype == uint64:
        bitwise_right_shift_op_core(<uint64_t*>x1_ptr, &x2[0], where_ptr, <uint64_t*>out_ptr, Ns )
    elif x1.dtype == uint256:
        bitwise_right_shift_op_core(<uint256_t*>x1_ptr, &x2[0], where_ptr, <uint256_t*>out_ptr, Ns )
    elif x1.dtype == uint1024:
        bitwise_right_shift_op_core(<uint1024_t*>x1_ptr, &x2[0], where_ptr, <uint1024_t*>out_ptr, Ns )
    elif x1.dtype == uint4096:
        bitwise_right_shift_op_core(<uint4096_t*>x1_ptr, &x2[0], where_ptr, <uint4096_t*>out_ptr, Ns )
    elif x1.dtype == uint16384:
        bitwise_right_shift_op_core(<uint16384_t*>x1_ptr, &x2[0], where_ptr, <uint16384_t*>out_ptr, Ns )    
    else:
        raise TypeError("basis dtype {} not recognized.".format(x1.dtype))
    
    return py_array_out

