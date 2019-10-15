# distutils: language=c++


from gmpy2 import popcount

cdef extern from "bitcount.h":
    int bitcount_32_C(NP_UINT32_t,int)
    int bitcount_64_C(NP_UINT64_t,int)


cdef inline NP_INT32_t bit_count(basis_type I,int l):
    if basis_type is NP_UINT32_t:
        return bitcount_32_C(I,l)
    elif basis_type is NP_UINT64_t:
        return bitcount_64_C(I,l)
    else:
        return popcount(I&((<object>(1)<<l)-1))

cdef inline basis_type shift(basis_type I,int shift,int period,NP_INT8_t * sign,basis_type[:] pars):
    # this functino is used to shift the bits of an integer by 'shift' bits.
    # it is used when constructing the momentum states
    cdef int l = (shift+period)%period
    cdef int N1,N2,i
    cdef basis_type I1 = (I >> (period - l))
    cdef basis_type I2 = ((I << l) & pars[2])

    N1 = bit_count(I1,period)
    N2 = bit_count(I2,period)
    sign[0] *= (-1 if (N1&1)&(N2&1)&pars[0] else 1)

    return (I2 | I1)


def py_shift(basis_type[:] x,int d,int length, basis_type[:] pars, NP_INT8_t[:] signs=None):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    cdef NP_INT8_t temp = 0
    if signs is not None:
        for i in range(Ns):
            x[i] = shift(x[i],d,length,&signs[i],pars)
    else:
        for i in range(Ns):
            x[i] = shift(x[i],d,length,&temp,pars)


cdef basis_type fliplr(basis_type I, int length,NP_INT8_t * sign, basis_type[:] pars):
    # this function flips the bits of an integer around the centre, e.g. 1010 -> 0101
    # (generator of) parity symmetry
    cdef basis_type out = 0
    cdef int s = length - 1
    
    cdef int N = bit_count(I,length)
    sign[0] *= (-1 if (N&2) and (pars[0]) else 1)

    out ^= (I&1)
    I >>= 1
    while(I):
        out <<= 1
        out ^= (I&1)
        I >>= 1
        s -= 1

    out <<= s

    return out



def py_fliplr(basis_type[:] x,int length, basis_type[:] pars, NP_INT8_t[:] signs=None):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    cdef NP_INT8_t temp=0
    if signs is not None:
        for i in range(Ns):
            x[i] = fliplr(x[i],length,&signs[i],pars)
    else:
        for i in range(Ns):
            x[i] = fliplr(x[i],length,&temp,pars)    





cdef inline basis_type flip_all(basis_type I, int length,NP_INT8_t * sign,basis_type[:] pars):
    # flip all bits
    ### why do we need the following three lines? where are they used?
    cdef basis_type II = I
    cdef int NA = 0
    cdef int N = 0

    return I^pars[2]


def py_flip_all(basis_type[:] x,int length, basis_type[:] pars, NP_INT8_t[:] signs = None):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    cdef NP_INT8_t temp = 0
    if signs is not None:
        for i in range(Ns):
            x[i] = flip_all(x[i],length,&signs[i],pars)
    else:
        for i in range(Ns):
            x[i] = flip_all(x[i],length,&temp,pars)        




cdef inline basis_type flip_sublat_A(basis_type I, int length,NP_INT8_t * sign,basis_type[:] pars):
    # flip all even bits: sublat A
    return I^pars[3]


def py_flip_sublat_A(basis_type[:] x,int length, basis_type[:] pars, NP_INT8_t[:] signs = None):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    cdef NP_INT8_t temp = 0
    if signs is not None:
        for i in range(Ns):
            x[i] = flip_sublat_A(x[i],length,&signs[i],pars)
    else:
        for i in range(Ns):
            x[i] = flip_sublat_A(x[i],length,&temp,pars)



cdef inline basis_type flip_sublat_B(basis_type I, int length,NP_INT8_t * sign,basis_type[:] pars):
    # flip all odd bits: sublat B
    return I^pars[4]



def py_flip_sublat_B(basis_type[:] x,int length, basis_type[:] pars, NP_INT8_t[:] signs=None):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    cdef NP_INT8_t temp = 0
    if signs is not None:
        for i in range(Ns):
            x[i] = flip_sublat_B(x[i],length,&signs[i],pars)
    else:
        for i in range(Ns):
            x[i] = flip_sublat_B(x[i],length,&temp,pars)        






