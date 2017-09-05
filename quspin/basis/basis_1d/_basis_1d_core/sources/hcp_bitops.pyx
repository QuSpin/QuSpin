
cdef NP_INT32_t bit_count(basis_type I, int l, int L):
    cdef NP_INT32_t out = 0
    cdef int i

    for i in range(0,l,1):
        out += (I & 1) 
        I >>= 1

    return out

cdef inline basis_type shift(basis_type I,int shift,int period,NP_INT8_t * sign,basis_type[:] pars):
# this functino is used to shift the bits of an integer by 'shift' bits.
# it is used when constructing the momentum states
    sign[0] *= 1
    cdef int l = (shift+period)%period
    return ((I << l) & pars[2]) | (I >> (period - l))


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
    sign[0] *= 1
    cdef basis_type out = 0
    cdef basis_type II = I
    cdef int i,j
    j = length - 1
    for i in range(length):
        out += (II&1) << j
        II >>= 1
        j -= 1
        
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
    sign[0] *= 1
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
    sign[0] *= 1
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
    sign[0] *= 1
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






