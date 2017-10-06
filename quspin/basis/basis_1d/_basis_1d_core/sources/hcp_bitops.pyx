
cdef NP_INT32_t bit_count(basis_type I, int l, int L):
    cdef NP_INT32_t out = 0
    cdef int i

    for i in range(l):
        out += (I & 1) 
        I >>= 1

    return out

cdef inline basis_type shift(basis_type I,int shift,int period,NP_INT8_t * sign,basis_type[:] pars):
    # this functino is used to shift the bits of an integer by 'shift' bits.
    # it is used when constructing the momentum states
    cdef int l = (shift+period)%period
    cdef int N1,N2,i
    cdef basis_type I1,I2

    if pars[0]:
        N1 = N2 = 0
        I1 = I >> (period - l)
        for i in range(period):
            N1 += (I1& 1) 
            I1 >>= 1

        I2 = (I << l) & pars[2]
        for i in range(period):
            N2 += (I2 & 1) 
            I2 >>= 1  
        
        sign[0] *= (-1 if (N1&1)&(N2&1) else 1)


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
    cdef basis_type out = 0
    cdef basis_type II = I
    cdef int i,j,N
    j = length - 1

    if pars[0]:
        N = 0
        for i in range(length):
            N += (II&1)
            out += (II&1) << j
            II >>= 1
            j -= 1
        sign[0] *= (-1 if N&2 else 1)
    else:
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
    cdef basis_type II = I
    cdef int NA = 0
    cdef int N = 0
    if pars[0]:
        for i in range(length):
            N += (i if (II&1) else 0)
            NA += ((II&1) if i&1 else 0)
            II >>= 1

        sign[0] *= (-1 if (NA&1 != N&1) else 1)

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






