
cdef NP_INT32_t bit_count(basis_type I,int l):
    cdef basis_type out = 0
    if basis_type is NP_UINT32_t:
        I &= (0x7FFFFFFF >> (31-l));
        I = I - ((I >> 1) & 0x55555555);
        I = (I & 0x33333333) + ((I >> 2) & 0x33333333);
        return (((I + (I >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;    
    elif basis_type is NP_UINT64_t:
        I &= (0x7FFFFFFFFFFFFFFF >> (63-l));
        I = I - ((I >> 1) & 0x5555555555555555);
        I = (I & 0x3333333333333333) + ((I >> 2) & 0x3333333333333333);
        return (((I + (I >> 4)) & 0x0F0F0F0F0F0F0F0F) * 0x0101010101010101) >> 56;
    else:
        for i in range(l):
            out += (I & 1) 
            I >>= 1

        return out


cdef basis_type shift_single(basis_type I,int shift,int period,NP_INT8_t * sign,basis_type ones):
    # this function is used to shift the bits of an integer by 'shift' bits.
    # it is used when constructing the momentum states
    cdef int l = (shift+period)%period
    cdef int N1,N2,i
    cdef basis_type I1 = (I >> (period - l))
    cdef basis_type I2 = ((I << l) & ones)

    N1 = bit_count(I1,period)
    N2 = bit_count(I2,period)
    sign[0] *= (-1 if (N1&1)&(N2&1) else 1)

    return (I2 | I1)

cdef basis_type shift(basis_type I,int shift,int period,NP_INT8_t * sign,basis_type[:] pars):
    cdef basis_type I_right = I & pars[1]
    cdef basis_type I_left = (I >> period)
    I_left = shift_single(I_left,shift,period,sign,pars[1])
    I_right = shift_single(I_right,shift,period,sign,pars[1])
    return I_right + (I_left << period)


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


cdef basis_type fliplr_single(basis_type I, int length,NP_INT8_t * sign):
    # this function flips the bits of an integer around the centre, e.g. 1010 -> 0101
    # (generator of) parity symmetry
    cdef basis_type out = 0
    cdef int s = length - 1
    cdef int N = bit_count(I,length)
    sign[0] *= (-1 if N&2 else 1)

    out ^= (I&1)
    I >>= 1
    while(I):
        out <<= 1
        out ^= (I&1)
        I >>= 1
        s -= 1

    out <<= s

    return out


cdef fliplr(basis_type I, int length,NP_INT8_t * sign,basis_type[:] pars):
    cdef basis_type I_right = I & pars[1]
    cdef basis_type I_left = (I >> length)
    I_left = fliplr_single(I_left,length,sign)
    I_right = fliplr_single(I_right,length,sign)
    return I_right + (I_left << length)


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
    cdef basis_type I_right = I & pars[1]
    cdef basis_type I_left = (I >> length)
    cdef int N_left,N_right

    N_left = bit_count(I_left,length)
    N_right = bit_count(I_right,length)

    sign[0] *= (-1 if (N_left&1)&(N_right&1) else 1)
    return I_left + (I_right << length)


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

  

cdef flip_sublat_A(basis_type I, int length,NP_INT8_t * sign,basis_type[:] pars):
    return I

cdef flip_sublat_B(basis_type I, int length,NP_INT8_t * sign,basis_type[:] pars):
    return I