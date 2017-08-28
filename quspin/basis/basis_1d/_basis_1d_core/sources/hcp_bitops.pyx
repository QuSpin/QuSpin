

cdef inline basis_type shift(basis_type I,int shift,int period,basis_type[:] pars):
# this functino is used to shift the bits of an integer by 'shift' bits.
# it is used when constructing the momentum states
    cdef int l = (shift+period)%period
    return ((I << l) & pars[2]) | (I >> (period - l))


def py_shift(basis_type[:] x,int d,int length, basis_type[:] pars):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    for i in range(Ns):
        x[i] = shift(x[i],d,length,pars)



cdef NP_INT32_t bit_count(basis_type I, int l, int L):
    cdef NP_INT32_t out = 0
    cdef int i
    for i in range(L-l-1,L-1,1):
        out += ((I >> i) & 1) 

    return out



cdef basis_type fliplr(basis_type I, int length, basis_type[:] pars):
    # this function flips the bits of an integer around the centre, e.g. 1010 -> 0101
    # (generator of) parity symmetry
    cdef basis_type out = 0
    cdef basis_type II = I
    cdef int i,j
    j = length - 1
    for i in range(length):
        out += (II&1) << j
        II >>= 1
        j -= 1
        
    return out



def py_fliplr(basis_type[:] x,int length, basis_type[:] pars):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    for i in range(Ns):
        x[i] = fliplr(x[i],length,pars)





cdef inline basis_type flip_all(basis_type I, int length,basis_type[:] pars):
    # flip all bits
    return I^pars[2]


def py_flip_all(basis_type[:] x,int length, basis_type[:] pars):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    for i in range(Ns):
        x[i] = flip_all(x[i],length,pars)




cdef inline basis_type flip_sublat_A(basis_type I, int length,basis_type[:] pars):
    # flip all even bits: sublat A
    return I^pars[3]


def py_flip_sublat_A(basis_type[:] x,int length, basis_type[:] pars):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    for i in range(Ns):
        x[i] = flip_sublat_A(x[i],length,pars)




cdef inline basis_type flip_sublat_B(basis_type I, int length,basis_type[:] pars):
    # flip all odd bits: sublat B
    return I^pars[4]



def py_flip_sublat_B(basis_type[:] x,int length, basis_type[:] pars):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    for i in range(Ns):
        x[i] = flip_sublat_B(x[i],length,pars)






