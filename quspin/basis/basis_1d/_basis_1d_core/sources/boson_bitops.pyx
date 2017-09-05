"""
This file contains the bit operations used to construct the hcb, spin-1//2, and fermion basis.
Each cython function comes with its python counterpart (used by get_vec)
"""


cdef basis_type shift(basis_type s,int shift,int length,NP_INT8_t * sign,basis_type[:] pars):
    """
    Returns integer representation of shifted bosonic state.

    ---arguments:

    s: integer representation of bosonic state to be shifted
    length: total number of sites
    shift: number of sites the occupations to be shifted by to the right
    sps: number of states per site
    M = [sps**i for i in range(L+1)]
    """
    sign[0] *= 1
    cdef basis_type v = 0
    cdef basis_type[:] M = pars[1:]
    cdef basis_type sps = M[1]
    cdef basis_type m = 1
    cdef int i,j

    for i in range(length):
        j = (i+shift+length)%length
        v += ( s%sps ) * M[j]
        s //= sps

    return v

def py_shift(basis_type[:] x,int d,int length, basis_type[:] pars, NP_INT8_t[:] signs = None):
    cdef npy_intp i 
    cdef npy_intp Ns = x.shape[0]
    cdef NP_INT8_t temp = 0
    if signs is not None:
        for i in range(Ns):
            x[i] = shift(x[i],d,length,&signs[i],pars)
    else:
        for i in range(Ns):
            x[i] = shift(x[i],d,length,&temp,pars)


cdef basis_type fliplr(basis_type s, int length, NP_INT8_t * sign, basis_type[:] pars):
    """
    Returns integer representation of reflected bosonic state wrt middle of chain.

    ---arguments:

    s: integer representation of bosonic state to be shifted
    length: total number of sites
    sps: number of states per site
    """
    sign[0] *= 1
    cdef basis_type v = 0
    cdef basis_type[:] M = pars[1:]
    cdef basis_type sps = M[1]
    cdef int i,j

    for i in range(length):
        j = (length-1) - i
        v += ( s%sps ) * M[j]
        s //= sps

    return v

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




cdef basis_type flip_all(basis_type s, int length,NP_INT8_t * sign,basis_type[:] pars):
    """
    Returns integer representation of bosonic state with complementary site occulation.

    ---arguments:

    s: integer representation of bosonic state to be shifted
    length: total number of sites
    sps: number of states per site
    M = [sps**i for i in range(L)]
    """
    sign[0] *= 1
    cdef basis_type v = 0
    cdef basis_type[:] M = pars[1:]
    cdef basis_type sps = M[1]
    cdef int i

    for i in range(length):
        v += ( sps - s%sps -1 ) * M[i]
        s //= sps

    return v


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




cdef basis_type flip_sublat_A(basis_type s, int length,NP_INT8_t * sign,basis_type[:] pars):
    """
    Returns integer representation of bosonic state with complementary site occulation on sublattice A.

    ---arguments:

    s: integer representation of bosonic state to be shifted
    length: total number of sites
    sps: number of states per site
    M = [sps**i for i in range(L)]
    """
    sign[0] *= 1
    cdef basis_type v = 0
    cdef basis_type[:] M = pars[1:]
    cdef basis_type sps = M[1]
    cdef int i

    for i in range(length):

        if i%2==0: # flip site occupation
            v += ( sps - s%sps -1 ) * M[i]
        else: # shift state by 0 sites
            v += ( s%sps ) * M[i]

        s //= sps

    return v


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



cdef basis_type flip_sublat_B(basis_type s, int length,NP_INT8_t * sign,basis_type[:] pars):
    """
    Returns integer representation of bosonic state with complementary site occulation on sublattice B.

    ---arguments:

    s: integer representation of bosonic state to be shifted
    length: total number of sites
    sps: number of states per site
    M = [sps**i for i in range(L)]
    """
    sign[0] *= 1
    cdef basis_type v = 0
    cdef basis_type[:] M = pars[1:]
    cdef basis_type sps = M[1]
    cdef int i

    for i in range(length):
        
        if i%2==1: # flip site occupation
            v += (sps - s%sps -1 ) * M[i]
        else: # shift state by 0 sites
            v += ( s%sps ) * M[i]

        s //= sps

    return v


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



