#!python
#cython: boundscheck=False
#cython: wraparound=False
# distutils: language=c++


cimport numpy as _np
from libc.math cimport sin,cos,sqrt
from numpy.math cimport sinl,cosl,sqrtl

from types cimport *


import numpy as _np
from numpy import array,right_shift,left_shift,invert,bitwise_and,bitwise_xor,bitwise_or
from scipy.misc import comb

_np.import_array()


include "sources/boson_bitops.pyx"
include "sources/checkstate.pyx"
include "sources/basis_templates.pyx"
include "sources/refstate.pyx"
include "sources/op_templates.pyx"

# implement templates for bosons
include "sources/boson_ops.pyx" 
include "sources/boson_basis.pyx"



cdef npy_intp H_dim(int N,int length,int m_max):
    # put in boson basis ops
    """
    Returns the total number of states in the bosonic Hilbert space

    --- arguments:

    N: total number of bosons in lattice
    length: total number of sites
    m_max+1: max number of states per site 
    """

    cdef npy_intp Ns = 0
    cdef int r, r_2

    for r in range(N//(m_max+1)+1):
        r_2 = N - r*(m_max+1)
        if r % 2 == 0:
            Ns +=  comb(length,r,exact=True) * comb(length + r_2 - 1,r_2,exact=True)
        else:
            Ns += -comb(length,r,exact=True) * comb(length + r_2 - 1,r_2,exact=True)

    return Ns




def get_basis_type(int L,object Np,object[basis_type,ndim=1,mode='c'] pars, **blocks):
    # calculates the datatype which will fit the largest representative state in basis
    m=pars[2]
    M=pars[1:] 
    if Np is None:
        # if no particle conservation the largest representative is m**L
        dtype = _np.min_scalar_type(m**L-1)
        return _np.result_type(dtype,_np.uint32)
    else:
        # if particles are conservated the largest representative is placing all particles as far left
        # as possible. 
        l=Np/(m-1)
        s_max = sum((m-1)*M[L-1-i]  for i in range(l))
        s_max += (Np%(m-1))*M[L-l-1]
        dtype = _np.min_scalar_type(s_max)
        return _np.result_type(dtype,_np.uint32)


def get_Ns(int L,object Np, object[basis_type,ndim=1,mode='c'] pars, **blocks):
    # this function esimate the size of the hilbert space 
    # here we only estaimte a reduction of there is momentum consrvations
    # as the size of the blocks for partiy are very hard to get for small systems.
    kblock = blocks.get("kblock")
    a = blocks.get("a")
    m=pars[2]
    
    if Np is None:
        Ns = m**L
    else:
        Ns = H_dim(Np,L,m)


    if kblock is not None:
        # return Ns/L + some extra goes to zero as the system increases. 
        return int((1+1.0/(L//a)**2)*Ns/(L//a)+(L//a))
    else:
        return Ns


