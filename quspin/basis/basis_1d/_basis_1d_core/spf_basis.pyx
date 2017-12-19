#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
# distutils: language=c++


cimport numpy as _np
from libc.math cimport cos

from types cimport *


import numpy as _np
from scipy.misc import comb

cdef extern from "glibc_fix.h":
    pass



def H_dim(N,length,m_max):
    """
    Returns the total number of states in the bosonic Hilbert space

    --- arguments:

    N: total number of bosons in lattice
    length: total number of sites
    m_max+1: max number of states per site 
    """

    Ns = 0
    for r in range(N//(m_max+1)+1):
        r_2 = N - r*(m_max+1)
        if r % 2 == 0:
            Ns +=  comb(length,r,exact=True) * comb(length + r_2 - 1,r_2,exact=True)
        else:
            Ns += -comb(length,r,exact=True) * comb(length + r_2 - 1,r_2,exact=True)

    return Ns




def get_basis_type(L, Np, sps, **blocks):
    # calculates the datatype which will fit the largest representative state in basis
    if Np is None:
        # if no particle conservation the largest representative is sps**L
        dtype = _np.min_scalar_type(int(sps**L-1))
        return _np.result_type(dtype,_np.uint32)
    else:
        # if particles are conservated the largest representative is placing all particles as far left
        # as possible.
        Np_left,Np_right = zip(*Np) 
        np_left = max(Np_left)
        np_right = max(Np_right)

        l=np_right//(sps-1)
        s_max_right = sum((sps-1)*sps**(L-1-i)  for i in range(l))
        s_max_right += int( (np_right%(sps-1))*sps**(L-l-1) ) # if l=L need to convert to int

        l=np_left//(sps-1)
        s_max_left = sum((sps-1)*sps**(L-1-i)  for i in range(l))
        s_max_left += int( (np_left%(sps-1))*sps**(L-l-1) ) # if l=L need to convert to int

        s_max = (s_max_left << L) + s_max_right

        dtype = _np.min_scalar_type(int(s_max))
        return _np.result_type(dtype,_np.uint32)


def get_Ns(L, Np, sps, **blocks):
    # this function esimate the size of the hilbert space 
    # here we only estaimte a reduction of there is momentum consrvations
    # as the size of the blocks for partiy are very hard to get for small systems.
    kblock = blocks.get("kblock")
    a = blocks.get("a")
    
    if Np is None:
        Ns = sps**(2*L)
    else:
        Ns=0
        for np_left,np_right in Np: 
            Ns += H_dim(np_left,L,sps-1)*H_dim(np_right,L,sps-1)

    if kblock is not None:
        # return Ns/L + some extra goes to zero as the system increases.
        return int( ( 1+1.0/(L//a) )*Ns/(L//a) + (L//a) )
    else:
        return Ns




include "sources/spf_bitops.pyx"
include "sources/spf_next_state.pyx"
include "sources/checkstate.pyx"
include "sources/basis_templates.pyx"
include "sources/spf_basis.pyx"

