import numpy as _np
from scipy.misc import comb

# tells whether or not the inputs into the ops needs Ns or 2*Ns elements
"""
op_array_size={"":1,
				"N":1,
				"Z":1,
				"ZA":1,
				"ZB":1,
				"ZA & ZB":1,
				"N & Z":1,
				"N & ZA":1,
				"N & ZB":1,
				"N & ZA & ZB":1,
				"P":1,
				"N & P":1,
				"PZ":1,
				"N & PZ":1,
				"P & Z":1,
				"N & P & Z":1,
				"T":1,
				"N & T":1,
				"T & Z":1,
				"T & ZA":1,
				"T & ZB":1,
				"T & ZA & ZB":1,
				"N & T & Z":1,
				"N & T & ZA":1,
				"N & T & ZB":1,
				"N & T & ZA & ZB":1,
				"T & P":2,
				"N & T & P":2,
				"T & PZ":2,
				"N & T & PZ":2,
				"T & P & Z":2,
				"N & T & P & Z":2
				}
"""


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
        l=Np//(sps-1)
        s_max = sum((sps-1)*sps**(L-1-i)  for i in range(l))
        s_max += (Np%(sps-1))*sps**(L-l-1)
        dtype = _np.min_scalar_type(int(s_max))
        return _np.result_type(dtype,_np.uint32)


def get_Ns(L, Np, sps, **blocks):
    # this function esimate the size of the hilbert space 
    # here we only estaimte a reduction of there is momentum consrvations
    # as the size of the blocks for partiy are very hard to get for small systems.
    kblock = blocks.get("kblock")
    a = blocks.get("a")
    
    if Np is None:
        Ns = sps**L
    else:
        Ns = H_dim(Np,L,sps-1)

    if kblock is not None:
        # return Ns/L + some extra goes to zero as the system increases. 
        return int((1+1.0/(L//a))*Ns/(L//a)+(L//a))
    else:
        return Ns




