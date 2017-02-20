#!python
#cython: boundscheck=False
#cython: wraparound=False
# distutils: language=c++


cimport numpy as _np
from libc.math cimport sin,cos,sqrt
from numpy.math cimport sinl,cosl,sqrtl

from types cimport *


import numpy as _np
from scipy.misc import comb

_np.import_array()

include "sources/bitops.pyx"
include "sources/refstate.pyx"
include "sources/checkstate.pyx"
include "sources/basis_templates.pyx"
include "sources/op_templates.pyx"


# impliment templates for spins
include "sources/hcp_ops.pyx"
include "sources/hcp_basis.pyx"


def get_basis_type(L,Np,**blocks):
	if L <= 32:
		return _np.uint32
	elif L <= 64:
		return _np.uint64
	else:
		return _np.object



def get_Ns(L,Np,**blocks):
	kblock = blocks.get("kblock")
	a = blocks.get("a")

	if Np is None:
		Ns = (1<<L)
	else:
		Ns = comb(L,Np,exact=True)


	if kblock is not None:
		return int((1+1.0/(L//a)**2)*Ns/(L//a)+(L//a))
	else:
		return Ns
