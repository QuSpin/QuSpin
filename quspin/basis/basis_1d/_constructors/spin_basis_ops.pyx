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

include "sources/bitops.pyx"
include "sources/refstate.pyx"
include "sources/checkstate.pyx"
include "sources/basis_templates.pyx"
include "sources/op_templates.pyx"

# impliment templates for spins
include "sources/spin_ops.pyx"
include "sources/spin_basis.pyx"















