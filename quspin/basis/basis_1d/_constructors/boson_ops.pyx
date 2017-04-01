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

include "sources/boson_bitops.pyx"
include "sources/refstate.pyx"
include "sources/op_templates.pyx"
include "sources/boson_ops.pyx"
