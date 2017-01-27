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

# basis sources

include "sources/basis/z_basis.pyx"
include "sources/basis/zA_basis.pyx"
include "sources/basis/zB_basis.pyx"
include "sources/basis/zA_zB_basis.pyx"
include "sources/basis/p_basis.pyx"
include "sources/basis/pz_basis.pyx"
include "sources/basis/p_z_basis.pyx"
include "sources/basis/n_basis.pyx"
include "sources/basis/t_basis.pyx"
include "sources/basis/t_z_basis.pyx"
include "sources/basis/t_zA_basis.pyx"
include "sources/basis/t_zB_basis.pyx"
include "sources/basis/t_zA_zB_basis.pyx"
include "sources/basis/t_p_basis.pyx"
include "sources/basis/t_pz_basis.pyx"
include "sources/basis/t_p_z_basis.pyx"

# op sources
include "sources/op/op.pyx"
include "sources/op/n_op.pyx"
include "sources/op/z_op.pyx"
include "sources/op/zA_op.pyx"
include "sources/op/zB_op.pyx"
include "sources/op/zA_zB_op.pyx"
include "sources/op/p_op.pyx"
include "sources/op/pz_op.pyx"
include "sources/op/p_z_op.pyx"
include "sources/op/t_op.pyx"
include "sources/op/t_z_op.pyx"
include "sources/op/t_zA_op.pyx"
include "sources/op/t_zB_op.pyx"
include "sources/op/t_zA_zB_op.pyx"
include "sources/op/t_p_op.pyx"
include "sources/op/t_pz_op.pyx"
include "sources/op/t_p_z_op.pyx"

# impliment templates for spins
include "sources/spin_ops.pyx"
include "sources/spin_basis.pyx"