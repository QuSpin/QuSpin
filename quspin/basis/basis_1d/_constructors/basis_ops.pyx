#!python
#cython: boundscheck=False
#cython: wraparound=False

cimport numpy as _np

import numpy as _np
from scipy.misc import comb
from libc.math cimport sin,cos,abs,sqrt
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString


_np.import_array()

#if hasattr(_np,"float128"):
#	NP_FLOAT128 = _np.float128
#if hasattr(_np,"complex256"):
#	NP_COMPLEX256 = _np.complex256

NP_INT64 = _np.int64
NP_INT32 = _np.int32
NP_INT16 = _np.int16
NP_INT8 = _np.int8

NP_UINT64 = _np.uint64
NP_UINT32 = _np.uint32
NP_UINT16 = _np.uint16
NP_UINT8 = _np.uint8

NP_FLOAT32 = _np.float32
NP_FLOAT64 = _np.float64
NP_COMPLEX64 = _np.complex64
NP_COMPLEX128 = _np.complex128

ctypedef _np.float32_t NP_FLOAT32_t
ctypedef _np.float64_t NP_FLOAT64_t
ctypedef _np.complex64_t NP_COMPLEX64_t
ctypedef _np.complex128_t NP_COMPLEX128_t

ctypedef _np.int64_t NP_INT64_t
ctypedef _np.int32_t NP_INT32_t
ctypedef _np.int16_t NP_INT16_t
ctypedef _np.int8_t NP_INT8_t

ctypedef _np.uint64_t NP_UINT64_t
ctypedef _np.uint32_t NP_UINT32_t
ctypedef _np.uint16_t NP_UINT16_t
ctypedef _np.uint8_t NP_UINT8_t


ctypedef fused index_type:
	NP_INT32_t
	NP_INT64_t

ctypedef fused basis_type:
	NP_UINT32_t
	NP_UINT64_t

ctypedef fused matrix_type:
	NP_FLOAT32_t
	NP_FLOAT64_t
	NP_COMPLEX64_t
	NP_COMPLEX128_t

ctypedef fused complex_type:
	NP_COMPLEX64_t
	NP_COMPLEX128_t

ctypedef _np.uint64_t state_type


ctypedef state_type (*bitop)(state_type, int)
ctypedef state_type (*shifter)(state_type, int, int)
ctypedef state_type (*ns_type)(state_type)
ctypedef int (*op_type)(state_type, basis_type*, int, str, NP_INT32_t*, complex_type,
						index_type*, matrix_type*, complex_type)




# auxillary files
cdef extern from "sources/complex_ops.h":
	pass

include "sources/bitops.pyx"
include "sources/refstate.pyx"
include "sources/checkstate.pyx"
include "sources/spinop.pyx"

# basis sources

include "sources/basis/z_basis.pyx"
include "sources/basis/zA_basis.pyx"
include "sources/basis/zB_basis.pyx"
include "sources/basis/zA_zB_basis.pyx"
include "sources/basis/p_basis.pyx"
include "sources/basis/pz_basis.pyx"
include "sources/basis/p_z_basis.pyx"
include "sources/basis/basis.pyx"
include "sources/basis/t_basis.pyx"
include "sources/basis/t_z_basis.pyx"
include "sources/basis/t_zA_basis.pyx"
include "sources/basis/t_zB_basis.pyx"
include "sources/basis/t_zA_zB_basis.pyx"
include "sources/basis/t_p_basis.pyx"
include "sources/basis/t_pz_basis.pyx"
include "sources/basis/t_p_z_basis.pyx"

# op sources
include "sources/op/spinop.pyx"
include "sources/op/m_op.pyx"
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
include "sources/spin.pyx"

