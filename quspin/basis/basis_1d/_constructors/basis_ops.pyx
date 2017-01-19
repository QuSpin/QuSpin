#!python
#cython: boundscheck=False
#cython: wraparound=False

cimport numpy as _np

import numpy as _np
from scipy.misc import comb
from libc.math cimport sin,cos,sqrt
from numpy.math cimport sinl,cosl,sqrtl
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from cpython.string cimport PyString_AsString


_np.import_array()


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
	int
#	long
#	long long

ctypedef fused basis_type:
	unsigned int
#	unsigned long
#	unsigned long long

ctypedef fused matrix_type:
	float
	double
#	long double
	float complex
	double complex
#	long double complex
	
ctypedef unsigned long long state_type
ctypedef long double complex scalar_type
ctypedef long double longdouble

ctypedef state_type (*bitop)(state_type, int)
ctypedef state_type (*shifter)(state_type, int, int)
ctypedef state_type (*ns_type)(state_type)
ctypedef int (*op_type)(index_type, basis_type*, str, NP_INT32_t*,scalar_type,
						index_type*, matrix_type*)


# tells whether or not the inputs into the ops needs Ns or 2*Ns elements
op_array_size={"":1,
				"M":1,
				"Z":1,
				"ZA":1,
				"ZB":1,
				"ZA & ZB":1,
				"M & Z":_1,
				"M & ZA":_1,
				"M & ZB":1,
				"M & ZA & ZB":1,
				"P":1,
				"M & P":1,
				"PZ":1,
				"M & PZ":1,
				"P & Z":1,
				"M & P & Z":1,
				"T":1,
				"M & T":1,
				"T & Z":1,
				"T & ZA":1,
				"T & ZB":1,
				"T & ZA & ZB":1,
				"M & T & Z":1,
				"M & T & ZA":1,
				"M & T & ZB":1,
				"M & T & ZA & ZB":1,
				"T & P":2,
				"M & T & P":2,
				"T & PZ":2,
				"M & T & PZ":2,
				"T & P & Z":2,
				"M & T & P & Z":2
				}




# auxillary files
cdef extern from "sources/complex_ops.h":
	pass

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
include "sources/spin.pyx"

