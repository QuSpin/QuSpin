# this header file defines all of the types used in the template classes
# distutils: language=c++


cimport numpy as _np
from numpy cimport npy_uintp,npy_intp
from libcpp cimport bool


ctypedef _np.int64_t NP_INT64_t
ctypedef _np.int32_t NP_INT32_t
ctypedef _np.int16_t NP_INT16_t
ctypedef _np.int8_t NP_INT8_t

ctypedef _np.uint64_t NP_UINT64_t
ctypedef _np.uint32_t NP_UINT32_t
ctypedef _np.uint16_t NP_UINT16_t
ctypedef _np.uint8_t NP_UINT8_t

ctypedef fused basis_type:
    NP_UINT32_t
    NP_UINT64_t
    object

ctypedef fused N_type:
    NP_INT8_t
    NP_INT32_t

ctypedef fused M1_type:
    NP_UINT8_t
    NP_UINT32_t

ctypedef fused M2_type:
    NP_UINT16_t
    NP_UINT32_t

ctypedef fused matrix_type:
    float
    double
    float complex
    double complex


    
ctypedef double complex scalar_type


ctypedef basis_type (*ns_type)(basis_type, basis_type[:])
