from numpy cimport npy_intp,uint32_t,uint64_t,uint8_t,uint16_t,int8_t,int16_t,int32_t,int64_t
from numpy cimport float32_t,float64_t,complex64_t,complex128_t
from libcpp cimport bool


cdef extern from "general_basis_core.h":
	cdef cppclass general_basis_core[I]:
		const int pers[]
		const int qs[]
		void map_state(I[],npy_intp,int,signed char[]) nogil

cdef extern from "make_general_basis.h":
	npy_intp make_basis[I,J](general_basis_core[I]*,npy_intp,npy_intp,I[], J[]) nogil
	npy_intp make_basis_pcon[I,J](general_basis_core[I]*,npy_intp,npy_intp,I,I[], J[]) nogil

cdef extern from "general_basis_op.h":
	int general_op[I,J,K,T](general_basis_core[I] *B,const int,const char[], const int[],
						  const double complex, const npy_intp, const I[], const J[], K[], K[], T[]) nogil

cdef extern from "general_basis_get_vec.h":
	bool get_vec_general_dense[I,J,T](general_basis_core[I] *B,const I[],const J[],const npy_intp,
									const npy_intp,const npy_intp,const T[],T[]) nogil

ctypedef fused index_type:
	int8_t
	int16_t
	int32_t
	int64_t

ctypedef fused dtype:
	float
	double
	float complex
	double complex

ctypedef fused norm_type:
	uint8_t
	uint16_t
	uint32_t
	uint64_t
