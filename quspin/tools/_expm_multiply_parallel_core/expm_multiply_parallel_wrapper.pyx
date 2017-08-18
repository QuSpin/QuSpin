cimport cython
cimport numpy as _np


ctypedef float complex floatcomplex
ctypedef double complex doublecomplex

ctypedef fused indtype:
	_np.int32_t
	_np.int64_t

ctypedef fused F_types:
	float
	double

ctypedef fused types_1:
	float
	float complex
	double
	double complex

ctypedef fused types_2:
	float
	float complex
	double
	double complex

cdef extern from "expm_multiply_parallel.h":
	T csr_trace[I,T](const I,const I,const I[],const I[],const T[]) nogil

	void _expm_multiply[I,T1,T2,T3](const I,const I,const I[],const I[],const T1[],
						  const int,const int,const T2, const T1, const T3, T3[], T3[], T3[]) nogil

@cython.boundscheck(False)
def _wrapper_csr_trace(indtype[:] Ap, indtype[:] Aj, types_1[:] Ax):
	cdef indtype n_row = Ap.shape[0] - 1
	return csr_trace(n_row,n_row,&Ap[0],&Aj[0],&Ax[0])	

@cython.boundscheck(False)
def _wrapper_expm_multiply(indtype[:] Ap, indtype[:] Aj, types_1[:] Ax,int m_star,int s,
							object a,F_types tol,types_1 mu,types_2[:] v,types_2[:] work):
	cdef indtype n_row = Ap.shape[0] - 1
	cdef int err = 0
	cdef types_2 aa = a

	if types_1 is float:
		if types_2 is float or types_2 is floatcomplex:
			if F_types is float:
				with nogil:
					_expm_multiply(n_row,n_row,&Ap[0],&Aj[0],&Ax[0],s,m_star,tol,mu,aa,&v[0],&work[0],&work[n_row])
			else:
				raise ValueError
		else:
			raise ValueError
	elif types_1 is double:
		if types_2 is double or types_2 is doublecomplex:
			if F_types is double:
				with nogil:
					_expm_multiply(n_row,n_row,&Ap[0],&Aj[0],&Ax[0],s,m_star,tol,mu,aa,&v[0],&work[0],&work[n_row])
			else:
				raise ValueError
		else:
			raise ValueError
	elif types_1 is floatcomplex:
		if types_2 is floatcomplex:
			if F_types is float:
				with nogil:
					_expm_multiply(n_row,n_row,&Ap[0],&Aj[0],&Ax[0],s,m_star,tol,mu,aa,&v[0],&work[0],&work[n_row])
			else:
				raise ValueError
		else:
			raise ValueError
	elif types_1 is doublecomplex:
		if types_2 is doublecomplex:
			if F_types is double:
				with nogil:
					_expm_multiply(n_row,n_row,&Ap[0],&Aj[0],&Ax[0],s,m_star,tol,mu,aa,&v[0],&work[0],&work[n_row])
			else:
				raise ValueError
		else:
			raise ValueError




