# cython: language_level=2
cimport cython
cimport numpy as _np

cdef extern from "glibc_fix.h":
	pass

ctypedef float complex floatcomplex
ctypedef double complex doublecomplex

ctypedef fused indtype:
	_np.int32_t
	_np.int64_t

ctypedef fused T1:
	float
	float complex
	double
	double complex

ctypedef fused T2:
	float
	double

ctypedef fused T3:
	float
	float complex
	double
	double complex

cdef extern from "expm_multiply_parallel.h":
	T csr_trace[I,T](const I,const I,const I[],const I[],const T[]) nogil

	void _expm_multiply[I,T1,T2,T3](const I,const I[],const I[],const T1[],
						  const int,const int,const T2, const T1, const T3, T3[], T3[], T3[]) nogil

@cython.boundscheck(False)
def _wrapper_csr_trace(indtype[:] Ap, indtype[:] Aj, T1[:] Ax):
	cdef indtype n_row = Ap.shape[0] - 1
	return csr_trace(n_row,n_row,&Ap[0],&Aj[0],&Ax[0])	

@cython.boundscheck(False)
def _wrapper_expm_multiply(indtype[:] Ap, indtype[:] Aj, T1[:] Ax,int m_star,int s,
							object a,T2 tol,T1 mu,T3[:] v,T3[:] work):
	cdef indtype n_row = Ap.shape[0] - 1
	cdef int err = 0
	cdef T3 aa = a

	if T1 is float:
		if T3 is float or T3 is floatcomplex:
			if T2 is float:
				with nogil:
					_expm_multiply(n_row,&Ap[0],&Aj[0],&Ax[0],s,m_star,tol,mu,aa,&v[0],&work[0],&work[n_row])
			else:
				raise ValueError
		else:
			raise ValueError
	elif T1 is double:
		if T3 is double or T3 is doublecomplex:
			if T2 is double:
				with nogil:
					_expm_multiply(n_row,&Ap[0],&Aj[0],&Ax[0],s,m_star,tol,mu,aa,&v[0],&work[0],&work[n_row])
			else:
				raise ValueError
		else:
			raise ValueError
	elif T1 is floatcomplex:
		if T3 is floatcomplex:
			if T2 is float:
				with nogil:
					_expm_multiply(n_row,&Ap[0],&Aj[0],&Ax[0],s,m_star,tol,mu,aa,&v[0],&work[0],&work[n_row])
			else:
				raise ValueError
		else:
			raise ValueError
	elif T1 is doublecomplex:
		if T3 is doublecomplex:
			if T2 is double:
				with nogil:
					_expm_multiply(n_row,&Ap[0],&Aj[0],&Ax[0],s,m_star,tol,mu,aa,&v[0],&work[0],&work[n_row])
			else:
				raise ValueError
		else:
			raise ValueError




