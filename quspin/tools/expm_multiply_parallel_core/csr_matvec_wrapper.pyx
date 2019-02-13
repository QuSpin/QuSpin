# cython: language_level=2
# distutils: language=c++
cimport numpy as np
import cython
from cython.parallel cimport parallel
from libcpp cimport bool

ctypedef double complex cdouble
ctypedef float complex cfloat

cdef extern from "csr_matvec.h":
    void csr_matvec[I,T1,T2,T3](const bool,const I,const I[],const I[],const T1[],
                              const T2,const T3[],I[],T3[],T3[]) nogil

ctypedef fused index:
  np.int32_t
  np.int64_t

ctypedef fused T1:
  float
  double
  float complex
  double complex

ctypedef fused T2:
  float
  double
  float complex
  double complex

ctypedef fused T3:
  float
  double
  float complex
  double complex

@cython.boundscheck(False)
def _csr_matvec(bool overwrite_y, index[:] Ap, index[:] Aj,
                  T1[:] Ax, T2 alpha, T3[:] Xx, T3[:] Yx):
  cdef index nr = Yx.shape[0]
  cdef index rco[128];
  cdef T3 vco[128];

  if T1 is cdouble:
    if T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
    else:
      raise ValueError("invalid types")

  elif T1 is double:
    if T2 is cdouble:
      if T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

    elif T2 is double:
      if T3 is double or T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

    elif T2 is cfloat:
      if T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

    else:
      if T3 is double or T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

  elif T1 is cfloat:
    if T2 is cdouble:
      if T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

    elif T2 is double:
      if T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

    elif T2 is cfloat:
      if T3 is cdouble or T3 is cfloat:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

    else:
      if T3 is cdouble or T3 is cfloat:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

  else: #T1 is float
    if T2 is cdouble:
      if T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")  

    elif T2 is double:
      if T3 is double or T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

    elif T2 is cfloat:
      if T3 is cdouble or T3 is cfloat:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")

    else:
      if T3 is cdouble or T3 is cfloat or T3 is double or T3 is cdouble:
        with nogil, parallel():
          csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
      else:
        raise ValueError("invalid types")



  # if T1 is double and T2 is double and T3 is double:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is double and T2 is double and T3 is cdouble:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is double and T2 is cdouble and T3 is cdouble:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is cdouble and T2 is cdouble and T3 is cdouble:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is cdouble and T2 is double and T3 is cdouble:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is float and T2 is float and T3 is float:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is float and T2 is float and T3 is cfloat:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is float and T2 is cfloat and T3 is cfloat:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is cfloat and T2 is cfloat and T3 is cfloat:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # elif T1 is cfloat and T2 is float and T3 is cfloat:
  #   with nogil, parallel():
  #     csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],alpha,&Xx[0],&rco[0],&vco[0],&Yx[0])
  # else:
  #   raise ValueError("invalid types")("imcompatbile types")
