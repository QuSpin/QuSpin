# cython: language_level=2
# distutils: language=c++
cimport numpy as np
import cython
from cython.parallel cimport parallel
from libcpp cimport bool
from libcpp.vector cimport vector
import scipy.sparse as _sp
import numpy as _np
from numpy cimport npy_intp

ctypedef double complex cdouble
ctypedef float complex cfloat

cdef extern from "matvec.h":
    void csr_matvec[I,T1,T2](const bool,const I,const I[],const I[],const T1[],
                              const T1,const npy_intp,const T2[],I[],T2[],const npy_intp,T2 []) nogil

    void csc_matvec[I,T1,T2](const bool,const I,const I,const I[],const I[],const T1[],
                              const T1,const npy_intp,const T2[],const npy_intp,T2 []) nogil

    void dia_matvec[I,T1,T2](const bool,const I,const I,const I,const I,const I[],
                              const T1[],const T1,const npy_intp,const T2[],const npy_intp,T2[]) nogil


cdef extern from "matvecs.h":
    void csr_matvecs[I,T1,T2](const bool,const I,const I,const I[],const I[],
                              const T1[],const T1,const npy_intp,const npy_intp,const T2[],
                              const npy_intp,const npy_intp,T2 []) nogil

    void csc_matvecs[I,T1,T2](const bool,const I,const I,const I,const I[],const I[],
                              const T1[],const T1,const npy_intp,const npy_intp,const T2[],
                              const npy_intp,const npy_intp,T2 []) nogil

    void dia_matvecs[I,T1,T2](const bool,const I,const I,const I,const I,const I,
                              const I[],const T1[],const T1,const npy_intp,const npy_intp,const T2[],
                              const npy_intp,const npy_intp,T2 []) nogil



cdef extern from "openmp.h":
  int omp_get_max_threads()



ctypedef fused indtype:
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



@cython.boundscheck(False)   
def _csr_matvec(bool overwrite_y, indtype[::1] Ap, indtype[::1] Aj,T1[::1] Ax, T1 a, T2[:] Xx, T2[:] Yx):
  cdef indtype nr = Yx.shape[0]
  cdef int nthreads = omp_get_max_threads()
  cdef vector[indtype] rco
  cdef vector[T2] vco
  cdef npy_intp ys = Yx.strides[0]/Yx.itemsize;
  cdef npy_intp xs = Xx.strides[0]/Xx.itemsize;

  rco.resize(nthreads)
  vco.resize(nthreads)

  if T1 is cdouble:
    if T2 is cdouble:
      with nogil, parallel():
        csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],a,xs,&Xx[0],&rco[0],&vco[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  elif T1 is double:
    if T2 is cdouble or T2 is double:
      with nogil, parallel():
        csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],a,xs,&Xx[0],&rco[0],&vco[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  elif T1 is cfloat:
    if T2 is cdouble or T2 is cfloat:
      with nogil, parallel():
        csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],a,xs,&Xx[0],&rco[0],&vco[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  else:
    with nogil, parallel():
      csr_matvec(overwrite_y,nr,&Ap[0],&Aj[0],&Ax[0],a,xs,&Xx[0],&rco[0],&vco[0],ys,&Yx[0])

@cython.boundscheck(False)
def _csr_matvecs(bool overwrite_y, indtype[::1] Ap, indtype[::1] Aj,T1[::1] Ax, T1 a, T2[:,:] Xx, T2[:,:] Yx):
  cdef indtype nr = Yx.shape[0]
  cdef indtype nv = Xx.shape[1]
  cdef npy_intp ysr = Yx.strides[0]/Yx.itemsize
  cdef npy_intp ysc = Yx.strides[1]/Yx.itemsize
  cdef npy_intp xsr = Xx.strides[0]/Xx.itemsize
  cdef npy_intp xsc = Xx.strides[1]/Xx.itemsize

  if T1 is cdouble:
    if T2 is cdouble:
      with nogil:
        csr_matvecs(overwrite_y,nr,nv,&Ap[0],&Aj[0],&Ax[0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  elif T1 is double:
    if T2 is cdouble or T2 is double:
      with nogil:
        csr_matvecs(overwrite_y,nr,nv,&Ap[0],&Aj[0],&Ax[0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  elif T1 is cfloat:
    if T2 is cdouble or T2 is cfloat:
      with nogil:
        csr_matvecs(overwrite_y,nr,nv,&Ap[0],&Aj[0],&Ax[0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  else:
    with nogil:
      csr_matvecs(overwrite_y,nr,nv,&Ap[0],&Aj[0],&Ax[0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])

@cython.boundscheck(False)   
def _csc_matvec(bool overwrite_y, indtype[::1] Ap, indtype[::1] Aj,T1[::1] Ax, T1 a, T2[:] Xx, T2[:] Yx):
  cdef indtype nr = Yx.shape[0]
  cdef indtype nc = Xx.shape[0]
  cdef npy_intp ys = Yx.strides[0]/Yx.itemsize;
  cdef npy_intp xs = Xx.strides[0]/Xx.itemsize;

  if T1 is cdouble:
    if T2 is cdouble:
      with nogil,parallel():
        csc_matvec(overwrite_y,nr,nc,&Ap[0],&Aj[0],&Ax[0],a,xs,&Xx[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  elif T1 is double:
    if T2 is cdouble or T2 is double:
      with nogil,parallel():
        csc_matvec(overwrite_y,nr,nc,&Ap[0],&Aj[0],&Ax[0],a,xs,&Xx[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  elif T1 is cfloat:
    if T2 is cdouble or T2 is cfloat:
      with nogil,parallel():
        csc_matvec(overwrite_y,nr,nc,&Ap[0],&Aj[0],&Ax[0],a,xs,&Xx[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  else:
    with nogil,parallel():
      csc_matvec(overwrite_y,nr,nc,&Ap[0],&Aj[0],&Ax[0],a,xs,&Xx[0],ys,&Yx[0])

@cython.boundscheck(False)   
def _csc_matvecs(bool overwrite_y, indtype[::1] Ap, indtype[::1] Aj,T1[::1] Ax, T1 a, T2[:,:] Xx, T2[:,:] Yx):
  cdef indtype nr = Yx.shape[0]
  cdef indtype nc = Xx.shape[0]
  cdef indtype nv = Xx.shape[1]
  cdef npy_intp ysr = Yx.strides[0]/Yx.itemsize
  cdef npy_intp ysc = Yx.strides[1]/Yx.itemsize
  cdef npy_intp xsr = Xx.strides[0]/Xx.itemsize
  cdef npy_intp xsc = Xx.strides[1]/Xx.itemsize

  if T1 is cdouble:
    if T2 is cdouble:
      with nogil:
        csc_matvecs(overwrite_y,nr,nc,nv,&Ap[0],&Aj[0],&Ax[0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  elif T1 is double:
    if T2 is cdouble or T2 is double:
      with nogil:
        csc_matvecs(overwrite_y,nr,nc,nv,&Ap[0],&Aj[0],&Ax[0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  elif T1 is cfloat:
    if T2 is cdouble or T2 is cfloat:
      with nogil:
        csc_matvecs(overwrite_y,nr,nc,nv,&Ap[0],&Aj[0],&Ax[0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  else:
    with nogil:
      csc_matvecs(overwrite_y,nr,nc,nv,&Ap[0],&Aj[0],&Ax[0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])

@cython.boundscheck(False)   
def _dia_matvec(bool overwrite_y, indtype[::1] offsets ,T1[:,::1] diags, T1 a, T2[:] Xx, T2[:] Yx):
  cdef indtype nr = Yx.shape[0]
  cdef indtype nc = Xx.shape[0]
  cdef indtype L = diags.shape[1]
  cdef indtype nd = offsets.shape[0]
  cdef npy_intp ys = Yx.strides[0]/Yx.itemsize;
  cdef npy_intp xs = Xx.strides[0]/Xx.itemsize;

  if T1 is cdouble:
    if T2 is cdouble:
      with nogil,parallel():
        dia_matvec(overwrite_y,nr,nc,nd,L,&offsets[0],&diags[0,0],a,xs,&Xx[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  elif T1 is double:
    if T2 is cdouble or T2 is double:
      with nogil,parallel():
        dia_matvec(overwrite_y,nr,nc,nd,L,&offsets[0],&diags[0,0],a,xs,&Xx[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  elif T1 is cfloat:
    if T2 is cdouble or T2 is cfloat:
      with nogil,parallel():
        dia_matvec(overwrite_y,nr,nc,nd,L,&offsets[0],&diags[0,0],a,xs,&Xx[0],ys,&Yx[0])
    else:
      raise TypeError("invalid types")

  else:
    with nogil,parallel():
      dia_matvec(overwrite_y,nr,nc,nd,L,&offsets[0],&diags[0,0],a,xs,&Xx[0],ys,&Yx[0])

@cython.boundscheck(False)   
def _dia_matvecs(bool overwrite_y, indtype[::1] offsets ,T1[:,::1] diags, T1 a, T2[:,:] Xx, T2[:,:] Yx):
  cdef indtype nr = Yx.shape[0]
  cdef indtype nc = Xx.shape[0]
  cdef indtype nv = Xx.shape[1]
  cdef indtype L = diags.shape[1]
  cdef indtype nd = offsets.shape[0]
  cdef npy_intp ysr = Yx.strides[0]/Yx.itemsize
  cdef npy_intp ysc = Yx.strides[1]/Yx.itemsize
  cdef npy_intp xsr = Xx.strides[0]/Xx.itemsize
  cdef npy_intp xsc = Xx.strides[1]/Xx.itemsize

  if T1 is cdouble:
    if T2 is cdouble:
      with nogil:
        dia_matvecs(overwrite_y,nr,nc,nv,nd,L,&offsets[0],&diags[0,0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  elif T1 is double:
    if T2 is cdouble or T2 is double:
      with nogil:
        dia_matvecs(overwrite_y,nr,nc,nv,nd,L,&offsets[0],&diags[0,0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  elif T1 is cfloat:
    if T2 is cdouble or T2 is cfloat:
      with nogil:
        dia_matvecs(overwrite_y,nr,nc,nv,nd,L,&offsets[0],&diags[0,0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])
    else:
      raise TypeError("invalid types")

  else:
    with nogil:
      dia_matvecs(overwrite_y,nr,nc,nv,nd,L,&offsets[0],&diags[0,0],a,xsr,xsc,&Xx[0,0],ysr,ysc,&Yx[0,0])


def matvec(mat_obj,other,overwrite_out=False,out=None,a=1.0):
  result_dtype = _np.result_type(mat_obj.dtype,other.dtype)

  if out is None:
    overwrite_out = True
    out = _np.zeros(mat_obj.shape[:1]+other.shape[1:],dtype=result_dtype,order="C")

  if _sp.isspmatrix_csr(mat_obj):
    if other.ndim == 1:
      _csr_matvec(overwrite_out,mat_obj.indptr,mat_obj.indices,mat_obj.data,a,other,out)
    else:
      _csr_matvecs(overwrite_out,mat_obj.indptr,mat_obj.indices,mat_obj.data,a,other,out)
   
  elif _sp.isspmatrix_csc(mat_obj):
    if other.ndim == 1:
      _csc_matvec(overwrite_out,mat_obj.indptr,mat_obj.indices,mat_obj.data,a,other,out)
    else:
      _csc_matvecs(overwrite_out,mat_obj.indptr,mat_obj.indices,mat_obj.data,a,other,out)

  elif _sp.isspmatrix_dia(mat_obj):
    if other.ndim == 1:
      _dia_matvec(overwrite_out,mat_obj.offsets,mat_obj.data,a,other,out)
    else:
      _dia_matvecs(overwrite_out,mat_obj.offsets,mat_obj.data,a,other,out)

  else:
    inter_res = mat_obj.dot(other)

    if overwrite_out:
      _np.multiply(a,inter_res,out=out)
    else:
      try:
        inter_res *= a
        out += inter_res
      except TypeError:
        out += a * inter_res

  return out

