# cython: language_level=2
import numpy as _np
cimport numpy as _np
cimport cython
from numpy cimport npy_intp
from numpy cimport float32_t,float64_t,complex64_t,complex128_t
from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t


ctypedef fused npy_type:
    int8_t
    int16_t
    int32_t
    int64_t
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    float32_t
    float64_t
    complex64_t
    complex128_t


@cython.boundscheck(False)
@cython.cdivision(True)
def _transpose_array_core(const npy_intp sps,npy_intp[::1] T_tup,npy_type[:,::1] A, npy_type[:,::1] A_T):
    cdef npy_intp i_new,i_old,i,j,r
    cdef npy_intp N_extra_dim = A.shape[0]
    cdef npy_intp Ns = A.shape[1]
    cdef npy_intp M = max(T_tup)+1
    cdef npy_intp nd = T_tup.size
    cdef npy_intp[::1] sps_pow = sps**(_np.arange(M)[::-1])

    with nogil:
        if sps > 2:
            for i in range(Ns):
                j = 0
                for i_old in range(nd):
                    i_new = T_tup[i_old]
                    j += ((i/(sps_pow[i_new])) % sps)*(sps_pow[i_old])

                for r in range(N_extra_dim):
                    A_T[r,j] = A[r,i]

        else:
            for i in range(Ns):
                j = 0
                for i_old in range(nd):
                    i_new = T_tup[i_old]
                    j += ((i>>(M-i_new-1))&1)<<(M-i_old-1)


                for r in range(N_extra_dim):
                    A_T[r,j] = A[r,i]


def _shuffle_sites(npy_intp sps,T_tup,A):
    A = _np.asanyarray(A)
    T_tup = _np.array(T_tup,dtype=_np.intp)


    extra_dim = A.shape[:-1]
    last_dim = A.shape[-1:]
    new_shape = (-1,)+A.shape[-1:]

    A = _np.ascontiguousarray(A)
    A = _np.reshape(A,new_shape,order="C")
    A_T = _np.zeros(A.shape,dtype=A.dtype,order="C")

    _transpose_array_core(sps,T_tup,A,A_T)
    A_T = _np.reshape(A_T,extra_dim+last_dim,order="C")

    return A_T


