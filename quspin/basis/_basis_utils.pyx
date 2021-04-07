# cython: embedsignature=True
# cython: language_level=2
# distutils: language=c++
# cython imports
cimport numpy as _np
cimport cython
from numpy cimport npy_intp,npy_uintp
from numpy cimport float32_t,float64_t,complex64_t,complex128_t
from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t
from libcpp.vector cimport vector
from libcpp cimport bool
# python imports
import numpy as _np


#from general_basis_types cimport *
ctypedef fused state_type:
    uint32_t
    uint64_t


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


cdef extern from "shuffle_sites.h":
    void shuffle_sites_strid[T](const int32_t,const npy_intp*,const int32_t*,const npy_intp,const npy_intp,const T*,T*) nogil

cdef extern from "fermion_ptrace_ordering.h":
    void fermion_ptrace_sign_core[I,P](I, const int[], const int, P&) nogil




@cython.boundscheck(False)
def _shuffle_sites_core(npy_intp[::1] R_tup,int32_t[::1] T_tup,npy_type[:,::1] A, npy_type[:,::1] A_T):

    cdef npy_intp n_row = A.shape[0]
    cdef npy_intp n_col = A.shape[1]
    cdef int32_t nd = T_tup.size
    cdef int32_t * T_tup_ptr = &T_tup[0]
    cdef npy_intp * R_tup_ptr = &R_tup[0]
    cdef npy_type * A_ptr = &A[0,0]
    cdef npy_type * A_T_ptr = &A_T[0,0]

    if nd > 64:
        raise ValueError("can't transpose more than 64 dimensions")

    with nogil:
        shuffle_sites_strid(nd,R_tup_ptr,T_tup_ptr,n_row,n_col,A_ptr,A_T_ptr)






def _reduce_transpose(T_tup,sps=2):
    """
    This function is used to reduce the size of the reshaping and transposing tuples.
    It does this by finding consecutive sites in T_tup and then grouping those indices together
    replacing it with a single index which has a range of values given by sps**(num_sites) where
    'num_sites' is the number of sites within that grouping.

    for example when the subsystem is just a consecutive section starting at site 0, this function suggests that there is no 
    need to do a transpose returning T_tup: (0,) and R_tup: (2**N,).
    """
    T_tup = list(T_tup)

    sub_lists = []
    while(len(T_tup)>0):
        n = len(T_tup)
        sub_list = (T_tup[0],)
        for i in range(n-1):
            if T_tup[i]+1 == T_tup[i+1]:
                sub_list = sub_list + (T_tup[i+1],)
            else:
                break

        for s in sub_list:
            T_tup.remove(s)

        sub_lists.append(sub_list)

    sub_lists_sorted = sorted(sub_lists,key=lambda x:x[0])

    T_tup = []
    R_tup = []
    for sub_list in sub_lists_sorted:
        R_tup.append(sps**len(sub_list))

    for i,sub_list in enumerate(sub_lists):
        j = sub_lists_sorted.index(sub_list)
        T_tup.append(j)

    return tuple(T_tup),tuple(R_tup)



def _shuffle_sites(npy_intp sps,T_tup,A):
    T_tup_reduced,R_tup_reduced = _reduce_transpose(T_tup,sps)

    if len(T_tup_reduced) > 1:

        extra_dim = A.shape[:-1]
        last_dim = A.shape[-1:]
        new_shape = (-1,)+A.shape[-1:]
        A = A.reshape(new_shape,order="C")
        T_tup = _np.array(T_tup_reduced,dtype=_np.int32)
        R_tup = _np.array(R_tup_reduced,dtype=_np.intp)

        if len(T_tup_reduced) < 32: # use numpy as it is much faster
            reshape_tup = A.shape[:1] + tuple(R_tup)
            A = A.reshape(reshape_tup,order="C")
            transpose_tup = (0,)+tuple(T_tup+1)
            A = A.transpose(transpose_tup)
            A_T = A.reshape(extra_dim+last_dim,order="C")
        else: # spill into less efficient C++ code
            A_T = _np.zeros(A.shape,dtype=A.dtype,order="C")
            _shuffle_sites_core(R_tup,T_tup,A,A_T)

        return A_T
    else:
        return _np.ascontiguousarray(A)


@cython.boundscheck(False)
def fermion_ptrace_sign(int N, state_type[::1] states, int8_t[::1] signs, object sub_sys_A):

    sub_sys_B = list(set(range(N))-set(sub_sys_A) )

    cdef int Ns=states.shape[0]
    cdef _np.ndarray map = _np.concatenate([sub_sys_A, sub_sys_B])
    cdef void * ptr = _np.PyArray_GETPTR1(map,0)

    with nogil:
        for j in range(Ns):
            fermion_ptrace_sign_core(states[j], <int*>ptr, N, signs[j])

