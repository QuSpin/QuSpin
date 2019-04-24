# distutils: language=c++
from general_basis_core cimport *
from numpy import pi, array, uint64
from libc.math cimport cos,sin,abs,sqrt
import scipy.sparse as _sp
cimport numpy as _np
import numpy as _np
from libcpp.vector cimport vector
from libcpp.set cimport set

from .general_basis_utils import uint32,uint64,uint256,uint1024,uint4096,uint16384


@cython.boundscheck(False)
cdef get_proj_helper(general_basis_core[npy_uint] * B, npy_uint * basis, int nt, int nnt,
                        int8_t[::1] sign, dtype[::1] c, index_type[::1] indices, index_type[::1] indptr,object P):
    cdef int per = B.pers[nt-nnt]
    cdef npy_intp Ns_full = P.shape[0]
    cdef npy_intp Ns = P.shape[1]

    cdef double q = (2*pi*B.qs[nt-nnt])/per
    cdef double complex cc = cos(q)-1j*sin(q)
    cdef double norm
    cdef npy_intp i,j

    if nnt > 1:
        for j in range(per):
            if dtype is float or dtype is double:
                if abs(cc.imag)>1.1e-15:
                    raise TypeError("attemping to use real type for complex elements.")

                P = get_proj_helper(B,basis,nt,nnt-1,sign,c,indices,indptr,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc.real
            else:
                P = get_proj_helper(B,basis,nt,nnt-1,sign,c,indices,indptr,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc

            with nogil:
                B.map_state(&basis[0],Ns,nt-nnt,&sign[0])

        return P
    else:
        for j in range(per):
            if dtype is float or dtype is double:
                if abs(cc.imag)>1.1e-15:
                    raise TypeError("attemping to use real type for complex elements.")

                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i]
                        indices[i] = Ns_full - <npy_intp>(basis[i]) - 1

                P = P + _sp.csc_matrix((c,indices,indptr),shape=P.shape,copy=False)

                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i] * cc.real

            else:
                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i]
                        indices[i] = Ns_full - <npy_intp>(basis[i]) - 1

                P = P + _sp.csc_matrix((c,indices,indptr),shape=P.shape,copy=False)
                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i] * cc

            with nogil:
                B.map_state(&basis[0],Ns,nt-nnt,&sign[0])


        return P





@cython.boundscheck(False)
cdef get_proj_pcon_helper(general_basis_core[state_type] * B, state_type * basis, int nt, int nnt,
                        int8_t[::1] sign, dtype[::1] c, index_type[::1] indices, index_type[::1] indptr,state_type * basis_pcon,object P):
    cdef int per = B.pers[nt-nnt]
    cdef npy_intp Ns_full = P.shape[0]
    cdef npy_intp Ns = P.shape[1]

    cdef double q = (2*pi*B.qs[nt-nnt])/per
    cdef double complex cc = cos(q)-1j*sin(q)
    cdef double norm
    cdef npy_intp i,j

    if nnt > 1:
        for j in range(per):
            if dtype is float or dtype is double:
                if abs(cc.imag)>1.1e-15:
                    raise TypeError("attemping to use real type for complex elements.")

                P = get_proj_pcon_helper(B,basis,nt,nnt-1,sign,c,indices,indptr,basis_pcon,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc.real
            else:
                P = get_proj_pcon_helper(B,basis,nt,nnt-1,sign,c,indices,indptr,basis_pcon,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc

            with nogil:
                B.map_state(basis,Ns,nt-nnt,&sign[0])

        return P
    else:
        for j in range(per):
            if dtype is float or dtype is double:
                if abs(cc.imag)>1.1e-15:
                    raise TypeError("attemping to use real type for complex elements.")

                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i]
                        indices[i] = <npy_intp>binary_search(Ns_full,basis_pcon,basis[i])

                P = P + _sp.csc_matrix((c,indices,indptr),shape=P.shape,copy=False)

                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i] * cc.real

            else:
                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i]
                        indices[i] = <npy_intp>binary_search(Ns_full,basis_pcon,basis[i])

                P = P + _sp.csc_matrix((c,indices,indptr),shape=P.shape,copy=False)
                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i] * cc

            with nogil:
                B.map_state(basis,Ns,nt-nnt,&sign[0])


        return P








"""
overall strategy to avoid copying the classes for each basis type:

simply cast every pointer to a void pointer and when comes time to pass 
memory into low level C functions just do a type check on the 'basis' input.
depending on the basis dtype do an if tree to and then cast the pointers accordingly.
This allows us to use non-standard numpy dtypes stored as np.dtype(np.void,...)
which can't be easily turned into a buffer.
"""




cdef state_type python_to_basis_int(object python_val, state_type val):
    cdef int i = 0
    val = <state_type>(0)

    while(python_val!=0):
        val = val ^ ((<state_type>(<int>(python_val&1))) << i)
        i += 1
        python_val = python_val >> 1

    return val




cdef set[vector[int]] load_pcon_list(object Np):
    cdef set[vector[int]] Np_cpp
    cdef vector[int] np_cpp

    if type(Np) is int:
        np_cpp.push_back(Np)
        Np_cpp.insert(np_cpp)
        np_cpp.clear()
    else:
        Np_iter = iter(Np)

        for np in Np_iter:
            if type(np) is int:
                np_cpp.push_back(np)
            else:
                np_iter = iter(np)
                for n in np_iter:
                    np_cpp.push_back(n)

            Np_cpp.insert(np_cpp)
            np_cpp.clear()

    return Np_cpp





cdef class general_basis_core_wrap:
    cdef int _N
    cdef int _nt
    cdef int _sps
    cdef object _Ns_full
    cdef void * _basis_core

    def __cinit__(self):
        pass

    @cython.boundscheck(False)
    def op(self,index_type[::1] row,index_type[::1] col,dtype[::1] M,object opstr,int[::1] indx,object J,_np.ndarray basis,norm_type[::1] n):
        cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = basis.shape[0]
        cdef bool basis_full = self._Ns_full == basis.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if not basis.flags["CARRAY"]:
            raise ValueError("basis array must be writable and C-contiguous")

        if basis.dtype == uint32:
            with nogil:
                err = general_op(<general_basis_core[uint32_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,<uint32_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        elif basis.dtype == uint64:
            with nogil:
                err = general_op(<general_basis_core[uint64_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,<uint64_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        elif basis.dtype == uint256:
            with nogil:
                err = general_op(<general_basis_core[uint256_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,<uint256_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        elif basis.dtype == uint1024:
            with nogil:
                err = general_op(<general_basis_core[uint1024_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,<uint1024_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        elif basis.dtype == uint4096:
            with nogil:
                err = general_op(<general_basis_core[uint4096_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,<uint4096_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        elif basis.dtype == uint16384:
            with nogil:
                err = general_op(<general_basis_core[uint16384_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,<uint16384_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        else:
            raise TypeError("basis dtype {} not recognized.".format(basis.dtype))

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")
 
    @cython.boundscheck(False)
    def inplace_op(self,dtype[:,::1] v_in,dtype[:,::1] v_out,bool transposed,bool conjugated,object opstr,int[::1] indx,object J,_np.ndarray basis,norm_type[::1] n):
        cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = basis.shape[0]
        cdef npy_intp nvecs = v_in.shape[1]
        cdef bool basis_full = self._Ns_full == basis.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if not basis.flags["CARRAY"]:
            raise ValueError("basis array must be writable and C-contiguous")

        if basis.dtype == uint32:
            with nogil:
                err = general_inplace_op(<general_basis_core[uint32_t]*>B,transposed,conjugated,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,nvecs,
                                                        <uint32_t*>basis_ptr,&n[0],&v_in[0,0],&v_out[0,0])
        elif basis.dtype == uint64:
            with nogil:
                err = general_inplace_op(<general_basis_core[uint64_t]*>B,transposed,conjugated,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,nvecs,
                                                        <uint64_t*>basis_ptr,&n[0],&v_in[0,0],&v_out[0,0])
        elif basis.dtype == uint256:
            with nogil:
                err = general_inplace_op(<general_basis_core[uint256_t]*>B,transposed,conjugated,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,nvecs,
                                                        <uint256_t*>basis_ptr,&n[0],&v_in[0,0],&v_out[0,0])
        elif basis.dtype == uint1024:
            with nogil:
                err = general_inplace_op(<general_basis_core[uint1024_t]*>B,transposed,conjugated,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,nvecs,
                                                        <uint1024_t*>basis_ptr,&n[0],&v_in[0,0],&v_out[0,0])
        elif basis.dtype == uint4096:
            with nogil:
                err = general_inplace_op(<general_basis_core[uint4096_t]*>B,transposed,conjugated,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,nvecs,
                                                        <uint4096_t*>basis_ptr,&n[0],&v_in[0,0],&v_out[0,0])
        elif basis.dtype == uint16384:
            with nogil:
                err = general_inplace_op(<general_basis_core[uint16384_t]*>B,transposed,conjugated,n_op,&c_opstr[0],&indx[0],JJ,basis_full,Ns,nvecs,
                                                        <uint16384_t*>basis_ptr,&n[0],&v_in[0,0],&v_out[0,0])
        else:
            raise TypeError("basis dtype {} not recognized.".format(basis.dtype))

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")

    @cython.boundscheck(False)
    def get_vec_dense(self, _np.ndarray basis, norm_type[::1] n, dtype[:,::1] v_in, dtype[:,::1] v_out,_np.ndarray basis_pcon=None):
        cdef npy_intp Ns = v_in.shape[0]
        cdef npy_intp Ns_full = 0
        cdef npy_intp n_vec = v_in.shape[1]
        cdef bool err
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * basis_pcon_ptr = NULL
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if not basis.flags["CARRAY"]:
            raise ValueError("basis array must be writable and C-contiguous")

        if basis_pcon is not None:
            Ns_full = basis_pcon.shape[0]

            if not basis_pcon.flags["C_CONTIGUOUS"]:
                raise ValueError("input array must be C-contiguous")

            if basis.dtype != basis_pcon.dtype:
                raise TypeError("basis_pcon must match dtype of the basis.")

            basis_pcon_ptr = _np.PyArray_GETPTR1(basis_pcon,0)
            if basis.dtype == uint32:
                with nogil:
                    err = get_vec_general_pcon_dense(<general_basis_core[uint32_t]*>B,<uint32_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,<uint32_t*>basis_pcon_ptr,&v_in[0,0],&v_out[0,0])
            elif basis.dtype == uint64:
                with nogil:
                    err = get_vec_general_pcon_dense(<general_basis_core[uint64_t]*>B,<uint64_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,<uint64_t*>basis_pcon_ptr,&v_in[0,0],&v_out[0,0])            
            elif basis.dtype == uint256:
                with nogil:
                    err = get_vec_general_pcon_dense(<general_basis_core[uint256_t]*>B,<uint256_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,<uint256_t*>basis_pcon_ptr,&v_in[0,0],&v_out[0,0])
            elif basis.dtype == uint1024:
                with nogil:
                    err = get_vec_general_pcon_dense(<general_basis_core[uint1024_t]*>B,<uint1024_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,<uint1024_t*>basis_pcon_ptr,&v_in[0,0],&v_out[0,0])            
            elif basis.dtype == uint4096:
                with nogil:
                    err = get_vec_general_pcon_dense(<general_basis_core[uint4096_t]*>B,<uint4096_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,<uint4096_t*>basis_pcon_ptr,&v_in[0,0],&v_out[0,0])
            elif basis.dtype == uint16384:
                with nogil:
                    err = get_vec_general_pcon_dense(<general_basis_core[uint16384_t]*>B,<uint16384_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,<uint16384_t*>basis_pcon_ptr,&v_in[0,0],&v_out[0,0])            
            else:
                raise TypeError("basis dtype {} not recognized.".format(basis.dtype))
        else:
            Ns_full = self._Ns_full

            if basis.dtype == uint32:
                with nogil:
                    err = get_vec_general_dense(<general_basis_core[uint32_t]*>B,<uint32_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,&v_in[0,0],&v_out[0,0])
            elif basis.dtype == uint64:
                with nogil:
                    err = get_vec_general_dense(<general_basis_core[uint64_t]*>B,<uint64_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,&v_in[0,0],&v_out[0,0])            
            else:
                raise TypeError("basis dtype {} not recognized.".format(basis.dtype))

        if not err:
            raise TypeError("attemping to use real type for complex elements.")

    @cython.boundscheck(False)
    def get_proj(self, _np.ndarray basis, object Ptype,int8_t[::1] sign, dtype[::1] c, index_type[::1] indices, index_type[::1] indptr,_np.ndarray basis_pcon = None):
        cdef npy_intp Ns = basis.shape[0]
        cdef npy_intp Ns_full = 0
        cdef object P
        cdef npy_intp i=0
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * basis_pcon_ptr = NULL
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting
        cdef uint32_t* uint32_basis_ptr = NULL
        cdef uint64_t* uint64_basis_ptr = NULL

        if not basis.flags["CARRAY"]:
            raise ValueError("basis array must be writable and C-contiguous")

        if basis_pcon is not None:
            Ns_full = basis_pcon.shape[0]

            if not basis_pcon.flags["C_CONTIGUOUS"]:
                raise ValueError("basis array must be C-contiguous")

            if basis.dtype != basis_pcon.dtype:
                raise TypeError("basis_pcon must match dtype of the basis.")

            basis_pcon_ptr = _np.PyArray_GETPTR1(basis_pcon,0)
        else:
            Ns_full = self._Ns_full

        if Ns == 0:
            return _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)

        if basis_pcon is None:
            if self._nt <= 0: # no symmetries at all
                if basis.dtype == uint32: 
                    with nogil:
                        uint32_basis_ptr = <uint32_t*>basis_ptr
                        for i in range(Ns):
                            indices[i] = Ns_full - <npy_intp>uint32_basis_ptr[i] - 1
                elif basis.dtype == uint64:
                    with nogil:
                        uint64_basis_ptr = <uint64_t*>basis_ptr
                        for i in range(Ns):
                            indices[i] = Ns_full - <npy_intp>uint64_basis_ptr[i] - 1
                else:
                    raise TypeError("Projector index dtype with no particles conservation must be either uint32 or uint64")

                return _sp.csc_matrix((c,indices,indptr),shape=(Ns_full,Ns),dtype=Ptype)
            else:
                P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)

                if basis.dtype == uint32: 
                    return get_proj_helper[uint32_t,dtype,index_type](<general_basis_core[uint32_t]*>B,<uint32_t*>basis_ptr,self._nt,self._nt,sign,c,indices,indptr,P)
                elif basis.dtype == uint64:
                    return get_proj_helper[uint64_t,dtype,index_type](<general_basis_core[uint64_t]*>B,<uint64_t*>basis_ptr,self._nt,self._nt,sign,c,indices,indptr,P)
                else:
                    raise TypeError("basis dtype must be either uint32 or uint64")  
        else:
            if self._nt <= 0: # basis is already full particle conserving basis
                return _sp.identity(Ns,dtype=Ptype)
            else:
                P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)

                if basis.dtype == uint32:
                    return get_proj_pcon_helper[uint32_t,dtype,index_type](<general_basis_core[uint32_t]*>B,<uint32_t*>basis_ptr,self._nt,self._nt,
                        sign,c,indices,indptr,<uint32_t*>basis_pcon_ptr,P)
                elif basis.dtype == uint64:
                    return get_proj_pcon_helper[uint64_t,dtype,index_type](<general_basis_core[uint64_t]*>B,<uint64_t*>basis_ptr,self._nt,self._nt,
                        sign,c,indices,indptr,<uint64_t*>basis_pcon_ptr,P)             
                elif basis.dtype == uint256:
                    return get_proj_pcon_helper[uint256_t,dtype,index_type](<general_basis_core[uint256_t]*>B,<uint256_t*>basis_ptr,self._nt,self._nt,
                        sign,c,indices,indptr,<uint256_t*>basis_pcon_ptr,P)             
                elif basis.dtype == uint1024:
                    return get_proj_pcon_helper[uint1024_t,dtype,index_type](<general_basis_core[uint1024_t]*>B,<uint1024_t*>basis_ptr,self._nt,self._nt,
                        sign,c,indices,indptr,<uint1024_t*>basis_pcon_ptr,P)             
                elif basis.dtype == uint4096:
                    return get_proj_pcon_helper[uint4096_t,dtype,index_type](<general_basis_core[uint4096_t]*>B,<uint4096_t*>basis_ptr,self._nt,self._nt,
                        sign,c,indices,indptr,<uint4096_t*>basis_pcon_ptr,P)             
                elif basis.dtype == uint16384:
                    return get_proj_pcon_helper[uint16384_t,dtype,index_type](<general_basis_core[uint16384_t]*>B,<uint16384_t*>basis_ptr,self._nt,self._nt,
                        sign,c,indices,indptr,<uint16384_t*>basis_pcon_ptr,P)             
                else:
                    raise TypeError("basis dtype {} not recognized.".format(basis.dtype))


    @cython.boundscheck(False)
    def _make_basis_full(self,_np.ndarray basis,norm_type[::1] n):
        cdef npy_intp Ns = self._Ns_full
        cdef npy_intp mem_MAX = basis.shape[0]
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if not basis.flags["CARRAY"]:
            raise ValueError("basis array must be writable and C-contiguous")

        if basis.dtype == uint32:
            with nogil:
                Ns = make_basis(<general_basis_core[uint32_t]*>B,Ns,mem_MAX,<uint32_t*>basis_ptr,&n[0])
        elif basis.dtype == uint64:
            with nogil:
                Ns = make_basis(<general_basis_core[uint64_t]*>B,Ns,mem_MAX,<uint64_t*>basis_ptr,&n[0])
        else:
            raise TypeError("basis dtype {} not recognized.".format(basis.dtype))
        return Ns

    @cython.boundscheck(False)
    def _make_basis_pcon(self,object Np,_np.ndarray basis,norm_type[::1] n):
        cdef npy_intp Ns = self.get_Ns_pcon(Np)
        cdef object s  = self.get_s0_pcon(Np)
        cdef npy_intp mem_MAX = basis.shape[0]
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0)
        cdef void * B = self._basis_core
        cdef uint32_t s32 = <uint32_t>(0)
        cdef uint64_t s64 = <uint64_t>(0)
        cdef uint256_t s128 = <uint256_t>(0)
        cdef uint1024_t s256 = <uint1024_t>(0)
        cdef uint4096_t s512 = <uint4096_t>(0)
        cdef uint16384_t s1024 = <uint16384_t>(0)


        if not basis.flags["CARRAY"]:
            raise ValueError("basis array must be writable and C-contiguous")

        if basis.dtype == uint32:
            s32 = python_to_basis_int(s,s32)
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint32_t]*>B,Ns,mem_MAX,s32,<uint32_t*>basis_ptr,&n[0])
        elif basis.dtype == uint64:
            s64 = python_to_basis_int(s,s64)
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint64_t]*>B,Ns,mem_MAX,s64,<uint64_t*>basis_ptr,&n[0])
        elif basis.dtype == uint256:
            s128 = python_to_basis_int(s,s128)
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint256_t]*>B,Ns,mem_MAX,s128,<uint256_t*>basis_ptr,&n[0])
        elif basis.dtype == uint1024:
            s256 = python_to_basis_int(s,s256)
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint1024_t]*>B,Ns,mem_MAX,s256,<uint1024_t*>basis_ptr,&n[0])
        elif basis.dtype == uint4096:
            s512 = python_to_basis_int(s,s512)
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint4096_t]*>B,Ns,mem_MAX,s512,<uint4096_t*>basis_ptr,&n[0])
        elif basis.dtype == uint16384:
            s1024 = python_to_basis_int(s,s1024)
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint16384_t]*>B,Ns,mem_MAX,s1024,<uint16384_t*>basis_ptr,&n[0])
        else:
            raise TypeError("basis dtype {} not recognized.".format(basis.dtype))

        return Ns


    @cython.boundscheck(False)
    def op_bra_ket(self,_np.ndarray ket,_np.ndarray bra,dtype[::1] M,object opstr,int[::1] indx,object J, object Np):
        cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = ket.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        cdef int Npcon_blocks 
        cdef set[vector[int]] Np_set
        cdef void * B = self._basis_core
        cdef void * ket_ptr = _np.PyArray_GETPTR1(ket,0)
        cdef void * bra_ptr = _np.PyArray_GETPTR1(bra,0)

        if not ket.flags["C_CONTIGUOUS"]:
            raise ValueError("key array must be C-contiguous")

        if not bra.flags["CARRAY"]:
            raise ValueError("bra array must be writable and C-contiguous")
        
        if ket.dtype != bra.dtype:
            raise ValueError("ket and bra arrays must have same dtype.")

        if Np is None:
            if ket.dtype == uint32:
                with nogil:
                    err = general_op_bra_ket(<general_basis_core[uint32_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint32_t*>ket_ptr,<uint32_t*>bra_ptr,&M[0])
            elif ket.dtype == uint64:
                with nogil:
                    err = general_op_bra_ket(<general_basis_core[uint64_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint64_t*>ket_ptr,<uint64_t*>bra_ptr,&M[0])
            elif ket.dtype == uint256:
                with nogil:
                    err = general_op_bra_ket(<general_basis_core[uint256_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint256_t*>ket_ptr,<uint256_t*>bra_ptr,&M[0])
            elif ket.dtype == uint1024:
                with nogil:
                    err = general_op_bra_ket(<general_basis_core[uint1024_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint1024_t*>ket_ptr,<uint1024_t*>bra_ptr,&M[0])
            elif ket.dtype == uint4096:
                with nogil:
                    err = general_op_bra_ket(<general_basis_core[uint4096_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint4096_t*>ket_ptr,<uint4096_t*>bra_ptr,&M[0])
            elif ket.dtype == uint16384:
                with nogil:
                    err = general_op_bra_ket(<general_basis_core[uint16384_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint16384_t*>ket_ptr,<uint16384_t*>bra_ptr,&M[0])
            else:
                raise TypeError("basis dtype {} not recognized.".format(ket.dtype))
        else:
            Np_set = load_pcon_list(Np)

            if ket.dtype == uint32:
                with nogil:
                    err = general_op_bra_ket_pcon(<general_basis_core[uint32_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,Np_set,<uint32_t*>ket_ptr,<uint32_t*>bra_ptr,&M[0])
            elif ket.dtype == uint64:
                with nogil:
                    err = general_op_bra_ket_pcon(<general_basis_core[uint64_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,Np_set,<uint64_t*>ket_ptr,<uint64_t*>bra_ptr,&M[0])
            elif ket.dtype == uint256:
                with nogil:
                    err = general_op_bra_ket_pcon(<general_basis_core[uint256_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,Np_set,<uint256_t*>ket_ptr,<uint256_t*>bra_ptr,&M[0])
            elif ket.dtype == uint1024:
                with nogil:
                    err = general_op_bra_ket_pcon(<general_basis_core[uint1024_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,Np_set,<uint1024_t*>ket_ptr,<uint1024_t*>bra_ptr,&M[0])
            elif ket.dtype == uint4096:
                with nogil:
                    err = general_op_bra_ket_pcon(<general_basis_core[uint4096_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,Np_set,<uint4096_t*>ket_ptr,<uint4096_t*>bra_ptr,&M[0])
            elif ket.dtype == uint16384:
                with nogil:
                    err = general_op_bra_ket_pcon(<general_basis_core[uint16384_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,Np_set,<uint16384_t*>ket_ptr,<uint16384_t*>bra_ptr,&M[0])
            else:
                raise TypeError("basis dtype {} not recognized.".format(ket.dtype))

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")

    @cython.boundscheck(False)
    def representative(self,_np.ndarray states,_np.ndarray ref_states,int[:,::1] g_out=None,int8_t[::1] sign_out=None):
        cdef npy_intp Ns = states.shape[0]
        cdef int * g_out_ptr = NULL
        cdef int8_t * sign_out_ptr = NULL
        cdef void * B = self._basis_core
        cdef void * states_ptr = _np.PyArray_GETPTR1(states,0)
        cdef void * ref_states_ptr = _np.PyArray_GETPTR1(ref_states,0)
        
        if not states.flags["C_CONTIGUOUS"]:
            raise ValueError("states array must be C-contiguous")

        if not ref_states.flags["CARRAY"]:
            raise ValueError("out array must be C-contiguous and writable")
        
        if states.dtype != ref_states.dtype:
            raise ValueError("ket and bra arrays must have same dtype.")

        if g_out is not None:
            g_out_ptr = &g_out[0,0]

        if sign_out is not None:
            sign_out_ptr = &sign_out[0]


        if states.dtype == uint32:
            with nogil:
                general_representative(<general_basis_core[uint32_t]*>B,<uint32_t*>states_ptr,<uint32_t*>ref_states_ptr,g_out_ptr,sign_out_ptr,Ns)
        elif states.dtype == uint64:
            with nogil:
                general_representative(<general_basis_core[uint64_t]*>B,<uint64_t*>states_ptr,<uint64_t*>ref_states_ptr,g_out_ptr,sign_out_ptr,Ns)
        elif states.dtype == uint256:
            with nogil:
                general_representative(<general_basis_core[uint256_t]*>B,<uint256_t*>states_ptr,<uint256_t*>ref_states_ptr,g_out_ptr,sign_out_ptr,Ns)
        elif states.dtype == uint1024:
            with nogil:
                general_representative(<general_basis_core[uint1024_t]*>B,<uint1024_t*>states_ptr,<uint1024_t*>ref_states_ptr,g_out_ptr,sign_out_ptr,Ns)
        elif states.dtype == uint4096:
            with nogil:
                general_representative(<general_basis_core[uint4096_t]*>B,<uint4096_t*>states_ptr,<uint4096_t*>ref_states_ptr,g_out_ptr,sign_out_ptr,Ns)
        elif states.dtype == uint16384:
            with nogil:
                general_representative(<general_basis_core[uint16384_t]*>B,<uint16384_t*>states_ptr,<uint16384_t*>ref_states_ptr,g_out_ptr,sign_out_ptr,Ns)
        else:
            raise TypeError("basis dtype {} not recognized.".format(states.dtype))




    @cython.boundscheck(False)
    def normalization(self,_np.ndarray states,norm_type[::1] norms):
        cdef npy_intp Ns = states.shape[0]
        cdef void * B = self._basis_core
        cdef void * states_ptr = _np.PyArray_GETPTR1(states,0)

        if not states.flags["C_CONTIGUOUS"]:
            raise ValueError("states arrays must be C-contiguous")


        if states.dtype == uint32:
            with nogil:
                err = general_normalization(<general_basis_core[uint32_t]*>B,<uint32_t*>states_ptr,&norms[0],Ns)
        elif states.dtype == uint64:
            with nogil:
                err = general_normalization(<general_basis_core[uint64_t]*>B,<uint64_t*>states_ptr,&norms[0],Ns)
        elif states.dtype == uint256:
            with nogil:
                err = general_normalization(<general_basis_core[uint256_t]*>B,<uint256_t*>states_ptr,&norms[0],Ns)
        elif states.dtype == uint1024:
            with nogil:
                err = general_normalization(<general_basis_core[uint1024_t]*>B,<uint1024_t*>states_ptr,&norms[0],Ns)
        elif states.dtype == uint4096:
            with nogil:
                err = general_normalization(<general_basis_core[uint4096_t]*>B,<uint4096_t*>states_ptr,&norms[0],Ns)
        elif states.dtype == uint16384:
            with nogil:
                err = general_normalization(<general_basis_core[uint16384_t]*>B,<uint16384_t*>states_ptr,&norms[0],Ns)
        else:
            raise TypeError("basis dtype {} not recognized.".format(states.dtype))

        if err > 0:
            raise TypeError("normalization values exceeds given data type. Increase data type of signed integer for normalizations array.") 


    @cython.boundscheck(False)
    def get_amp(self,_np.ndarray states,dtype [::1] out,npy_intp Ns,object mode):
        cdef void * B = self._basis_core
        cdef void * states_ptr = _np.PyArray_GETPTR1(states,0)
        

        if mode=='representative':
            if states.dtype == uint32:
                with nogil:
                    err = get_amp_general_light(<general_basis_core[uint32_t]*>B,<uint32_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint64:
                with nogil:
                    err = get_amp_general_light(<general_basis_core[uint64_t]*>B,<uint64_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint256:
                with nogil:
                    err = get_amp_general_light(<general_basis_core[uint256_t]*>B,<uint256_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint1024:
                with nogil:
                    err = get_amp_general_light(<general_basis_core[uint1024_t]*>B,<uint1024_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint4096:
                with nogil:
                    err = get_amp_general_light(<general_basis_core[uint4096_t]*>B,<uint4096_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint16384:
                with nogil:
                    err = get_amp_general_light(<general_basis_core[uint16384_t]*>B,<uint16384_t*>states_ptr,&out[0],Ns)
            else:
                raise TypeError("basis dtype {} not recognized.".format(states.dtype))
                
        elif mode=='full_basis':
            if states.dtype == uint32:
                with nogil:
                    err = get_amp_general(<general_basis_core[uint32_t]*>B,<uint32_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint64:
                with nogil:
                    err = get_amp_general(<general_basis_core[uint64_t]*>B,<uint64_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint256:
                with nogil:
                    err = get_amp_general(<general_basis_core[uint256_t]*>B,<uint256_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint1024:
                with nogil:
                    err = get_amp_general(<general_basis_core[uint1024_t]*>B,<uint1024_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint4096:
                with nogil:
                    err = get_amp_general(<general_basis_core[uint4096_t]*>B,<uint4096_t*>states_ptr,&out[0],Ns)
            elif states.dtype == uint16384:
                with nogil:
                    err = get_amp_general(<general_basis_core[uint16384_t]*>B,<uint16384_t*>states_ptr,&out[0],Ns)
            else:
                raise TypeError("basis dtype {} not recognized.".format(states.dtype))
        else:
            ValueError("mode accepts only the values 'representative' and 'full_basis'.")


        

        if err > 0:
            raise TypeError("attemping to use real type for complex matrix elements.")





