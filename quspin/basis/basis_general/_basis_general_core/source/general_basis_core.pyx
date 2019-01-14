from general_basis_core cimport *
from numpy import pi
from libc.math cimport cos,sin,abs,sqrt
import scipy.sparse as _sp
cimport numpy as _np
import numpy as _np




@cython.boundscheck(False)
cdef get_proj_helper(general_basis_core[state_type] * B, state_type * basis, int nt, int nnt,
                        int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,object P):
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

                P = get_proj_helper(B,basis,nt,nnt-1,sign,c,row,col,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc.real
            else:
                P = get_proj_helper(B,basis,nt,nnt-1,sign,c,row,col,P)
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
                        row[i] = Ns_full - <npy_intp>(basis[i]) - 1

                P = P + _sp.csc_matrix((c,(row,col)),shape=P.shape)

                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i] * cc.real

            else:
                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i]
                        row[i] = Ns_full - <npy_intp>(basis[i]) - 1

                P = P + _sp.csc_matrix((c,(row,col)),shape=P.shape)
                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i] * cc

            with nogil:
                B.map_state(&basis[0],Ns,nt-nnt,&sign[0])


        return P


@cython.boundscheck(False)
cdef get_proj_pcon_helper(general_basis_core[state_type] * B, state_type * basis, int nt, int nnt,
                        int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,state_type * basis_pcon,object P):
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

                P = get_proj_pcon_helper(B,basis,nt,nnt-1,sign,c,row,col,basis_pcon,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc.real
            else:
                P = get_proj_pcon_helper(B,basis,nt,nnt-1,sign,c,row,col,basis_pcon,P)
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
                        row[i] = <npy_intp>binary_search(Ns_full,&basis_pcon[0],basis[i])

                P = P + _sp.csc_matrix((c,(row,col)),shape=P.shape)

                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i] * cc.real

            else:
                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i]
                        row[i] = <npy_intp>binary_search(Ns_full,&basis_pcon[0],basis[i])

                P = P + _sp.csc_matrix((c,(row,col)),shape=P.shape)
                with nogil:
                    for i in range(Ns):
                        c[i] *= sign[i] * cc

            with nogil:
                B.map_state(&basis[0],Ns,nt-nnt,&sign[0])


        return P





"""
overall strategy to avoid copying the classes for each basis type:

simply cast every pointer to a void pointer and when comes time to pass 
memory into low level C functions just do a type check on the 'basis' input.
depending on the basis dtype do an if tree to and then cast the pointers accordingly.
This allows us to use non-standard numpy dtypes stored as np.dtype(np.void,...)
which can't be easily turned into a buffer.
"""


cdef class general_basis_core_wrap:
    cdef int _N
    cdef int _nt
    cdef int _sps
    cdef object _Ns_full
    cdef void * _basis_core

    def __cinit__(self):
        pass

    @cython.boundscheck(False)
    def op(self,index_type[:] row,index_type[:] col,dtype[:] M,object opstr,int[:] indx,object J,_np.ndarray basis,norm_type[:] n):
        cdef char[:] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = basis.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if basis.dtype == _np.uint32:
            with nogil:
                err = general_op(<general_basis_core[uint32_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint32_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        elif basis.dtype == _np.uint64:
            with nogil:
                err = general_op(<general_basis_core[uint64_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint64_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        elif basis.dtype == _np.dtype((_np.void,16)):
            if index_type is int32_t:
                if norm_type is uint8_t:
                    if dtype is float32_t:
                        general_op_wrapper[uint128_t,uint8_t,int32_t,float32_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is float64_t:
                        general_op_wrapper[uint128_t,uint8_t,int32_t,float64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex64_t:
                        general_op_wrapper[uint128_t,uint8_t,int32_t,complex64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex128_t:
                        general_op_wrapper[uint128_t,uint8_t,int32_t,complex128_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])

                elif norm_type is uint16_t:
                    if dtype is float32_t:
                        general_op_wrapper[uint128_t,uint16_t,int32_t,float32_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is float64_t:
                        general_op_wrapper[uint128_t,uint16_t,int32_t,float64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex64_t:
                        general_op_wrapper[uint128_t,uint16_t,int32_t,complex64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex128_t:
                        general_op_wrapper[uint128_t,uint16_t,int32_t,complex128_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])

                elif norm_type is uint32_t:
                    if dtype is float32_t:
                        general_op_wrapper[uint128_t,uint32_t,int32_t,float32_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is float64_t:
                        general_op_wrapper[uint128_t,uint32_t,int32_t,float64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex64_t:
                        general_op_wrapper[uint128_t,uint32_t,int32_t,complex64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex128_t:
                        general_op_wrapper[uint128_t,uint32_t,int32_t,complex128_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])

                elif norm_type is uint64_t:
                    if dtype is float32_t:
                        general_op_wrapper[uint128_t,uint64_t,int32_t,float32_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is float64_t:
                        general_op_wrapper[uint128_t,uint64_t,int32_t,float64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex64_t:
                        general_op_wrapper[uint128_t,uint64_t,int32_t,complex64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex128_t:
                        general_op_wrapper[uint128_t,uint64_t,int32_t,complex128_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])

            elif index_type is int64_t:
                if norm_type is uint8_t:
                    if dtype is float32_t:
                        general_op_wrapper[uint128_t,uint8_t,int64_t,float32_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is float64_t:
                        general_op_wrapper[uint128_t,uint8_t,int64_t,float64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex64_t:
                        general_op_wrapper[uint128_t,uint8_t,int64_t,complex64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex128_t:
                        general_op_wrapper[uint128_t,uint8_t,int64_t,complex128_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])

                elif norm_type is uint16_t:
                    if dtype is float32_t:
                        general_op_wrapper[uint128_t,uint16_t,int64_t,float32_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is float64_t:
                        general_op_wrapper[uint128_t,uint16_t,int64_t,float64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex64_t:
                        general_op_wrapper[uint128_t,uint16_t,int64_t,complex64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex128_t:
                        general_op_wrapper[uint128_t,uint16_t,int64_t,complex128_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])

                elif norm_type is uint32_t:
                    if dtype is float32_t:
                        general_op_wrapper[uint128_t,uint32_t,int64_t,float32_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is float64_t:
                        general_op_wrapper[uint128_t,uint32_t,int64_t,float64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex64_t:
                        general_op_wrapper[uint128_t,uint32_t,int64_t,complex64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex128_t:
                        general_op_wrapper[uint128_t,uint32_t,int64_t,complex128_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])

                elif norm_type is uint64_t:
                    if dtype is float32_t:
                        general_op_wrapper[uint128_t,uint64_t,int64_t,float32_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is float64_t:
                        general_op_wrapper[uint128_t,uint64_t,int64_t,float64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex64_t:
                        general_op_wrapper[uint128_t,uint64_t,int64_t,complex64_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])
                    elif dtype is complex128_t:
                        general_op_wrapper[uint128_t,uint64_t,int64_t,complex128_t](B,n_op,&c_opstr[0],&indx[0],JJ,Ns,basis_ptr,&n[0],&row[0],&col[0],&M[0])

        else:
            raise TypeError("basis dtype must be either uint32 or uint64")

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")

    @cython.boundscheck(False)
    def get_vec_dense(self, _np.ndarray basis, norm_type[:] n, dtype[:,::1] v_in, dtype[:,::1] v_out,_np.ndarray basis_pcon=None):
        cdef npy_intp Ns = v_in.shape[0]
        cdef npy_intp Ns_full = 0
        cdef npy_intp n_vec = v_in.shape[1]
        cdef bool err
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * basis_pcon_ptr = NULL
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if basis_pcon is not None:
            Ns_full = basis_pcon.shape[0]
            basis_pcon_ptr = _np.PyArray_GETPTR1(basis_pcon,0)
        else:
            Ns_full = self._Ns_full


        if basis.dtype == _np.uint32:
            with nogil:
                err = get_vec_general_dense(<general_basis_core[uint32_t]*>B,<uint32_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,<uint32_t*>basis_pcon_ptr,&v_in[0,0],&v_out[0,0])
        elif basis.dtype == _np.uint64:
            with nogil:
                err = get_vec_general_dense(<general_basis_core[uint64_t]*>B,<uint64_t*>basis_ptr,&n[0],n_vec,Ns,Ns_full,<uint64_t*>basis_pcon_ptr,&v_in[0,0],&v_out[0,0])            
        else:
            raise TypeError("basis dtype must be either uint32 or uint64")

        if not err:
            raise TypeError("attemping to use real type for complex elements.")

    @cython.boundscheck(False)
    def get_proj(self, _np.ndarray basis, object Ptype,int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,_np.ndarray basis_pcon = None):
        cdef npy_intp Ns = basis.shape[0]
        cdef npy_intp Ns_full = 0
        cdef object P
        cdef npy_intp i=0
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * basis_pcon_ptr = NULL
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if basis_pcon is not None:
            Ns_full = basis_pcon.shape[0]
            basis_pcon_ptr = _np.PyArray_GETPTR1(basis_pcon,0)
        else:
            Ns_full = self._Ns_full

        if Ns == 0:
            return _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)

        if basis_pcon is None:
            if self._nt <= 0: # no symmetries at all
                if basis.dtype == _np.uint32: 
                    with nogil:
                        for i in range(Ns):
                            row[i] = Ns_full - <npy_intp>(<uint32_t*>basis_ptr)[i] - 1
                elif basis.dtype == _np.uint64:
                    with nogil:
                        for i in range(Ns):
                            row[i] = Ns_full - <npy_intp>(<uint64_t*>basis_ptr)[i] - 1
                else:
                    raise TypeError("Projector index dtype with no particles conservation must be either uint32 or uint64")

                return _sp.csc_matrix((c,(row,col)),shape=(Ns_full,Ns),dtype=Ptype)
            else:
                P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
                if basis.dtype == _np.uint32: 
                    return get_proj_helper[uint32_t,dtype,index_type](<general_basis_core[uint32_t]*>B,<uint32_t*>basis_ptr,self._nt,self._nt,sign,c,row,col,P)
                elif basis.dtype == _np.uint64:
                    return get_proj_helper[uint64_t,dtype,index_type](<general_basis_core[uint64_t]*>B,<uint64_t*>basis_ptr,self._nt,self._nt,sign,c,row,col,P)
                else:
                    raise TypeError("basis dtype must be either uint32 or uint64")  


        else:
            if self._nt <= 0: # basis is already full particle conserving basis
                return _sp.identity(Ns,dtype=Ptype)
            else:
                P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
                if basis.dtype == _np.uint32:
                    return get_proj_pcon_helper[uint32_t,dtype,index_type](<general_basis_core[uint32_t]*>B,<uint32_t*>basis_ptr,self._nt,self._nt,sign,c,row,col,<uint32_t*>basis_pcon_ptr,P)
                elif basis.dtype == _np.uint64:
                    return get_proj_pcon_helper[uint64_t,dtype,index_type](<general_basis_core[uint64_t]*>B,<uint64_t*>basis_ptr,self._nt,self._nt,sign,c,row,col,<uint64_t*>basis_pcon_ptr,P)             
                else:
                    raise TypeError("basis dtype must be either uint32 or uint64")  

    @cython.boundscheck(False)
    def _make_basis_full(self,_np.ndarray basis,norm_type[:] n):
        cdef npy_intp Ns = self._Ns_full
        cdef npy_intp mem_MAX = basis.shape[0]
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if basis.dtype == _np.uint32:
            with nogil:
                Ns = make_basis(<general_basis_core[uint32_t]*>B,Ns,mem_MAX,<uint32_t*>basis_ptr,&n[0])
        elif basis.dtype == _np.uint64:
            with nogil:
                Ns = make_basis(<general_basis_core[uint64_t]*>B,Ns,mem_MAX,<uint64_t*>basis_ptr,&n[0])
        elif basis.dtype == _np.dtype((_np.void,16)):
            if norm_type is uint8_t:
                Ns = make_basis_wrapper[uint128_t,uint8_t](B,Ns,mem_MAX,basis_ptr,&n[0])
            elif norm_type is uint16_t:
                Ns = make_basis_wrapper[uint128_t,uint16_t](B,Ns,mem_MAX,basis_ptr,&n[0])
            elif norm_type is uint32_t:
                Ns = make_basis_wrapper[uint128_t,uint32_t](B,Ns,mem_MAX,basis_ptr,&n[0])
            elif norm_type is uint64_t:
                Ns = make_basis_wrapper[uint128_t,uint64_t](B,Ns,mem_MAX,basis_ptr,&n[0])          
        else:
            raise TypeError("basis dtype must be either uint32 or uint64")  
        return Ns

    @cython.boundscheck(False)
    def _make_basis_pcon(self,object Np,_np.ndarray basis,norm_type[:] n):
        cdef npy_intp Ns = self.get_Ns_pcon(Np)
        cdef uint64_t s  = self.get_s0_pcon(Np)
        cdef npy_intp mem_MAX = basis.shape[0]
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0)
        cdef void * B = self._basis_core

        if basis.dtype == _np.uint32:
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint32_t]*>B,Ns,mem_MAX,<uint32_t>s,<uint32_t*>basis_ptr,&n[0])
        elif basis.dtype == _np.uint64:
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint64_t]*>B,Ns,mem_MAX,<uint64_t>s,<uint64_t*>basis_ptr,&n[0])
        elif basis.dtype == _np.dtype((_np.void,16)):
            if norm_type is uint8_t:
                Ns = make_basis_pcon_wrapper[uint128_t,uint8_t](B,Ns,mem_MAX,s,basis_ptr,&n[0])
            elif norm_type is uint16_t:
                Ns = make_basis_pcon_wrapper[uint128_t,uint16_t](B,Ns,mem_MAX,s,basis_ptr,&n[0])
            elif norm_type is uint32_t:
                Ns = make_basis_pcon_wrapper[uint128_t,uint32_t](B,Ns,mem_MAX,s,basis_ptr,&n[0])
            elif norm_type is uint64_t:
                Ns = make_basis_pcon_wrapper[uint128_t,uint64_t](B,Ns,mem_MAX,s,basis_ptr,&n[0])    
        else:
            raise TypeError("basis dtype must be either uint32 or uint64")  
        return Ns







# cdef class general_basis_core_wrap_32:
#     cdef int _N
#     cdef int _nt
#     cdef int _sps
#     cdef object _Ns_full
#     cdef general_basis_core[uint32_t] * _basis_core

#     def __cinit__(self):
#         pass

#     @cython.boundscheck(False)
#     def op(self,index_type[:] row,index_type[:] col,dtype[:] M,object opstr,int[:] indx,object J,uint32_t[:] basis,norm_type[:] n):
#         cdef char[:] c_opstr = bytearray(opstr,"utf-8")
#         cdef int n_op = indx.shape[0]
#         cdef npy_intp Ns = basis.shape[0]
#         cdef int err = 0;
#         cdef double complex JJ = J
#         with nogil:
#             err = general_op(self._basis_core,n_op,&c_opstr[0],&indx[0],JJ,Ns,&basis[0],&n[0],&row[0],&col[0],&M[0])

#         if err == -1:
#             raise ValueError("operator not recognized.")
#         elif err == 1:
#             raise TypeError("attemping to use real type for complex matrix elements.")

#     @cython.boundscheck(False)
#     def get_vec_dense(self, uint32_t[:] basis, norm_type[:] n, dtype[:,::1] v_in, dtype[:,::1] v_out,uint32_t[:] basis_pcon=None):
#         cdef npy_intp Ns = v_in.shape[0]
#         cdef npy_intp n_vec = v_in.shape[1]
#         cdef bool err
#         cdef uint32_t * ptr = NULL
#         cdef npy_intp Ns_full = 0

#         if basis_pcon is not None:
#             ptr = &basis_pcon[0]
#             Ns_full = basis_pcon.shape[0]
#         else:
#             Ns_full = self._Ns_full

#         with nogil:
#             err = get_vec_general_dense(self._basis_core,&basis[0],&n[0],n_vec,Ns,Ns_full,ptr,&v_in[0,0],&v_out[0,0])

#         if not err:
#             raise TypeError("attemping to use real type for complex elements.")

#     @cython.boundscheck(False)
#     def get_proj(self, uint32_t[:] basis, object Ptype,int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,uint32_t[:] basis_pcon = None):
#         cdef npy_intp Ns = basis.shape[0]
#         cdef npy_intp Ns_full = 0
#         cdef object P
#         cdef npy_intp i=0

#         if basis_pcon is not None:
#             Ns_full = basis_pcon.shape[0]
#         else:
#             Ns_full = self._Ns_full

#         if Ns == 0:
#             return _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)

#         if basis_pcon is None:
#             if self._nt <= 0:
#                 with nogil:
#                     for i in range(Ns):
#                         row[i] = Ns_full-basis[i]-1    

#                 return _sp.csc_matrix((c,(row,col)),shape=(Ns_full,Ns))
#             else:
#                 P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
#                 return get_proj_helper[uint32_t,dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,P)

#         else:
#             if self._nt <= 0: # basis is already just the full particle conserving basis
#                 return _sp.identity(Ns,dtype=Ptype)
#             else:
#                 P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
#                 return get_proj_pcon_helper[uint32_t,dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,basis_pcon,P)           


# cdef class general_basis_core_wrap_64:
#     cdef int _N
#     cdef int _nt
#     cdef int _sps
#     cdef object _Ns_full
#     cdef general_basis_core[uint64_t] * _basis_core

#     def __cinit__(self):
#         pass

#     @cython.boundscheck(False)
#     def op(self,index_type[:] row,index_type[:] col,dtype[:] M,object opstr,int[:] indx,object J,uint64_t[:] basis,norm_type[:] n):
#         cdef char[:] c_opstr = bytearray(opstr,"utf-8")
#         cdef int n_op = indx.shape[0]
#         cdef npy_intp Ns = basis.shape[0]
#         cdef int err = 0;
#         cdef double complex JJ = J
#         with nogil:
#             err = general_op(self._basis_core,n_op,&c_opstr[0],&indx[0],JJ,Ns,&basis[0],&n[0],&row[0],&col[0],&M[0])

#         if err == -1:
#             raise ValueError("operator not recognized.")
#         elif err == 1:
#             raise TypeError("attemping to use real type for complex matrix elements.")

#     @cython.boundscheck(False)
#     def get_vec_dense(self, uint64_t[:] basis, norm_type[:] n, dtype[:,::1] v_in, dtype[:,::1] v_out,uint64_t[:] basis_pcon=None):
#         cdef npy_intp Ns = v_in.shape[0]
#         cdef npy_intp n_vec = v_in.shape[1]
#         cdef bool err
#         cdef uint64_t * ptr = NULL
#         cdef npy_intp Ns_full = 0

#         if basis_pcon is not None:
#             ptr = &basis_pcon[0]
#             Ns_full = basis_pcon.shape[0]
#         else:
#             Ns_full = self._Ns_full

#         with nogil:
#             err = get_vec_general_dense(self._basis_core,&basis[0],&n[0],n_vec,Ns,Ns_full,ptr,&v_in[0,0],&v_out[0,0])

#         if not err:
#             raise TypeError("attemping to use real type for complex elements.")

#     @cython.boundscheck(False)
#     def get_proj(self, uint64_t[:] basis, object Ptype, int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,uint64_t[:] basis_pcon = None):
#         cdef npy_intp Ns = basis.shape[0]
#         cdef npy_intp Ns_full = 0
#         cdef object P
#         cdef npy_intp i=0

#         if basis_pcon is not None:
#             Ns_full = basis_pcon.shape[0]
#         else:
#             Ns_full = self._Ns_full

#         if Ns == 0:
#             return _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)

#         if basis_pcon is None:
#             if self._nt <= 0:
#                 with nogil:
#                     for i in range(Ns):
#                         row[i] = Ns_full-basis[i]-1    

#                 return _sp.csc_matrix((c,(row,col)),shape=(Ns_full,Ns))
#             else:
#                 P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
#                 return get_proj_helper[uint64_t,dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,P)

#         else:
#             if self._nt <= 0: # basis is already just the full particle conserving basis
#                 return _sp.identity(Ns,dtype=Ptype)
#             else:
#                 P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
#                 return get_proj_pcon_helper[uint64_t,dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,basis_pcon,P)   

