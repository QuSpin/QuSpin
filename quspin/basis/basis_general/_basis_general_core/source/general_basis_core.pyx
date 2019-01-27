from general_basis_core cimport *
from numpy import pi, array, uint64
from libc.math cimport cos,sin,abs,sqrt
import scipy.sparse as _sp
cimport numpy as _np
import numpy as _np
from libcpp.vector cimport vector
from libcpp.set cimport set

@cython.boundscheck(False)
cdef get_proj_helper(general_basis_core[state_type] * B, state_type * basis, int nt, int nnt,
                        int8_t[::1] sign, dtype[::1] c, index_type[::1] row, index_type[::1] col,object P):
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
                        int8_t[::1] sign, dtype[::1] c, index_type[::1] row, index_type[::1] col,state_type * basis_pcon,object P):
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
        cdef int err = 0;
        cdef double complex JJ = J
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if not basis.flags["CARRAY"]:
            raise ValueError("input array must be C-contiguous")

        if basis.dtype == _np.uint32:
            with nogil:
                err = general_op(<general_basis_core[uint32_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint32_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        elif basis.dtype == _np.uint64:
            with nogil:
                err = general_op(<general_basis_core[uint64_t]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,<uint64_t*>basis_ptr,&n[0],&row[0],&col[0],&M[0])
        else:
            raise TypeError("basis dtype must be either uint32 or uint64")

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
        cdef int err = 0;
        cdef double complex JJ = J
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if not basis.flags["CARRAY"]:
            raise ValueError("basis array must be C-contiguous")

        if basis.dtype == _np.uint32:
            with nogil:
                err = general_inplace_op(<general_basis_core[uint32_t]*>B,transposed,conjugated,n_op,&c_opstr[0],&indx[0],JJ,Ns,nvecs,
                                                        <uint32_t*>basis_ptr,&n[0],&v_in[0,0],&v_out[0,0])
        elif basis.dtype == _np.uint64:
            with nogil:
                err = general_inplace_op(<general_basis_core[uint64_t]*>B,transposed,conjugated,n_op,&c_opstr[0],&indx[0],JJ,Ns,nvecs,
                                                        <uint64_t*>basis_ptr,&n[0],&v_in[0,0],&v_out[0,0])
        else:
            raise TypeError("basis dtype must be either uint32 or uint64")

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
            raise ValueError("input array must be C-contiguous")

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
    def get_proj(self, _np.ndarray basis, object Ptype,int8_t[::1] sign, dtype[::1] c, index_type[::1] row, index_type[::1] col,_np.ndarray basis_pcon = None):
        cdef npy_intp Ns = basis.shape[0]
        cdef npy_intp Ns_full = 0
        cdef object P
        cdef npy_intp i=0
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * basis_pcon_ptr = NULL
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if not basis.flags["CARRAY"]:
            raise ValueError("input array must be C-contiguous")

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
    def _make_basis_full(self,_np.ndarray basis,norm_type[::1] n):
        cdef npy_intp Ns = self._Ns_full
        cdef npy_intp mem_MAX = basis.shape[0]
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0) # use standard numpy API function
        cdef void * B = self._basis_core # must define local cdef variable to do the pointer casting

        if not basis.flags["CARRAY"]:
            raise ValueError("input array must be C-contiguous")

        if basis.dtype == _np.uint32:
            with nogil:
                Ns = make_basis(<general_basis_core[uint32_t]*>B,Ns,mem_MAX,<uint32_t*>basis_ptr,&n[0])
        elif basis.dtype == _np.uint64:
            with nogil:
                Ns = make_basis(<general_basis_core[uint64_t]*>B,Ns,mem_MAX,<uint64_t*>basis_ptr,&n[0])
        else:
            raise TypeError("basis dtype must be either uint32 or uint64")  
        return Ns

    @cython.boundscheck(False)
    def _make_basis_pcon(self,object Np,_np.ndarray basis,norm_type[::1] n):
        cdef npy_intp Ns = self.get_Ns_pcon(Np)
        cdef uint64_t s  = self.get_s0_pcon(Np)
        cdef npy_intp mem_MAX = basis.shape[0]
        cdef void * basis_ptr = _np.PyArray_GETPTR1(basis,0)
        cdef void * B = self._basis_core

        if not basis.flags["CARRAY"]:
            raise ValueError("input array must be C-contiguous")

        if basis.dtype == _np.uint32:
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint32_t]*>B,Ns,mem_MAX,<uint32_t>s,<uint32_t*>basis_ptr,&n[0])
        elif basis.dtype == _np.uint64:
            with nogil:
                Ns = make_basis_pcon(<general_basis_core[uint64_t]*>B,Ns,mem_MAX,<uint64_t>s,<uint64_t*>basis_ptr,&n[0])
        else:
            raise TypeError("basis dtype must be either uint32 or uint64")  
        return Ns



    ############################################################################ todo



    @cython.boundscheck(False)
    def op_bra_ket(self,state_type[::1] ket,state_type[::1] bra,dtype[::1] M,object opstr,int[::1] indx,object J, object Np):
        cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = ket.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        cdef int Npcon_blocks 
        #cdef unsigned long int[::1] Np_array
        cdef set[vector[int]] Np_set
        cdef void * B = self._basis_core
        
        if Np is None:
            with nogil:
                err = general_op_bra_ket(<general_basis_core[state_type]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,&ket[0],&bra[0],&M[0])
        else:
            if type(Np) is int:
                Npcon_blocks=1
            else:
                Npcon_blocks=len(Np)
            
            Np_set = load_pcon_list(Np)
            #Np_array=array(Np,ndmin=1,dtype=uint64)

            with nogil:
                err = general_op_bra_ket_pcon(<general_basis_core[state_type]*>B,n_op,&c_opstr[0],&indx[0],JJ,Ns,Npcon_blocks,Np_set,&ket[0],&bra[0],&M[0])

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")

    @cython.boundscheck(False)
    def representative(self,state_type[::1] states,state_type[::1] ref_states,int[:,::1] g_out=None,int8_t[::1] sign_out=None):
        cdef npy_intp Ns = states.shape[0]
        cdef int * g_out_ptr = NULL
        cdef int8_t * sign_out_ptr = NULL
        cdef void * B = self._basis_core
        
        if g_out is not None:
            g_out_ptr = &g_out[0,0]

        if sign_out is not None:
            sign_out_ptr = &sign_out[0]

        with nogil:
            general_representative(<general_basis_core[state_type]*>B,&states[0],&ref_states[0],g_out_ptr,sign_out_ptr,Ns)


    @cython.boundscheck(False)
    def normalization(self,state_type[::1] states,norm_type[::1] norms):
        cdef npy_intp Ns = states.shape[0]
        cdef void * B = self._basis_core
        with nogil:
            err = general_normalization(<general_basis_core[state_type]*>B,&states[0],&norms[0],Ns)

        if err > 0:
            raise TypeError("normalization values exceeds given data type. Increase data type of signed integer for normalizations array.") 



