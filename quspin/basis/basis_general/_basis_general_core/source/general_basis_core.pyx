from general_basis_core cimport *
from numpy import pi, array, uint64
from libc.math cimport cos,sin,abs,sqrt
import scipy.sparse as _sp
from libcpp.vector cimport vector
from libcpp.set cimport set

@cython.boundscheck(False)
cdef get_proj_helper_64(general_basis_core[uint64_t] * B, uint64_t[:] basis, int nt, int nnt,
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

                P = get_proj_helper_64(B,basis,nt,nnt-1,sign,c,row,col,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc.real
            else:
                P = get_proj_helper_64(B,basis,nt,nnt-1,sign,c,row,col,P)
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
cdef get_proj_helper_32(general_basis_core[uint32_t] * B, uint32_t[:] basis, int nt, int nnt,
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

                P = get_proj_helper_32(B,basis,nt,nnt-1,sign,c,row,col,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc.real
            else:
                P = get_proj_helper_32(B,basis,nt,nnt-1,sign,c,row,col,P)
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





cdef class general_basis_core_wrap_32:
    cdef int _N
    cdef int _nt
    cdef int _sps
    cdef npy_intp _Ns_full
    cdef general_basis_core[uint32_t] * _basis_core

    def __cinit__(self):
        pass

    @cython.boundscheck(False)
    def op(self,index_type[:] row,index_type[:] col,dtype[:] M,object opstr,int[:] indx,object J,uint32_t[:] basis,norm_type[:] n):
        cdef char[:] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = basis.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        with nogil:
            err = general_op(self._basis_core,n_op,&c_opstr[0],&indx[0],JJ,Ns,&basis[0],&n[0],&row[0],&col[0],&M[0])

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")
    
    @cython.boundscheck(False)
    def get_vec_dense(self, uint32_t[:] basis, norm_type[:] n, dtype[:,::1] v_in, dtype[:,::1] v_out):
        cdef npy_intp Ns = v_in.shape[0]
        cdef npy_intp n_vec = v_in.shape[1]
        cdef npy_intp Ns_full = self._sps**self._N
        cdef bool err

        with nogil:
            err = get_vec_general_dense(self._basis_core,&basis[0],&n[0],n_vec,Ns,Ns_full,&v_in[0,0],&v_out[0,0])

        if not err:
            raise TypeError("attemping to use real type for complex elements.")

    @cython.boundscheck(False)
    def get_proj(self, uint32_t[:] basis, object Ptype,int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col):
        cdef npy_intp Ns = basis.shape[0]
        cdef npy_intp Ns_full = (self._sps**self._N)

        cdef object P = _sp.csr_matrix((Ns_full,Ns),dtype=Ptype)
        if Ns == 0:
            return P

        if self._nt <= 0:
            with nogil:
                for i in range(Ns):
                    row[i] = Ns_full-basis[i]-1    

            P = P + _sp.csr_matrix((c,(row,col)),shape=P.shape)

            return P
        else:
            return get_proj_helper_32[dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,P)

    @cython.boundscheck(False)
    def op_bra_ket(self,uint32_t[:] ket,uint32_t[:] bra,dtype[:] M,object opstr,int[:] indx,object J, object Np):
        cdef char[:] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = ket.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        cdef int Npcon_blocks 
        #cdef unsigned long int[:] Np_array
        cdef set[vector[int]] Np_set
        
        if Np is None:
            with nogil:
                err = general_op_bra_ket(self._basis_core,n_op,&c_opstr[0],&indx[0],JJ,Ns,&ket[0],&bra[0],&M[0])
        else:
            if type(Np) is int:
                Npcon_blocks=1
            else:
                Npcon_blocks=len(Np)
            
            Np_set = load_pcon_list(Np)
            #Np_array=array(Np,ndmin=1,dtype=uint64)

            with nogil:
                err = general_op_bra_ket_pcon(self._basis_core,n_op,&c_opstr[0],&indx[0],JJ,Ns,Npcon_blocks,Np_set,&ket[0],&bra[0],&M[0])

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")

    @cython.boundscheck(False)
    def representative(self,uint32_t[:] states,uint32_t[:] ref_states,int[:,::1] g_out=None,int8_t[:] sign_out=None):
        cdef npy_intp Ns = states.shape[0]
        cdef int * g_out_ptr = NULL
        cdef int8_t * sign_out_ptr = NULL
        
        if g_out is not None:
            g_out_ptr = &g_out[0,0]

        if sign_out is not None:
            sign_out_ptr = &sign_out[0]

        with nogil:
            general_representative(self._basis_core,&states[0],&ref_states[0],g_out_ptr,sign_out_ptr,Ns)


    @cython.boundscheck(False)
    def normalization(self,uint32_t[:] states,norm_type[:] norms):
        cdef npy_intp Ns = states.shape[0]

        with nogil:
            err = general_normalization(self._basis_core,&states[0],&norms[0],Ns)

        if err > 0:
            raise TypeError("normalization values exceeds given data type. Increase data type of signed integer for normalizations array.")	



cdef class general_basis_core_wrap_64:
    cdef int _N
    cdef int _nt
    cdef int _sps
    cdef npy_intp _Ns_full
    cdef general_basis_core[uint64_t] * _basis_core

    def __cinit__(self):
        pass

    @cython.boundscheck(False)
    def op(self,index_type[:] row,index_type[:] col,dtype[:] M,object opstr,int[:] indx,object J,uint64_t[:] basis,norm_type[:] n):
        cdef char[:] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = basis.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        with nogil:
            err = general_op(self._basis_core,n_op,&c_opstr[0],&indx[0],JJ,Ns,&basis[0],&n[0],&row[0],&col[0],&M[0])

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")

    @cython.boundscheck(False)
    def get_vec_dense(self, uint64_t[:] basis, norm_type[:] n, dtype[:,::1] v_in, dtype[:,::1] v_out):
        cdef npy_intp Ns = v_in.shape[0]
        cdef npy_intp n_vec = v_in.shape[1]
        cdef bool err

        with nogil:
            err = get_vec_general_dense(self._basis_core,&basis[0],&n[0],n_vec,Ns,self._Ns_full,&v_in[0,0],&v_out[0,0])

        if not err:
            raise TypeError("attemping to use real type for complex elements.")

    @cython.boundscheck(False)
    def get_proj(self, uint64_t[:] basis, object Ptype, int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col):
        cdef npy_intp Ns = basis.shape[0]

        cdef object P = _sp.csc_matrix((self._Ns_full,Ns),dtype=Ptype)
        if Ns == 0:
            return P

        if self._nt <= 0:
            with nogil:
                for i in range(Ns):
                    row[i] = self._Ns_full-basis[i]-1    

            P = P + _sp.csc_matrix((c,(row,col)),shape=P.shape)

            return P
        else:
            return get_proj_helper_64[dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,P)
    

    @cython.boundscheck(False)
    def op_bra_ket(self,uint64_t[:] ket,uint64_t[:] bra,dtype[:] M,object opstr,int[:] indx,object J, object Np):
        cdef char[:] c_opstr = bytearray(opstr,"utf-8")
        cdef int n_op = indx.shape[0]
        cdef npy_intp Ns = ket.shape[0]
        cdef int err = 0;
        cdef double complex JJ = J
        cdef int Npcon_blocks 
        #cdef unsigned long int[:] Np_array
        cdef set[vector[int]] Np_set
        
        if Np is None:
            with nogil:
                err = general_op_bra_ket(self._basis_core,n_op,&c_opstr[0],&indx[0],JJ,Ns,&ket[0],&bra[0],&M[0])
        else:
            if type(Np) is int:
                Npcon_blocks=1
            else:
                Npcon_blocks=len(Np)
            
            Np_set = load_pcon_list(Np)
            #Np_array=array(Np,ndmin=1,dtype=uint64)

            with nogil:
                err = general_op_bra_ket_pcon(self._basis_core,n_op,&c_opstr[0],&indx[0],JJ,Ns,Npcon_blocks,Np_set,&ket[0],&bra[0],&M[0])

        if err == -1:
            raise ValueError("operator not recognized.")
        elif err == 1:
            raise TypeError("attemping to use real type for complex matrix elements.")

    @cython.boundscheck(False)
    def representative(self,uint64_t[:] states,uint64_t[:] ref_states,int[:,::1] g_out=None,int8_t[:] sign_out=None):
        cdef npy_intp Ns = states.shape[0]
        cdef int * g_out_ptr = NULL
        cdef int8_t * sign_out_ptr = NULL
        
        if g_out is not None:
            g_out_ptr = &g_out[0,0]

        if sign_out is not None:
            sign_out_ptr = &sign_out[0]

        with nogil:
            general_representative(self._basis_core,&states[0],&ref_states[0],g_out_ptr,sign_out_ptr,Ns)

    @cython.boundscheck(False)
    def normalization(self,uint64_t[:] states,norm_type[:] norms):
        cdef npy_intp Ns = states.shape[0]

        with nogil:
            err = general_normalization(self._basis_core,&states[0],&norms[0],Ns)

        if err > 0:
            raise TypeError("normalization values exceeds given data type. Increase data type of signed integer for normalizations array.")	

