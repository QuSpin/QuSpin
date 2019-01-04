from general_basis_core cimport *
from numpy import pi
from libc.math cimport cos,sin,abs,sqrt
import scipy.sparse as _sp

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



@cython.boundscheck(False)
cdef get_proj_pcon_helper_64(general_basis_core[uint64_t] * B, uint64_t[:] basis, int nt, int nnt,
                        int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,uint64_t[:] basis_pcon,object P):
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

                P = get_proj_pcon_helper_64(B,basis,nt,nnt-1,sign,c,row,col,basis_pcon,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc.real
            else:
                P = get_proj_pcon_helper_64(B,basis,nt,nnt-1,sign,c,row,col,basis_pcon,P)
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




@cython.boundscheck(False)
cdef get_proj_pcon_helper_32(general_basis_core[uint32_t] * B, uint32_t[:] basis, int nt, int nnt,
                        int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,uint32_t[:] basis_pcon,object P):
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

                P = get_proj_pcon_helper_32(B,basis,nt,nnt-1,sign,c,row,col,basis_pcon,P)
                with nogil:
                    for i in range(Ns):
                        c[i] *= cc.real
            else:
                P = get_proj_pcon_helper_32(B,basis,nt,nnt-1,sign,c,row,col,basis_pcon,P)
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
        cdef bool err

        with nogil:
            err = get_vec_general_dense(self._basis_core,&basis[0],&n[0],n_vec,Ns,self._Ns_full,&v_in[0,0],&v_out[0,0])

        if not err:
            raise TypeError("attemping to use real type for complex elements.")

    @cython.boundscheck(False)
    def get_proj(self, uint32_t[:] basis, object Ptype,int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,uint32_t[:] basis_pcon = None):
        cdef npy_intp Ns = basis.shape[0]
        cdef npy_intp Ns_full = self._Ns_full
        cdef object P
        cdef npy_intp i=0

        if basis_pcon is not None:
            Ns_full = basis_pcon.shape[0]

        if Ns == 0:
            return _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)

        if basis_pcon is None:
            if self._nt <= 0:
                with nogil:
                    for i in range(Ns):
                        row[i] = Ns_full-basis[i]-1    

                return _sp.csc_matrix((c,(row,col)),shape=(Ns_full,Ns))
            else:
                P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
                return get_proj_helper_32[dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,P)

        else:

            if self._nt <= 0: # basis is already just the full particle conserving basis
                return _sp.identity(Ns,dtype=Ptype)
            else:
                P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
                return get_proj_pcon_helper_32[dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,basis_pcon,P)           


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
    def get_proj(self, uint64_t[:] basis, object Ptype, int8_t[:] sign, dtype[:] c, index_type[:] row, index_type[:] col,uint64_t[:] basis_pcon = None):
        cdef npy_intp Ns = basis.shape[0]
        cdef npy_intp Ns_full = self._Ns_full
        cdef object P
        cdef npy_intp i=0

        if basis_pcon is not None:
            Ns_full = basis_pcon.shape[0]

        if Ns == 0:
            return _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)

        if basis_pcon is None:
            if self._nt <= 0:
                with nogil:
                    for i in range(Ns):
                        row[i] = Ns_full-basis[i]-1    

                return _sp.csc_matrix((c,(row,col)),shape=(Ns_full,Ns))
            else:
                P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
                return get_proj_helper_64[dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,P)

        else:

            if self._nt <= 0: # basis is already just the full particle conserving basis
                return _sp.identity(Ns,dtype=Ptype)
            else:
                P = _sp.csc_matrix((Ns_full,Ns),dtype=Ptype)
                return get_proj_pcon_helper_64[dtype,index_type](self._basis_core,basis,self._nt,self._nt,sign,c,row,col,basis_pcon,P)   

