
# cython template, do not call from script
cdef int hcb_op_func(index_type Ns, basis_type *basis,
                    str opstr,NP_INT32_t *indx,scalar_type J,
                    index_type *row, matrix_type *ME,void *op_pars):

    cdef index_type i
    cdef state_type r,b
    cdef int j,error
    cdef int N_indx = len(opstr)
    cdef bool a
    cdef scalar_type M_E
    cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")

    cdef char I = "I"
    cdef char x = "x"
    cdef char y = "y"
    cdef char z = "z"
    cdef char p = "+"
    cdef char m = "-"

    error = 0

    for i in range(Ns): #loop over basis
        M_E = 1.0
        r = basis[i]
        
        for j in range(N_indx-1,-1,-1): #loop over the copstr

            b = ( 1ull << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbil
            a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

            if c_opstr[j] == I:
                continue
            elif c_opstr[j] == z:
                M_E *= (1.0 if a else -1.0)
            elif c_opstr[j] == x:
                r = r ^ b
            elif c_opstr[j] == y:
                r = r ^ b
                M_E *= (1.0j if a else -1.0j)
            elif c_opstr[j] == p:
                M_E *= (0.0 if a else 2.0)
                r = r ^ b
            elif c_opstr[j] == m:
                M_E *= (2.0 if a else 0.0)
                r = r ^ b
            else:
                error = 1
                return error

            if M_E == 0.0:
                break
        M_E *= J
        if matrix_type is float or matrix_type is double or matrix_type is longdouble:
            if M_E.imag != 0.0:
                error = -1
                return error

            ME[i] = M_E.real
            row[i] = r
        else:
            ME[i] = M_E
            row[i] = r

    return error



# operator
def hcb_op(_np.ndarray[index_type,ndim=1] row, _np.ndarray[index_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,**blocks):
    cdef index_type Ns = basis.shape[0]
    return op_template[index_type,basis_type,matrix_type](hcb_op_func,NULL,Ns,&basis[0],opstr,&indx[0],J,&row[0],&col[0],&ME[0])




