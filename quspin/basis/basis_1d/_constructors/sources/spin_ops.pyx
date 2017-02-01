


cdef int spin_op_func(npy_intp Ns, object[basis_type,ndim=1,mode="c"] basis,
                    str opstr,NP_INT32_t *indx,scalar_type J,
                    object[basis_type,ndim=1,mode="c"] row, matrix_type *ME,object[basis_type,ndim=1,mode="c"] op_pars):

    cdef npy_intp i
    cdef basis_type r,b
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
def spin_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return op_template[basis_type,matrix_type](spin_op_func,None,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_n_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return n_op_template[basis_type,matrix_type](spin_op_func,None,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_p_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    return p_op_template[basis_type,matrix_type](spin_op_func,None,fliplr,None,L,pblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_p_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]
    return p_z_op_template[basis_type,matrix_type](spin_op_func,None,fliplr,flip_all,None,L,pblock,zblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_pz_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pzblock = blocks["pzblock"]
    return pz_op_template[basis_type,matrix_type](spin_op_func,None,fliplr,flip_all,None,L,pzblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_t_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int a = blocks["a"]
    return t_op_template[basis_type,matrix_type](spin_op_func,None,shift,None,L,kblock,a,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def spin_t_p_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[NP_INT8_t,ndim=1] N,
                _np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int a = blocks["a"]

    return t_p_op_template[basis_type,matrix_type](spin_op_func,None,shift,fliplr,None,L,kblock,pblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_t_p_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[NP_INT8_t,ndim=1] N,
                _np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]
    cdef int a = blocks["a"]

    return t_p_z_op_template[basis_type,matrix_type](spin_op_func,None,shift,fliplr,flip_all,None,L,kblock,pblock,zblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_t_pz_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[NP_INT8_t,ndim=1] N,
                _np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pzblock = blocks["pzblock"]
    cdef int a = blocks["a"]

    return t_pz_op_template[basis_type,matrix_type](spin_op_func,None,shift,fliplr,flip_all,None,L,kblock,pzblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_t_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[NP_INT8_t,ndim=1] N,
                _np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zblock = blocks["zblock"]
    cdef int a = blocks["a"]

    return t_z_op_template[basis_type,matrix_type](spin_op_func,None,shift,flip_all,None,L,kblock,zblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_t_zA_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[NP_INT8_t,ndim=1] N,
                _np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["zAblock"]
    cdef int a = blocks["a"]

    return t_zA_op_template[basis_type,matrix_type](spin_op_func,None,shift,flip_sublat_A,None,L,kblock,zAblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_t_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[NP_INT8_t,ndim=1] N,
                _np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zBblock = blocks["zBblock"]
    cdef int a = blocks["a"]

    return t_zB_op_template[basis_type,matrix_type](spin_op_func,None,shift,flip_sublat_B,None,L,kblock,zBblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_t_zA_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[NP_INT8_t,ndim=1] N,
                _np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["zAblock"]
    cdef int zBblock = blocks["zBblock"]
    cdef int a = blocks["a"]

    return t_zA_zB_op_template[basis_type,matrix_type](spin_op_func,None,shift,flip_sublat_A,flip_sublat_B,flip_all,None,L,kblock,zAblock,zBblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zblock = blocks["zblock"]
    return z_op_template[basis_type,matrix_type](spin_op_func,None,flip_all,None,L,zblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_zA_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zAblock = blocks["zAblock"]
    return zA_op_template[basis_type,matrix_type](spin_op_func,None,flip_sublat_A,None,L,zAblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zBblock = blocks["zBblock"]
    return zB_op_template[basis_type,matrix_type](spin_op_func,None,flip_sublat_B,None,L,zBblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])


def spin_zA_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zBblock = blocks["zBblock"]
    cdef int zAblock = blocks["zAblock"]
    return zA_zB_op_template[basis_type,matrix_type](spin_op_func,None,flip_sublat_A,flip_sublat_B,flip_all,None,L,zAblock,zBblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])






