
# cython template[basis_type,matrix_type,N_type], do not call from script
cdef int hcp_op_func(npy_intp Ns, object[basis_type,ndim=1,mode="c"] basis,
                    str opstr,NP_INT32_t *indx,scalar_type J,
                    object[basis_type,ndim=1,mode="c"] row, matrix_type *ME,object[basis_type,ndim=1,mode="c"] op_pars):

    cdef npy_intp i
    cdef basis_type r,b
    cdef int j,error,sign
    cdef int N_indx = len(opstr)
    cdef bool a,fermion_op
    cdef scalar_type M_E
    cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")
    cdef basis_type one = 1

    cdef char I = "I" # identity (do nothing)
    cdef char n = "n" # hcb/fermionic number operator
    cdef char x = "x" # S^x spin operator
    cdef char y = "y" # S^y spin operator
    cdef char z = "z" # S^z spin operator, or particle-hole symmetric n operator
    cdef char p = "+" # S^+ or creation operator
    cdef char m = "-" # S^- or annihilation operator

    # parameters passed in much tell whether or not the chain is fermionic
    # 0 -> False
    # 1 -> True
    fermion_op = op_pars[0]
    sign = 1
    error = 0

    for i in range(Ns): #loop over basis
        M_E = 1.0
        r = basis[i]
        
        for j in range(N_indx-1,-1,-1): #loop over the copstr

            b = ( one << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbit
            a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

            # calculate fermionic ME sign if the chain is fermionic
            if fermion_op:
                if bit_count(r,indx[j]) % 2 == 0: # counts number of 1 bits up to and excluding site indx[j]
                    sign=1
                else:
                    sign=-1


            if c_opstr[j] == I:
                continue
            elif c_opstr[j] == z:
                M_E *= (0.5 if a else -0.5)
            elif c_opstr[j] == n:
                M_E *= (1.0 if a else 0.0)
            elif c_opstr[j] == x:
                M_E *= 0.5
                r = r ^ b
            elif c_opstr[j] == y:
                M_E *= (0.5j if a else -0.5j)
                r = r ^ b
            elif c_opstr[j] == p:
                M_E *= sign*(0.0 if a else 1.0)
                r = r ^ b
            elif c_opstr[j] == m:
                M_E *= sign*(1.0 if a else 0.0)
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
def op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return op_template[basis_type,matrix_type](hcp_op_func,pars,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])

def n_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
              str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
              _np.ndarray[basis_type,ndim=1] basis, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return n_op_template[basis_type,matrix_type](hcp_op_func,pars,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])

def p_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    return p_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,fliplr,pars,L,pblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def p_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]
    return p_z_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,fliplr,flip_all,pars,L,pblock,zblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def pz_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pzblock = blocks["pzblock"]
    return pz_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,fliplr,flip_all,pars,L,pzblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def t_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int a = blocks["a"]
    return t_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,shift,pars,L,kblock,a,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def t_p_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] M, _np.ndarray[basis_type,ndim=1] basis, int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int a = blocks["a"]

    return t_p_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,shift,fliplr,pars,L,kblock,pblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_p_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[m_type,ndim=1] M, _np.ndarray[basis_type,ndim=1] basis, int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]
    cdef int a = blocks["a"]

    return t_p_z_op_template[basis_type,matrix_type,N_type,m_type](hcp_op_func,pars,shift,fliplr,flip_all,pars,L,kblock,pblock,zblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_pz_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] M, _np.ndarray[basis_type,ndim=1] basis, int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pzblock = blocks["pzblock"]
    cdef int a = blocks["a"]

    return t_pz_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,shift,fliplr,flip_all,pars,L,kblock,pzblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] M, _np.ndarray[basis_type,ndim=1] basis, int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zblock = blocks["zblock"]
    cdef int a = blocks["a"]

    return t_z_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,shift,flip_all,pars,L,kblock,zblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zA_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] M, _np.ndarray[basis_type,ndim=1] basis, int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["zAblock"]
    cdef int a = blocks["a"]

    return t_zA_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,shift,flip_sublat_A,pars,L,kblock,zAblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] M, _np.ndarray[basis_type,ndim=1] basis, int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zBblock = blocks["zBblock"]
    cdef int a = blocks["a"]

    return t_zB_op_template[basis_type,matrix_type,N_type](hcp_op_func,pars,shift,flip_sublat_B,pars,L,kblock,zBblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zA_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[m_type,ndim=1] M, _np.ndarray[basis_type,ndim=1] basis, int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["zAblock"]
    cdef int zBblock = blocks["zBblock"]
    cdef int a = blocks["a"]

    return t_zA_zB_op_template[basis_type,matrix_type,N_type,m_type](hcp_op_func,pars,shift,flip_sublat_A,flip_sublat_B,flip_all,pars,L,kblock,zAblock,zBblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zblock = blocks["zblock"]
    return z_op_template[basis_type,matrix_type](hcp_op_func,pars,flip_all,pars,L,zblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])



def zA_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zAblock = blocks["zAblock"]
    return zA_op_template[basis_type,matrix_type](hcp_op_func,pars,flip_sublat_A,pars,L,zAblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])



def zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zBblock = blocks["zBblock"]
    return zB_op_template[basis_type,matrix_type](hcp_op_func,pars,flip_sublat_B,pars,L,zBblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])



def zA_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L, _np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zBblock = blocks["zBblock"]
    cdef int zAblock = blocks["zAblock"]
    return zA_zB_op_template[basis_type,matrix_type](hcp_op_func,pars,flip_sublat_A,flip_sublat_B,flip_all,pars,L,zAblock,zBblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])



