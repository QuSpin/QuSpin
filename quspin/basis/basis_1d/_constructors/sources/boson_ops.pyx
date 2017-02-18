

# cython template[basis_type,matrix_type,N_type], do not call from script
cdef int boson_op_func(npy_intp Ns, object[basis_type,ndim=1,mode="c"] basis,
                    str opstr,NP_INT32_t *indx,scalar_type J,
                    object[basis_type,ndim=1,mode="c"] row, matrix_type *ME,object[basis_type,ndim=1,mode="c"] op_pars):

    cdef npy_intp i
    cdef basis_type r,occ,b
    cdef int j,error
    cdef int N_indx = len(opstr)
    cdef scalar_type M_E
    cdef long double F_c # coefficient coming from bosonic creation operators
    cdef basis_type Nmax = op_pars[2] # max number of particles allowed per site (equals m-1)
    cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")
    cdef int L = op_pars[0]
    cdef object[basis_type,ndim=1,mode="c"] M = op_pars[1:L]

    cdef char I = "I"
    cdef char n = "n"
    cdef char z = "z"
    cdef char p = "+"
    cdef char m = "-"

    error = 0

    for i in range(Ns): #loop over basis
        F_c = 1.0
        r = basis[i]
        
        for j in range(N_indx-1,-1,-1): #loop over the copstr
            b = M[indx[j]]
            occ = (r/b)%(Nmax+1)  #calculate occupation of site ind[j]
            
            # loop over site positions
            if c_opstr[j] == I:
                continue
            elif c_opstr[j] == z: # S^z = n - (m-1)/2 for 2S=2,34,... and m=2S+1
                F_c *=  ( occ-Nmax/2.0 )*( occ-Nmax/2.0 )
            elif c_opstr[j] == n:
                F_c *= occ*occ # square root taken below
            elif c_opstr[j] == p:
                F_c *= (0.0 if occ==Nmax else occ+1)
                r += b
            elif c_opstr[j] == m:
                F_c *= (0.0 if occ==0 else occ)
                r -= b
            else:
                error = 1
                return error

            if F_c == 0.0:
                break

        M_E = J*sqrtl(F_c)

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
def boson_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return op_template[basis_type,matrix_type](boson_op_func,pars,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])

def boson_n_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
              str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
              _np.ndarray[basis_type,ndim=1] basis,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return n_op_template[basis_type,matrix_type](boson_op_func,pars,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])

def boson_p_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]

    return p_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,fliplr,pars,L,pblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_p_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["cblock"]

    return p_z_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,fliplr,flip_all,pars,L,pblock,zblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_pz_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pzblock = blocks["pcblock"]

    return pz_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,fliplr,flip_all,pars,L,pzblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def boson_t_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int a = blocks["a"]

    return t_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,shift,pars,L,kblock,a,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def boson_t_p_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int a = blocks["a"]

    return t_p_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,shift,fliplr,pars,L,kblock,pblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_t_p_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[m_type,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["cblock"]
    cdef int a = blocks["a"]

    return t_p_z_op_template[basis_type,matrix_type,N_type,m_type](boson_op_func,pars,shift,fliplr,flip_all,pars,L,kblock,pblock,zblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_t_pz_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pzblock = blocks["pcblock"]
    cdef int a = blocks["a"]

    return t_pz_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,shift,fliplr,flip_all,pars,L,kblock,pzblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_t_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zblock = blocks["cblock"]
    cdef int a = blocks["a"]

    return t_z_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,shift,flip_all,pars,L,kblock,zblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_t_zA_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["cAblock"]
    cdef int a = blocks["a"]

    return t_zA_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,shift,flip_sublat_A,pars,L,kblock,zAblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_t_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[N_type,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zBblock = blocks["cBblock"]
    cdef int a = blocks["a"]

    return t_zB_op_template[basis_type,matrix_type,N_type](boson_op_func,pars,shift,flip_sublat_B,pars,L,kblock,zBblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_t_zA_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
                str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J, _np.ndarray[N_type,ndim=1] N,
                _np.ndarray[m_type,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["cAblock"]
    cdef int zBblock = blocks["cBblock"]
    cdef int a = blocks["a"]

    return t_zA_zB_op_template[basis_type,matrix_type,N_type,m_type](boson_op_func,pars,shift,flip_sublat_A,flip_sublat_B,flip_all,pars,L,kblock,zAblock,zBblock,a,Ns,&N[0],&m[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_z_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zblock = blocks["cblock"]

    return z_op_template[basis_type,matrix_type](boson_op_func,pars,flip_all,pars,L,zblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])



def boson_zA_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zAblock = blocks["cAblock"]

    return zA_op_template[basis_type,matrix_type](boson_op_func,pars,flip_sublat_A,pars,L,zAblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])


def boson_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zBblock = blocks["cBblock"]

    return zB_op_template[basis_type,matrix_type](boson_op_func,pars,flip_sublat_B,pars,L,zBblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])



def boson_zA_zB_op(_np.ndarray[basis_type,ndim=1] row, _np.ndarray[basis_type,ndim=1] col, _np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,int L,_np.ndarray[basis_type,ndim=1] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zBblock = blocks["cBblock"]
    cdef int zAblock = blocks["cAblock"]

    return zA_zB_op_template[basis_type,matrix_type](boson_op_func,pars,flip_sublat_A,flip_sublat_B,flip_all,pars,L,zAblock,zBblock,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])



