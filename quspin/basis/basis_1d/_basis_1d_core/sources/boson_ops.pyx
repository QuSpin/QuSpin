

# cython template, do not call from script
cdef int op_func(npy_intp Ns, basis_type[:] basis,
                    str opstr,NP_INT32_t *indx,scalar_type J,
                    basis_type[:] row, matrix_type *ME,basis_type[:] op_pars):

    cdef npy_intp i
    cdef basis_type r,occ,b
    cdef int j,error
    cdef int N_indx = len(opstr)
    cdef scalar_type M_E
    cdef double M_E_offdiag, M_E_diag # coefficient coming from bosonic creation operators
    cdef basis_type Nmax = op_pars[2]-1 # max number of particles allowed per site (equals m-1)
    cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")
    cdef int L = op_pars[0]
    cdef basis_type[:] M = op_pars[1:]
    cdef bool spin_me = op_pars[L+2]
    cdef double S = Nmax/2.0

    cdef char I = "I"
    cdef char n = "n"
    cdef char z = "z"
    cdef char p = "+"
    cdef char m = "-"

    error = 0
    for i in range(Ns): #loop over basis
        M_E_offdiag = 1.0
        M_E_diag = 1.0
        r = basis[i]
        
        for j in range(N_indx-1,-1,-1): #loop over the copstr
            b = M[L-indx[j]-1]
            occ = (r/b)%(Nmax+1)  #calculate occupation of site ind[j]
            
            # loop over site positions
            if c_opstr[j] == I:
                continue
            elif c_opstr[j] == z: # S^z = n - (m-1)/2 for 2S=2,3,4,... and m=2S+1
                M_E_diag *= occ-S 
            elif c_opstr[j] == n:
                M_E_diag *= occ # square root taken below
            elif c_opstr[j] == p: # (S-S^z)*(S+S^z+1) = (n_max-n)*(n+1)
                M_E_offdiag *= (occ+1 if occ<Nmax else 0.0)
                M_E_offdiag *= (Nmax-occ if spin_me else 1.0) 
                r   += (b if occ<Nmax else 0)
            elif c_opstr[j] == m:# (S+S^z)*(S-S^z+1) = n*(n_max-n+1)
                M_E_offdiag *= occ
                M_E_offdiag *= (Nmax-occ+1 if spin_me else 1.0)
                r   -= (b if occ>0 else 0)
            else:
                error = 1
                return error

            if M_E_offdiag == 0.0:
                r = basis[i]
                break

        M_E = J*sqrt(M_E_offdiag)*M_E_diag

        if matrix_type is float or matrix_type is double:
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
def op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            basis_type[:] basis,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return op_template(pars,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])

def n_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
              str opstr, NP_INT32_t[:] indx, scalar_type J,
              basis_type[:] basis,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return n_op_template(pars,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])

def p_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]

    return p_op_template(pars,pars,L,pblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def p_z_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]

    return p_z_op_template(pars,pars,L,pblock,zblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def pz_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pzblock = blocks["pzblock"]

    return pz_op_template(pars,pars,L,pzblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def t_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int a = blocks["a"]

    return t_op_template(pars,pars,L,kblock,a,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def t_p_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int a = blocks["a"]

    return t_p_op_template(pars,pars,L,kblock,pblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_p_z_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M2_type[:] M, basis_type[:] basis, int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]
    cdef int a = blocks["a"]

    return t_p_z_op_template(pars,pars,L,kblock,pblock,zblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_pz_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pzblock = blocks["pzblock"]
    cdef int a = blocks["a"]

    return t_pz_op_template(pars,pars,L,kblock,pzblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_z_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zblock = blocks["zblock"]
    cdef int a = blocks["a"]

    return t_z_op_template(pars,pars,L,kblock,zblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zA_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["zAblock"]
    cdef int a = blocks["a"]

    return t_zA_op_template(pars,pars,L,kblock,zAblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zB_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zBblock = blocks["zBblock"]
    cdef int a = blocks["a"]

    return t_zB_op_template(pars,pars,L,kblock,zBblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zA_zB_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M2_type[:] M, basis_type[:] basis, int L,basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["zAblock"]
    cdef int zBblock = blocks["zBblock"]
    cdef int a = blocks["a"]

    return t_zA_zB_op_template(pars,pars,L,kblock,zAblock,zBblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])

def z_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zblock = blocks["zblock"]
    return z_op_template(pars,pars,L,zblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def zA_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zAblock = blocks["zAblock"]
    return zA_op_template(pars,pars,L,zAblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def zB_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zBblock = blocks["zBblock"]
    return zB_op_template(pars,pars,L,zBblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def zA_zB_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int zBblock = blocks["zBblock"]
    cdef int zAblock = blocks["zAblock"]
    return zA_zB_op_template(pars,pars,L,zAblock,zBblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



