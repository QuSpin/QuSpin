
# cython template, do not call from script
cdef int op_func(npy_intp Ns, basis_type[:] basis,
                    str opstr,NP_INT32_t *indx,scalar_type J,
                    basis_type[:] row, matrix_type *ME,basis_type[:] op_pars):

    cdef npy_intp i
    cdef basis_type r,b
    cdef int j,error,sign,i_op
    cdef int L = 2*op_pars[0] # factor of 2 for spinful fermions (effectively double lattice sites)
    cdef int N_indx = len(opstr)
    cdef bool a
    cdef scalar_type M_E
    cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")
    cdef basis_type one = 1

    cdef char I = "I" # identity (do nothing)
    cdef char n = "n" # hcb/fermionic number operator
    cdef char z = "z" # S^z spin operator, or particle-hole symmetric n operator
    cdef char p = "+" # S^+ or creation operator
    cdef char m = "-" # S^- or annihilation operator

    error = 0
    sign = 1

    for i in range(Ns): #loop over basis
        M_E = 1.0
        r = basis[i]
        
        for j in range(N_indx-1,-1,-1): #loop over the copstr
            i_op = (L-indx[j]-1)

            b = ( one << i_op ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbit
            a = ( r >> i_op ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

            if bit_count(r,i_op) % 2 == 0: # counts number of 1 bits up to and excluding site indx[j]
                sign=1
            else:
                sign=-1


            if c_opstr[j] == I:
                continue
            elif c_opstr[j] == z:
                M_E *= (0.5 if a else -0.5)
            elif c_opstr[j] == n:
                M_E *= (1.0 if a else 0.0)
            elif c_opstr[j] == p:
                M_E *= sign*(0.0 if a else 1.0)
                r ^= b
            elif c_opstr[j] == m:
                M_E *= sign*(1.0 if a else 0.0)
                r ^= b
            else:
                error = 1
                return error

            if M_E == 0.0:
                r = basis[i]
                break

        M_E *= J
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
            basis_type[:] basis, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return op_template(pars,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])

def n_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
              str opstr, NP_INT32_t[:] indx, scalar_type J,
              basis_type[:] basis, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    return n_op_template(pars,Ns,basis,opstr,&indx[0],J,row,col,&ME[0])

def p_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    return p_op_template(pars,pars,L,pblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def p_z_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]
    return p_z_op_template(pars,pars,L,pblock,zblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def pz_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int pzblock = blocks["pzblock"]
    return pz_op_template(pars,pars,L,pzblock,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def t_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
            str opstr, NP_INT32_t[:] indx, scalar_type J,
            N_type[:] N,basis_type[:] basis,int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int a = blocks["a"]
    return t_op_template(pars,pars,L,kblock,a,Ns,&N[0],basis,opstr,&indx[0],J,row,col,&ME[0])



def t_p_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int a = blocks["a"]

    return t_p_op_template(pars,pars,L,kblock,pblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_p_z_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M2_type[:] M, basis_type[:] basis, int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]
    cdef int a = blocks["a"]

    return t_p_z_op_template(pars,pars,L,kblock,pblock,zblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_pz_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int pzblock = blocks["pzblock"]
    cdef int a = blocks["a"]

    return t_pz_op_template(pars,pars,L,kblock,pzblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_z_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zblock = blocks["zblock"]
    cdef int a = blocks["a"]

    return t_z_op_template(pars,pars,L,kblock,zblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zA_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zAblock = blocks["zAblock"]
    cdef int a = blocks["a"]

    return t_zA_op_template(pars,pars,L,kblock,zAblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zB_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M1_type[:] M, basis_type[:] basis, int L, basis_type[:] pars,**blocks):
    cdef npy_intp Ns = basis.shape[0]
    cdef int kblock = blocks["kblock"]
    cdef int zBblock = blocks["zBblock"]
    cdef int a = blocks["a"]

    return t_zB_op_template(pars,pars,L,kblock,zBblock,a,Ns,&N[0],&M[0],basis,opstr,&indx[0],J,row,col,&ME[0])


def t_zA_zB_op(basis_type[:] row, basis_type[:] col, matrix_type[:] ME,
                str opstr, NP_INT32_t[:] indx, scalar_type J, N_type[:] N,
                M2_type[:] M, basis_type[:] basis, int L, basis_type[:] pars,**blocks):
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



