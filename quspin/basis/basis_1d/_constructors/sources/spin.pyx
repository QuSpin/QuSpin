


cdef int spin_op_func(state_type Ns, basis_type *basis,
                    int N_indx, str opstr,NP_INT32_t *indx,scalar_type J,
                    index_type *row, matrix_type *ME):

    cdef state_type i
    cdef state_type r,b
    cdef int j,error
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
        if matrix_type == float or matrix_type == double or matrix_type == longdouble:
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
def spin_op(_np.ndarray[index_type,ndim=1] row,_np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,**blocks):
    cdef state_type Ns = basis.shape[0]
    return op_template[basis_type,index_type,matrix_type](spin_op_func,Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0])

# magnetization 
def spin_m_basis(int L, int Nup, state_type Ns,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef int j

    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    make_n_basis_template[basis_type](next_state_pcon_hcb,Ns,s,&basis[0])


def spin_m_op(_np.ndarray[index_type,ndim=1] row,_np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[basis_type,ndim=1] basis,**blocks):
    cdef state_type Ns = basis.shape[0]
    return n_op_template[basis_type,index_type,matrix_type](spin_op_func,Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0])


# parity 
def spin_m_p_basis(int L,int Nup,int pblock,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j

    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    return make_p_basis_template[basis_type](fliplr,next_state_pcon_hcb,MAX,s,L,pblock,&N[0],&basis[0])

    
def spin_p_basis(int L,int pblock,_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s = 0
    cdef state_type MAX=1ull<<L

    return make_p_basis_template[basis_type](fliplr,next_state_inc_1,MAX,s,L,pblock,&N[0],&basis[0])


def spin_p_op(_np.ndarray[index_type,ndim=1] row,_np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef state_type Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    return p_op_template[basis_type,index_type,matrix_type](spin_op_func,fliplr,L,pblock,Ns,&N[0],&basis[0],opstr,&indx[0],J,&row[0],&ME[0])



# parity-spin inversion
def spin_m_p_z_basis(int L, int Nup, int pblock, int zblock, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    return make_p_z_basis_template[basis_type](fliplr,flip_all,next_state_pcon_hcb,MAX,s,L,pblock,zblock,&N[0],&basis[0])
    

def spin_p_z_basis(int L, int pblock, int zblock, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L

    return make_p_z_basis_template[basis_type](fliplr,flip_all,next_state_inc_1,MAX,s,L,pblock,zblock,&N[0],&basis[0])

def spin_p_z_op(_np.ndarray[index_type,ndim=1] row,_np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef state_type Ns = basis.shape[0]
    cdef int pblock = blocks["pblock"]
    cdef int zblock = blocks["zblock"]
    return p_z_op_template[basis_type,index_type,matrix_type](spin_op_func,fliplr,flip_all,L,pblock,zblock,Ns,&N[0],&basis[0],opstr,&indx[0],J,&row[0],&ME[0])


# (parity)*(spin inversion)
def spin_m_pz_basis(int L, int Nup, int pzblock, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_pz_basis_template[basis_type](fliplr,flip_all,next_state_pcon_hcb,MAX,s,L,pzblock,&N[0],&basis[0])
    

def spin_pz_basis(int L, int pzblock, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_pz_basis_template[basis_type](fliplr,flip_all,next_state_inc_1,MAX,s,L,pzblock,&N[0],&basis[0])

def spin_pz_op(_np.ndarray[index_type,ndim=1] row,_np.ndarray[matrix_type,ndim=1] ME,
            str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, scalar_type J,
            _np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis,int L,**blocks):
    cdef state_type Ns = basis.shape[0]
    cdef int pzblock = blocks["pzblock"]
    return pz_op_template[basis_type,index_type,matrix_type](spin_op_func,fliplr,flip_all,L,pzblock,Ns,&N[0],&basis[0],opstr,&indx[0],J,&row[0],&ME[0])


# translation
def spin_m_t_basis(int L, int Nup, int kblock,int a, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_basis_template[basis_type](shift,next_state_pcon_hcb,MAX,s,L,kblock,a,&N[0],&basis[0])


def spin_t_basis(int L, int kblock,int a, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_t_basis_template[basis_type](shift,next_state_inc_1,MAX,s,L,kblock,a,&N[0],&basis[0])


# translation-parity
def spin_m_t_p_basis(int L, int Nup,int pblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_p_basis_template[basis_type](shift,fliplr,next_state_pcon_hcb,MAX,s,L,pblock,kblock,a,&N[0],&m[0],&basis[0])


def spin_t_p_basis(int L,int pblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_t_p_basis_template[basis_type](shift,fliplr,next_state_inc_1,MAX,s,L,pblock,kblock,a,&N[0],&m[0],&basis[0])
    


# translation-parity-spin inversion
def spin_m_t_p_z_basis(int L, int Nup,int pblock,int zblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_p_z_basis_template[basis_type](shift,fliplr,next_state_pcon_hcb,MAX,s,L,pblock,zblock,kblock,a,&N[0],&m[0],&basis[0])

    

def spin_t_p_z_basis(int L,int pblock,int zblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_t_p_z_basis_template[basis_type](shift,fliplr,next_state_inc_1,MAX,s,L,pblock,zblock,kblock,a,&N[0],&m[0],&basis[0])


# translation-(parity)*(spin inversion)
def spin_m_t_pz_basis(int L, int Nup,int pzblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_pz_basis_template[basis_type](shift,fliplr,flip_all,next_state_pcon_hcb,MAX,s,L,pzblock,kblock,a,&N[0],&m[0],&basis[0])
    

def spin_t_pz_basis(int L,int pzblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_t_pz_basis_template[basis_type](shift,fliplr,flip_all,next_state_inc_1,MAX,s,L,pzblock,kblock,a,&N[0],&m[0],&basis[0])


# translation-spin inversion
def spin_m_t_z_basis(int L,int Nup,int zblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_z_basis_template[basis_type](shift,flip_all,next_state_pcon_hcb,MAX,s,L,zblock,kblock,a,&N[0],&m[0],&basis[0])


def spin_t_z_basis(int L,int zblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_t_z_basis_template[basis_type](shift,flip_all,next_state_inc_1,MAX,s,L,zblock,kblock,a,&N[0],&m[0],&basis[0])


# translation-spin inversion A
def spin_m_t_zA_basis(int L, int Nup,int zAblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_zA_basis_template[basis_type](shift,flip_sublat_A,next_state_pcon_hcb,MAX,s,L,zAblock,kblock,a,&N[0],&m[0],&basis[0])


def spin_t_zA_basis(int L,int zAblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_t_zA_basis_template[basis_type](shift,flip_sublat_A,next_state_inc_1,MAX,s,L,zAblock,kblock,a,&N[0],&m[0],&basis[0])


# translation-spin inversion B
def spin_m_t_zB_basis(int L, int Nup,int zBblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_zB_basis_template[basis_type](shift,flip_sublat_B,next_state_pcon_hcb,MAX,s,L,zBblock,kblock,a,&N[0],&m[0],&basis[0])
    

def spin_t_zB_basis(int L,int zBblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_t_zB_basis_template[basis_type](shift,flip_sublat_B,next_state_inc_1,MAX,s,L,zBblock,kblock,a,&N[0],&m[0],&basis[0])


# translation-spin inversion A-spin inversion B
def spin_m_t_zA_zB_basis(int L,int Nup,int zAblock,int zBblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )


def spin_t_zA_zB_basis(int L,int zAblock,int zBblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_t_zA_zB_basis_template[basis_type](shift,flip_sublat_A,flip_sublat_B,flip_all,next_state_inc_1,MAX,s,L,zAblock,zBblock,kblock,a,&N[0],&m[0],&basis[0])


# spin inversion
def spin_m_z_basis(int L,int Nup,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_z_basis_template[basis_type](flip_all,next_state_pcon_hcb,MAX,s,L,&basis[0])


def spin_z_basis(int L,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_z_basis_template[basis_type](flip_all,next_state_inc_1,MAX,s,L,&basis[0])


# spin inversion A
def spin_m_zA_basis(int L,int Nup,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zA_basis_template[basis_type](flip_sublat_A,next_state_pcon_hcb,MAX,s,L,&basis[0])
    

def spin_zA_basis(int L,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_zA_basis_template[basis_type](flip_sublat_A,next_state_inc_1,MAX,s,L,&basis[0])


# spin inversion B
def spin_m_zB_basis(int L,int Nup,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zB_basis_template[basis_type](flip_sublat_B,next_state_pcon_hcb,MAX,s,L,&basis[0])

    

def spin_zB_basis(int L,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_zB_basis_template[basis_type](flip_sublat_B,next_state_inc_1,MAX,s,L,&basis[0])


# spin inversion A-spin inversion B
def spin_m_zA_zB_basis(int L,int Nup,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef state_type MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zA_zB_basis_template[basis_type](flip_sublat_A,flip_sublat_B,flip_all,next_state_pcon_hcb,MAX,s,L,&basis[0])
    

def spin_zA_zB_basis(int L,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s=0
    cdef state_type MAX=1ull<<L
    return make_zA_zB_basis_template[basis_type](flip_sublat_A,flip_sublat_B,flip_all,next_state_inc_1,MAX,s,L,&basis[0])

    
