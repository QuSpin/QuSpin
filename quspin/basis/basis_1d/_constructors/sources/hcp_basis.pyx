# magnetization 
def n_basis(int L, int Nup, npy_uintp Ns, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef int j
    cdef npy_uintp MAX = comb(L,Nup,exact=True)
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    make_n_basis_template[basis_type](next_state_pcon_hcp,pars,MAX,s,basis)

# parity 
def n_p_basis(int L,int Nup,int pblock,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j

    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    return make_p_basis_template[basis_type,N_type](fliplr,next_state_pcon_hcp,pars,MAX,s,L,pblock,&N[0],basis)


def p_basis(int L,int pblock,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s = 0
    cdef npy_uintp MAX=1ull<<L

    return make_p_basis_template[basis_type,N_type](fliplr,next_state_inc_1,pars,MAX,s,L,pblock,&N[0],basis)


# parity-spin inversion
def n_p_z_basis(int L, int Nup, int pblock, int zblock,_np.ndarray[basis_type,ndim=1] pars, _np.ndarray[N_type,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    return make_p_z_basis_template[basis_type,N_type](fliplr,flip_all,next_state_pcon_hcp,pars,MAX,s,L,pblock,zblock,&N[0],basis)
    

def p_z_basis(int L, int pblock, int zblock,_np.ndarray[basis_type,ndim=1] pars, _np.ndarray[N_type,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L

    return make_p_z_basis_template[basis_type,N_type](fliplr,flip_all,next_state_inc_1,pars,MAX,s,L,pblock,zblock,&N[0],basis)


# (parity)*(spin inversion)
def n_pz_basis(int L, int Nup, int pzblock,_np.ndarray[basis_type,ndim=1] pars, _np.ndarray[N_type,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_pz_basis_template[basis_type,N_type](fliplr,flip_all,next_state_pcon_hcp,pars,MAX,s,L,pzblock,&N[0],basis)
    

def pz_basis(int L, int pzblock,_np.ndarray[basis_type,ndim=1] pars, _np.ndarray[N_type,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_pz_basis_template[basis_type,N_type](fliplr,flip_all,next_state_inc_1,pars,MAX,s,L,pzblock,&N[0],basis)


# translation
def n_t_basis(int L, int Nup, int kblock,int a,_np.ndarray[basis_type,ndim=1] pars, _np.ndarray[N_type,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_basis_template[basis_type,N_type](shift,next_state_pcon_hcp,pars,MAX,s,L,kblock,a,&N[0],basis)


def t_basis(int L, int kblock,int a,_np.ndarray[basis_type,ndim=1] pars, _np.ndarray[N_type,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_basis_template[basis_type,N_type](shift,next_state_inc_1,pars,MAX,s,L,kblock,a,&N[0],basis)


# translation-parity
def n_t_p_basis(int L, int Nup,int pblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_p_basis_template[basis_type,N_type](shift,fliplr,next_state_pcon_hcp,pars,MAX,s,L,pblock,kblock,a,&N[0],&M[0],basis)


def t_p_basis(int L,int pblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_p_basis_template[basis_type,N_type](shift,fliplr,next_state_inc_1,pars,MAX,s,L,pblock,kblock,a,&N[0],&M[0],basis)




# translation-parity-spin inversion
def n_t_p_z_basis(int L, int Nup,int pblock,int zblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[M_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_p_z_basis_template[basis_type,N_type,M_type](shift,fliplr,next_state_pcon_hcp,pars,MAX,s,L,pblock,zblock,kblock,a,&N[0],&M[0],basis)
    

def t_p_z_basis(int L,int pblock,int zblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[M_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_p_z_basis_template[basis_type,N_type,M_type](shift,fliplr,next_state_inc_1,pars,MAX,s,L,pblock,zblock,kblock,a,&N[0],&M[0],basis)




# translation-(parity)*(spin inversion)
def n_t_pz_basis(int L, int Nup,int pzblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_pz_basis_template[basis_type,N_type](shift,fliplr,flip_all,next_state_pcon_hcp,pars,MAX,s,L,pzblock,kblock,a,&N[0],&M[0],basis)
    

def t_pz_basis(int L,int pzblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_pz_basis_template[basis_type,N_type](shift,fliplr,flip_all,next_state_inc_1,pars,MAX,s,L,pzblock,kblock,a,&N[0],&M[0],basis)


# translation-spin inversion
def n_t_z_basis(int L,int Nup,int zblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_z_basis_template[basis_type,N_type](shift,flip_all,next_state_pcon_hcp,pars,MAX,s,L,zblock,kblock,a,&N[0],&M[0],basis)


def t_z_basis(int L,int zblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_z_basis_template[basis_type,N_type](shift,flip_all,next_state_inc_1,pars,MAX,s,L,zblock,kblock,a,&N[0],&M[0],basis)




# translation-spin inversion A
def n_t_zA_basis(int L, int Nup,int zAblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_zA_basis_template[basis_type,N_type](shift,flip_sublat_A,next_state_pcon_hcp,pars,MAX,s,L,zAblock,kblock,a,&N[0],&M[0],basis)


def t_zA_basis(int L,int zAblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_zA_basis_template[basis_type,N_type](shift,flip_sublat_A,next_state_inc_1,pars,MAX,s,L,zAblock,kblock,a,&N[0],&M[0],basis)



# translation-spin inversion B
def n_t_zB_basis(int L, int Nup,int zBblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_zB_basis_template[basis_type,N_type](shift,flip_sublat_B,next_state_pcon_hcp,pars,MAX,s,L,zBblock,kblock,a,&N[0],&M[0],basis)
    

def t_zB_basis(int L,int zBblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[N_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_zB_basis_template[basis_type,N_type](shift,flip_sublat_B,next_state_inc_1,pars,MAX,s,L,zBblock,kblock,a,&N[0],&M[0],basis)




# translation-spin inversion A-spin inversion B
def n_t_zA_zB_basis(int L,int Nup,int zAblock,int zBblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[M_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_zA_zB_basis_template[basis_type,N_type,M_type](shift,flip_sublat_A,flip_sublat_B,flip_all,next_state_pcon_hcp,pars,MAX,s,L,zAblock,zBblock,kblock,a,&N[0],&M[0],basis)

def t_zA_zB_basis(int L,int zAblock,int zBblock,int kblock,int a,_np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[M_type,ndim=1] M,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_zA_zB_basis_template[basis_type,N_type,M_type](shift,flip_sublat_A,flip_sublat_B,flip_all,next_state_inc_1,pars,MAX,s,L,zAblock,zBblock,kblock,a,&N[0],&M[0],basis)





# spin inversion
def n_z_basis(int L,int Nup, int zblock, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_z_basis_template[basis_type,N_type](flip_all,next_state_pcon_hcp,pars,MAX,s,L,zblock,&N[0],basis)


def z_basis(int L, int zblock, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_z_basis_template[basis_type,N_type](flip_all,next_state_inc_1,pars,MAX,s,L,zblock,&N[0],basis)




# spin inversion A
def n_zA_basis(int L,int Nup, int zAblock, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zA_basis_template[basis_type,N_type](flip_sublat_A,next_state_pcon_hcp,pars,MAX,s,L,zAblock,&N[0],basis)


def zA_basis(int L, int zAblock, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_zA_basis_template[basis_type,N_type](flip_sublat_A,next_state_inc_1,pars,MAX,s,L,zAblock,&N[0],basis)




# spin inversion B
def n_zB_basis(int L,int Nup, int zBblock, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zB_basis_template[basis_type,N_type](flip_sublat_B,next_state_pcon_hcp,pars,MAX,s,L,zBblock,&N[0],basis)

    

def zB_basis(int L, int zBblock, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_zB_basis_template[basis_type,N_type](flip_sublat_B,next_state_inc_1,pars,MAX,s,L,zBblock,&N[0],basis)



# spin inversion A-spin inversion B
def n_zA_zB_basis(int L,int Nup, int zAblock, int zBblock, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zA_zB_basis_template[basis_type,N_type](flip_sublat_A,flip_sublat_B,flip_all,next_state_pcon_hcp,pars,MAX,s,L,zAblock,zBblock,&N[0],basis)
    

def zA_zB_basis(int L, int zAblock, int zBblock, _np.ndarray[basis_type,ndim=1] pars,_np.ndarray[N_type,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_zA_zB_basis_template[basis_type,N_type](flip_sublat_A,flip_sublat_B,flip_all,next_state_inc_1,pars,MAX,s,L,zAblock,zBblock,&N[0],basis)

    
