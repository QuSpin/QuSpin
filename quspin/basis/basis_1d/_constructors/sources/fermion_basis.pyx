# magnetization 
def fermion_n_basis(int L, int Nup, npy_uintp Ns,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef int j
    cdef npy_uintp MAX = comb(L,Nup,exact=True)
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    make_n_basis_template[basis_type](next_state_pcon_hcb,None,MAX,s,basis)

# parity 
def fermion_n_p_basis(int L,int Nup,int pblock,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j

    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    return make_p_basis_template[basis_type](fliplr,next_state_pcon_hcb,None,MAX,s,L,pblock,&N[0],basis)


def fermion_p_basis(int L,int pblock,_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s = 0
    cdef npy_uintp MAX=1ull<<L

    return make_p_basis_template[basis_type](fliplr,next_state_inc_1,None,MAX,s,L,pblock,&N[0],basis)


# parity-spin inversion
def fermion_n_p_z_basis(int L, int Nup, int pblock, int zblock, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    return make_p_z_basis_template[basis_type](fliplr,flip_all,next_state_pcon_hcb,None,MAX,s,L,pblock,zblock,&N[0],basis)
    

def fermion_p_z_basis(int L, int pblock, int zblock, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L

    return make_p_z_basis_template[basis_type](fliplr,flip_all,next_state_inc_1,None,MAX,s,L,pblock,zblock,&N[0],basis)


# (parity)*(spin inversion)
def fermion_n_pz_basis(int L, int Nup, int pzblock, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_pz_basis_template[basis_type](fliplr,flip_all,next_state_pcon_hcb,None,MAX,s,L,pzblock,&N[0],basis)
    

def fermion_pz_basis(int L, int pzblock, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_pz_basis_template[basis_type](fliplr,flip_all,next_state_inc_1,None,MAX,s,L,pzblock,&N[0],basis)


# translation
def fermion_n_t_basis(int L, int Nup, int kblock,int a, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_basis_template[basis_type](shift,next_state_pcon_hcb,None,MAX,s,L,kblock,a,&N[0],basis)


def fermion_t_basis(int L, int kblock,int a, _np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_basis_template[basis_type](shift,next_state_inc_1,None,MAX,s,L,kblock,a,&N[0],basis)


# translation-parity
def fermion_n_t_p_basis(int L, int Nup,int pblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_p_basis_template[basis_type](shift,fliplr,next_state_pcon_hcb,None,MAX,s,L,pblock,kblock,a,&N[0],&m[0],basis)


def fermion_t_p_basis(int L,int pblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_p_basis_template[basis_type](shift,fliplr,next_state_inc_1,None,MAX,s,L,pblock,kblock,a,&N[0],&m[0],basis)




# translation-parity-spin inversion
def fermion_n_t_p_z_basis(int L, int Nup,int pblock,int zblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_p_z_basis_template[basis_type](shift,fliplr,next_state_pcon_hcb,None,MAX,s,L,pblock,zblock,kblock,a,&N[0],&m[0],basis)
    

def fermion_t_p_z_basis(int L,int pblock,int zblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_p_z_basis_template[basis_type](shift,fliplr,next_state_inc_1,None,MAX,s,L,pblock,zblock,kblock,a,&N[0],&m[0],basis)




# translation-(parity)*(spin inversion)
def fermion_n_t_pz_basis(int L, int Nup,int pzblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_pz_basis_template[basis_type](shift,fliplr,flip_all,next_state_pcon_hcb,None,MAX,s,L,pzblock,kblock,a,&N[0],&m[0],basis)
    

def fermion_t_pz_basis(int L,int pzblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_pz_basis_template[basis_type](shift,fliplr,flip_all,next_state_inc_1,None,MAX,s,L,pzblock,kblock,a,&N[0],&m[0],basis)


# translation-spin inversion
def fermion_n_t_z_basis(int L,int Nup,int zblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_z_basis_template[basis_type](shift,flip_all,next_state_pcon_hcb,None,MAX,s,L,zblock,kblock,a,&N[0],&m[0],basis)


def fermion_t_z_basis(int L,int zblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_z_basis_template[basis_type](shift,flip_all,next_state_inc_1,None,MAX,s,L,zblock,kblock,a,&N[0],&m[0],basis)




# translation-spin inversion A
def fermion_n_t_zA_basis(int L, int Nup,int zAblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_zA_basis_template[basis_type](shift,flip_sublat_A,next_state_pcon_hcb,None,MAX,s,L,zAblock,kblock,a,&N[0],&m[0],basis)


def fermion_t_zA_basis(int L,int zAblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_zA_basis_template[basis_type](shift,flip_sublat_A,next_state_inc_1,None,MAX,s,L,zAblock,kblock,a,&N[0],&m[0],basis)



# translation-spin inversion B
def fermion_n_t_zB_basis(int L, int Nup,int zBblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_t_zB_basis_template[basis_type](shift,flip_sublat_B,next_state_pcon_hcb,None,MAX,s,L,zBblock,kblock,a,&N[0],&m[0],basis)
    

def fermion_t_zB_basis(int L,int zBblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_zB_basis_template[basis_type](shift,flip_sublat_B,next_state_inc_1,None,MAX,s,L,zBblock,kblock,a,&N[0],&m[0],basis)




# translation-spin inversion A-spin inversion B
def fermion_n_t_zA_zB_basis(int L,int Nup,int zAblock,int zBblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )


def fermion_t_zA_zB_basis(int L,int zAblock,int zBblock,int kblock,int a,_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_t_zA_zB_basis_template[basis_type](shift,flip_sublat_A,flip_sublat_B,flip_all,next_state_inc_1,None,MAX,s,L,zAblock,zBblock,kblock,a,&N[0],&m[0],basis)





# spin inversion
def fermion_n_z_basis(int L,int Nup,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_z_basis_template[basis_type](flip_all,next_state_pcon_hcb,None,MAX,s,L,basis)


def fermion_z_basis(int L,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_z_basis_template[basis_type](flip_all,next_state_inc_1,None,MAX,s,L,basis)




# spin inversion A
def fermion_n_zA_basis(int L,int Nup,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zA_basis_template[basis_type](flip_sublat_A,next_state_pcon_hcb,None,MAX,s,L,basis)


def fermion_zA_basis(int L,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_zA_basis_template[basis_type](flip_sublat_A,next_state_inc_1,None,MAX,s,L,basis)




# spin inversion B
def fermion_n_zB_basis(int L,int Nup,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zB_basis_template[basis_type](flip_sublat_B,next_state_pcon_hcb,None,MAX,s,L,basis)

    

def fermion_zB_basis(int L,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_zB_basis_template[basis_type](flip_sublat_B,next_state_inc_1,None,MAX,s,L,basis)



# spin inversion A-spin inversion B
def fermion_n_zA_zB_basis(int L,int Nup,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s
    cdef npy_uintp MAX=comb(L,Nup,exact=True)
    cdef int j
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )
    return make_zA_zB_basis_template[basis_type](flip_sublat_A,flip_sublat_B,flip_all,next_state_pcon_hcb,None,MAX,s,L,basis)
    

def fermion_zA_zB_basis(int L,_np.ndarray[basis_type,ndim=1] basis):
    cdef npy_uintp s=0
    cdef npy_uintp MAX=1ull<<L
    return make_zA_zB_basis_template[basis_type](flip_sublat_A,flip_sublat_B,flip_all,next_state_inc_1,None,MAX,s,L,basis)

    
