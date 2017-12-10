
def get_state(L,Nup=None):
    s = 0
    
    if Nup is None:
         MAX = (<object>1 << L)
    else:
        MAX = comb(L,Nup,exact=True)
        for j in range(Nup):
            s += ( <object>1 << j )
       
    return s,MAX


# magnetization 
def n_basis(int L, object Nup_list, basis_type[:] pars,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        make_n_basis_template[basis_type](next_state_pcon_hcp,pars,MAX,s,basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+MAX] = Nup
            
        Ns_tot += MAX

# parity 
def n_p_basis(int L, object Nup_list,int pblock,basis_type[:] pars,N_type[:] N,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns = make_p_basis_template[basis_type,N_type](next_state_pcon_hcp,pars,MAX,s,L,pblock,&N[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot

def p_basis(int L,int pblock,basis_type[:] pars,N_type[:] N, basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_p_basis_template[basis_type,N_type](next_state_inc_1,pars,MAX,s,L,pblock,&N[0],basis)


# parity-spin inversion
def n_p_z_basis(int L, object Nup_list, int pblock, int zblock,basis_type[:] pars, N_type[:] N, basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns = make_p_z_basis_template[basis_type,N_type](next_state_pcon_hcp,pars,MAX,s,L,pblock,zblock,&N[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot   

def p_z_basis(int L, int pblock, int zblock,basis_type[:] pars, N_type[:] N, basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_p_z_basis_template[basis_type,N_type](next_state_inc_1,pars,MAX,s,L,pblock,zblock,&N[0],basis)


# (parity)*(spin inversion)
def n_pz_basis(int L, object Nup_list, int pzblock,basis_type[:] pars, N_type[:] N, basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns = make_pz_basis_template[basis_type,N_type](next_state_pcon_hcp,pars,MAX,s,L,pzblock,&N[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot    

def pz_basis(int L, int pzblock,basis_type[:] pars, N_type[:] N, basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_pz_basis_template[basis_type,N_type](next_state_inc_1,pars,MAX,s,L,pzblock,&N[0],basis)


# translation
def n_t_basis(int L, object Nup_list, int kblock,int a,basis_type[:] pars, N_type[:] N, basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns =  make_t_basis_template[basis_type,N_type](next_state_pcon_hcp,pars,MAX,s,L,kblock,a,&N[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot

def t_basis(int L, int kblock,int a,basis_type[:] pars, N_type[:] N, basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_t_basis_template[basis_type,N_type](next_state_inc_1,pars,MAX,s,L,kblock,a,&N[0],basis)


# translation-parity
def n_t_p_basis(int L, object Nup_list,int pblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns = make_t_p_basis_template[basis_type,N_type,M1_type](next_state_pcon_hcp,pars,MAX,s,L,pblock,kblock,a,&N[Ns_tot],&M[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot

def t_p_basis(int L,int pblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_t_p_basis_template[basis_type,N_type,M1_type](next_state_inc_1,pars,MAX,s,L,pblock,kblock,a,&N[0],&M[0],basis)




# translation-parity-spin inversion
def n_t_p_z_basis(int L, object Nup_list,int pblock,int zblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M2_type[:] M,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns = make_t_p_z_basis_template[basis_type,N_type,M2_type](next_state_pcon_hcp,pars,MAX,s,L,pblock,zblock,kblock,a,&N[Ns_tot],&M[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot    

def t_p_z_basis(int L,int pblock,int zblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M2_type[:] M,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_t_p_z_basis_template[basis_type,N_type,M2_type](next_state_inc_1,pars,MAX,s,L,pblock,zblock,kblock,a,&N[0],&M[0],basis)




# translation-(parity)*(spin inversion)
def n_t_pz_basis(int L, object Nup_list,int pzblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns = make_t_pz_basis_template[basis_type,N_type,M1_type](next_state_pcon_hcp,pars,MAX,s,L,pzblock,kblock,a,&N[Ns_tot],&M[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot   

def t_pz_basis(int L,int pzblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_t_pz_basis_template[basis_type,N_type,M1_type](next_state_inc_1,pars,MAX,s,L,pzblock,kblock,a,&N[0],&M[0],basis)


# translation-spin inversion
def n_t_z_basis(int L, object Nup_list,int zblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns = make_t_z_basis_template[basis_type,N_type,M1_type](next_state_pcon_hcp,pars,MAX,s,L,zblock,kblock,a,&N[Ns_tot],&M[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot

def t_z_basis(int L,int zblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_t_z_basis_template[basis_type,N_type,M1_type](next_state_inc_1,pars,MAX,s,L,zblock,kblock,a,&N[0],&M[0],basis)


# spin inversion
def n_z_basis(int L, object Nup_list, int zblock, basis_type[:] pars,N_type[:] N,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    cdef basis_type s
    cdef npy_intp MAX
    cdef npy_intp Ns = 0
    cdef npy_intp Ns_tot = 0
    for Nup in Nup_list:
        s,MAX = get_state(L,Nup)
        Ns = make_z_basis_template[basis_type,N_type](next_state_pcon_hcp,pars,MAX,s,L,zblock,&N[Ns_tot],basis[Ns_tot:])

        if Np_list is not None:
            Np_list[Ns_tot:Ns_tot+Ns] = Nup
            
        Ns_tot += Ns

    return Ns_tot

def z_basis(int L, int zblock, basis_type[:] pars,N_type[:] N,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_z_basis_template[basis_type,N_type](next_state_inc_1,pars,MAX,s,L,zblock,&N[0],basis)



# translation-spin inversion A
def n_t_zA_basis(int L, object Nup_list,int zAblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    raise ValueError("Nup can't be conserved for sublattice inversion symmetries.")


def t_zA_basis(int L,int zAblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_t_zA_basis_template[basis_type,N_type,M1_type](next_state_inc_1,pars,MAX,s,L,zAblock,kblock,a,&N[0],&M[0],basis)



# translation-spin inversion B
def n_t_zB_basis(int L, object Nup_list,int zBblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    raise ValueError("Nup can't be conserved for sublattice inversion symmetries.")
    

def t_zB_basis(int L,int zBblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M1_type[:] M,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_t_zB_basis_template[basis_type,N_type,M1_type](next_state_inc_1,pars,MAX,s,L,zBblock,kblock,a,&N[0],&M[0],basis)


# translation-spin inversion A-spin inversion B
def n_t_zA_zB_basis(int L, object Nup_list,int zAblock,int zBblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M2_type[:] M,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    raise ValueError("Nup can't be conserved for sublattice inversion symmetries.")    


def t_zA_zB_basis(int L,int zAblock,int zBblock,int kblock,int a,basis_type[:] pars,N_type[:] N,M2_type[:] M,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_t_zA_zB_basis_template[basis_type,N_type,M2_type](next_state_inc_1,pars,MAX,s,L,zAblock,zBblock,kblock,a,&N[0],&M[0],basis)


# spin inversion A
def n_zA_basis(int L, object Nup_list, int zAblock, basis_type[:] pars,N_type[:] N,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    raise ValueError("Nup can't be conserved for sublattice inversion symmetries.")


def zA_basis(int L, int zAblock, basis_type[:] pars,N_type[:] N,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_zA_basis_template[basis_type,N_type](next_state_inc_1,pars,MAX,s,L,zAblock,&N[0],basis)




# spin inversion B
def n_zB_basis(int L, object Nup_list, int zBblock, basis_type[:] pars,N_type[:] N,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    raise ValueError("Nup can't be conserved for sublattice inversion symmetries.")

    

def zB_basis(int L, int zBblock, basis_type[:] pars,N_type[:] N,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_zB_basis_template[basis_type,N_type](next_state_inc_1,pars,MAX,s,L,zBblock,&N[0],basis)



# spin inversion A-spin inversion B
def n_zA_zB_basis(int L, object Nup_list, int zAblock, int zBblock, basis_type[:] pars,N_type[:] N,basis_type[:] basis,NP_INT8_t[:] Np_list=None):
    raise ValueError("Nup can't be conserved for sublattice inversion symmetries.")
    

def zA_zB_basis(int L, int zAblock, int zBblock, basis_type[:] pars,N_type[:] N,basis_type[:] basis):
    cdef basis_type s
    cdef npy_intp MAX
    s,MAX = get_state(L)
    return make_zA_zB_basis_template[basis_type,N_type](next_state_inc_1,pars,MAX,s,L,zAblock,zBblock,&N[0],basis)

    
