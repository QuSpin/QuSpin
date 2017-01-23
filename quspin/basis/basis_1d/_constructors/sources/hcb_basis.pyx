# magnetization 
def hcb_m_basis(int L, int Nup, state_type Ns,_np.ndarray[basis_type,ndim=1] basis):
    cdef state_type s
    cdef int j
    cdef state_type MAX = Ns
    s = 0
    for j in range(Nup):
        s += ( 1ull << j )

    make_n_basis_template[basis_type](next_state_pcon_hcb,NULL,MAX,s,&basis[0])

