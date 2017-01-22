cdef state_type make_t_zB_basis_template(shifter shift,bitop flip_sublat_B,ns_type next_state, void *ns_pars,
											state_type MAX,state_type s,
											int L,int zBblock,int kblock,int a,
											NP_INT8_t*N, NP_INT8_t*m,basis_type*basis): 
	cdef double k 
	cdef state_type Ns
	cdef state_type i
	cdef NP_INT8_t mzB,r
	cdef NP_INT8_t R[2]
	cdef int j
	

	k = 2.0*_np.pi*kblock*a/L
	Ns = 0
	
	for i in range(MAX):
		CheckState_T_ZB_template(shift,flip_sublat_B,kblock,L,s,a,R,ns_pars)
		r = R[0]
		mzB = R[1]

		if r > 0:
			if mzB != -1:
				if 1 + zBblock*cos(k*mzB) == 0:
					r = -1				

			if r > 0:
				m[Ns] = mzB
				N[Ns] = r			
				basis[Ns] = s
				Ns += 1	
		
		s = next_state(s,ns_pars)

	return Ns
