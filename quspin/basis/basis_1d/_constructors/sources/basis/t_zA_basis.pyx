
cdef state_type make_t_zA_basis_template(shifter shift,bitop flip_sublat_A,ns_type next_state, void *ns_pars,
												state_type MAX,state_type s,
												int L,int zAblock,int kblock,int a,
												NP_INT8_t*N, NP_INT8_t*m, basis_type*basis): 
	cdef double k 
	cdef state_type i
	cdef state_type Ns
	cdef NP_INT8_t mzA,r
	cdef NP_INT8_t R[2]
	cdef int j	

	k = 2.0*_np.pi*kblock*a/L
	Ns = 0
	

	for i in range(MAX):
		CheckState_T_ZA_template(shift,flip_sublat_A,kblock,L,s,a,R,ns_pars)
		r = R[0]
		mzA = R[1]

		if r > 0:
			if mzA != -1:
				if 1 + zAblock*cos(k*mzA) == 0:
					r = -1				

			if r > 0:
				m[Ns] = mzA
				N[Ns] = r			
				basis[Ns] = s
				Ns += 1	

		s = next_state(s,ns_pars)
				

	return Ns
