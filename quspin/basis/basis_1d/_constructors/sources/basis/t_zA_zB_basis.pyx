
cdef state_type make_t_zA_zB_basis_template(shifter shift,bitop flip_sublat_A,bitop flip_sublat_B,
												bitop flip_all,ns_type next_state, void *ns_pars,
												state_type MAX,state_type s,
												int L,int zAblock,int zBblock,int kblock,int a,
												NP_INT8_t*N,NP_INT16_t*m,basis_type*basis): 
	cdef double k 
	cdef state_type Ns
	cdef NP_INT8_t mzA,mzB,mz,r
	cdef state_type i
	cdef NP_INT8_t R[4]
	cdef int j
	
	k = 2.0*_np.pi*kblock*a/L
	Ns = 0
	
	for i in range(MAX):
		CheckState_T_ZA_ZB_template(shift,flip_sublat_A,flip_sublat_B,flip_all,kblock,L,s,a,R,ns_pars)
		r = R[0]
		mzA = R[1]
		mzB = R[2]
		mz = R[3]

		if r > 0:
			if mzA == -1 and mzB == -1 and mz == -1:
				m[Ns] = (L+1)
				N[Ns] = r			
				basis[Ns] = s
				Ns += 1	

			if mzA != -1 and mzB == -1 and mz == -1:
				if 1 + zAblock*cos(k*mzA) != 0:			
					m[Ns] = mzA + 2*(L+1)
					N[Ns] = r			
					basis[Ns] = s
					Ns += 1	

			if mzA == -1 and mzB != -1 and mz == -1:
				if 1 + zBblock*cos(k*mzB) != 0:		
					m[Ns] = mzB + 3*(L+1)
					N[Ns] = r			
					basis[Ns] = s
					Ns += 1

			if mzA == -1 and mzB == -1 and mz != -1:
				if 1 + zAblock*zBblock*cos(k*mz) != 0:					
					m[Ns] = mz + 4*(L+1)
					N[Ns] = r			
					basis[Ns] = s
					Ns += 1
		
		s = next_state(s,ns_pars)
				
	return Ns
