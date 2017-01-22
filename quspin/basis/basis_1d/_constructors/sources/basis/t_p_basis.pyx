
cdef state_type make_t_p_basis_template(shifter shift,bitop fliplr,ns_type next_state, void *ns_pars,
											state_type MAX,state_type s,
											int L,int pblock,int kblock,int a, 
											NP_INT8_t*N,NP_INT8_t*m,basis_type*basis):
	cdef state_type Ns
	cdef NP_INT8_t r_temp,r,mp
	cdef int sigma,sigma_i,sigma_f,v
	cdef NP_INT8_t R[2]
	cdef state_type i
	cdef double k = (2.0*_np.pi*kblock*a)/L

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		sigma_i = 1
		sigma_f = 1
	else:
		sigma_i = -1
		sigma_f = 1

	Ns = 0

	for i in range(MAX):
		CheckState_T_P_template(shift,fliplr,kblock,L,s,a,R,ns_pars)
		r = R[0]
		mp = R[1]
		if r > 0:
			if mp != -1:
				for sigma in range(sigma_i,sigma_f+1,2):
					r_temp = r
					if 1 + sigma*pblock*cos(mp*k) == 0:
						r_temp = -1
					if (sigma == -1) and (1 - sigma*pblock*cos(mp*k) != 0):
						r_temp = -1
	
					if r_temp > 0:
						m[Ns] = mp
						N[Ns] = (sigma*r)				
						basis[Ns] = s
						Ns += 1
			else:
				for sigma in range(sigma_i,sigma_f+1,2):
					m[Ns] = -1
					N[Ns] = (sigma*r)				
					basis[Ns] = s
					Ns += 1

		s = next_state(s,ns_pars)

	return Ns
