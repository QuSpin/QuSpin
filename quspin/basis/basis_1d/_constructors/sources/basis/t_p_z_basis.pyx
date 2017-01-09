
cdef state_type make_t_p_z_basis_template(shifter shift,bitop fliplr,ns_type next_state,
												state_type MAX,state_type s,
												int L, int pblock, int zblock, int kblock, int a,
												NP_INT8_t*N, NP_INT16_t*m, basis_type*basis):
	cdef double k = 2.0*_np.pi*kblock*a/L
	cdef state_type Ns=0
	cdef state_type i
	cdef NP_INT8_t r,r_temp,mp,mz,mpz
	cdef int sigma,sigma_i,sigma_f
	cdef _np.ndarray[NP_INT8_t,ndim=1] R = _np.zeros(4,dtype=NP_INT8)

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		sigma_i = 1
		sigma_f = 1
	else:
		sigma_i = -1
		sigma_f = 1

	cdef int j

	for i in range(MAX):
		CheckState_T_P_Z_template(shift,fliplr,flip_all,kblock,L,s,a,R)
		r = R[0]
		mp = R[1]
		mz = R[2]
		mpz = R[3]
		if r>0:
			if mp == -1 and mz == -1 and mpz == -1:
				for sigma in range(sigma_i,sigma_f+1,2):
					m[Ns] = ((L+1)**2)
					N[Ns] = (sigma*r)				
					basis[Ns] = s
					Ns += 1
			if mp != -1 and mz == -1 and mpz == -1:
				for sigma in range(sigma_i,sigma_f+1,2):
					r_temp = r
					if 1 + sigma*pblock*cos(mp*k) == 0:
						r_temp = -1
					if (sigma == -1) and (1 - sigma*pblock*cos(mp*k) != 0):
						r_temp = -1
					if r_temp > 0:
						m[Ns] = (mp + 2*(L+1)**2)
						N[Ns] = (sigma*r)				
						basis[Ns] = s
						Ns += 1
			if mp == -1 and mz != -1 and mpz == -1:
				for sigma in range(sigma_i,sigma_f+1,2):
					r_temp = r
					if 1 + zblock*cos(k*mz) == 0:
						r_temp = -1
					if r_temp > 0:		
						m[Ns] = (mz*(L+1) + 3*(L+1)**2)
						N[Ns] = (sigma*r)				
						basis[Ns] = s
						Ns += 1
			if mp == -1 and mz == -1 and mpz != -1:
				for sigma in range(sigma_i,sigma_f+1,2):
					r_temp = r
					if 1 + sigma*pblock*zblock*cos(mpz*k) == 0:
						r_temp = -1
					if (sigma == -1) and (1 - sigma*pblock*zblock*cos(mpz*k) != 0):
						r_temp = -1
					if r_temp>0:
						m[Ns] = (mpz + 4*(L+1)**2)
						N[Ns] = (sigma*r)				
						basis[Ns] = s
						Ns += 1
			if mp != -1 and mz != -1:
				for sigma in range(sigma_i,sigma_f+1,2):
					r_temp = r
					if (1 + sigma*pblock*cos(mp*k) == 0) or (1 + zblock*cos(mz*k) == 0):
						r_temp = -1
					if (sigma == -1) and ( (1 - sigma*pblock*cos(mp*k) != 0) or (1 - zblock*cos(mz*k) != 0) ):
						r_temp = -1
					if r_temp>0:
						m[Ns] = (mp + (L+1)*mz + 5*(L+1)**2)
						N[Ns] = (sigma*r)				
						basis[Ns] = s
						Ns += 1

		s = next_state(s)

	return Ns
