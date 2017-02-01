
cdef void make_n_basis_template(ns_type next_state,object[basis_type,ndim=1,mode="c"] ns_pars,npy_uintp MAX,basis_type s,object[basis_type,ndim=1,mode="c"] basis):
	cdef npy_uintp i
	for i in range(MAX):
		basis[i] = s
		s = next_state(s,ns_pars)	



cdef npy_uintp make_p_basis_template(bitop fliplr,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
											npy_uintp MAX, basis_type s,
											int L,int pblock, NP_INT8_t * N, object[basis_type,ndim=1,mode="c"]  basis):
	cdef npy_uintp Ns
	cdef NP_INT8_t rp

	cdef int j
	cdef npy_uintp i

	Ns = 0

	for i in range(MAX):
		rp = CheckState_P_template[basis_type](fliplr,pblock,s,L,ns_pars)
		if rp > 0:
			basis[Ns] = s
			N[Ns] = rp
			Ns += 1

		s = next_state(s,ns_pars)

	return Ns





cdef npy_uintp make_p_z_basis_template(bitop fliplr,bitop flip_all,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
									npy_uintp MAX,basis_type s,int L, int pblock, int zblock,
									NP_INT8_t * N, object[basis_type,ndim=1,mode="c"]  basis):
	cdef npy_uintp Ns
	cdef NP_INT8_t rpz
	cdef npy_uintp i

	Ns = 0
	for i in range(MAX):
		rpz = CheckState_P_Z_template[basis_type](fliplr,flip_all,pblock,zblock,s,L,ns_pars)
		if rpz > 0:
			basis[Ns] = s
			N[Ns] = rpz
			Ns += 1
		s = next_state(s,ns_pars)

	return Ns





cdef npy_uintp make_pz_basis_template(bitop fliplr,bitop flip_all,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
										npy_uintp MAX,basis_type s,int L, int pzblock,
										NP_INT8_t * N, object[basis_type,ndim=1,mode="c"]  basis):
	cdef npy_uintp Ns
	cdef NP_INT8_t rpz
	cdef npy_uintp i

	Ns = 0
	for i in range(MAX):
		rpz = CheckState_PZ_template[basis_type](fliplr,flip_all,pzblock,s,L,ns_pars)
		if rpz > 0:
			basis[Ns] = s
			N[Ns] = rpz
			Ns += 1
		
		s = next_state(s,ns_pars)

	return Ns





cdef npy_uintp make_t_basis_template(shifter shift,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
										npy_uintp MAX,basis_type s,int L,
										int kblock,int a,NP_INT8_t * N, object[basis_type,ndim=1,mode="c"]  basis):
	cdef npy_uintp Ns
	cdef NP_INT8_t r
	cdef npy_uintp i

	Ns = 0
	for i in range(MAX):
		r=CheckState_T_template[basis_type](shift,kblock,L,s,a,ns_pars)
		if r > 0:
			N[Ns] = r				
			basis[Ns] = s
			Ns += 1		

		s = next_state(s,ns_pars)

	return Ns





cdef npy_uintp make_t_p_basis_template(shifter shift,bitop fliplr,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
											npy_uintp MAX,basis_type s,
											int L,int pblock,int kblock,int a, 
											NP_INT8_t*N,NP_INT8_t*m,object[basis_type,ndim=1,mode="c"] basis):
	cdef npy_uintp Ns
	cdef NP_INT8_t r_temp,r,mp
	cdef int sigma,sigma_i,sigma_f,v
	cdef NP_INT8_t R[2]
	cdef npy_uintp i
	cdef double k = (2.0*_np.pi*kblock*a)/L

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		sigma_i = 1
		sigma_f = 1
	else:
		sigma_i = -1
		sigma_f = 1

	Ns = 0

	R[0] = 0
	R[1] = 0

	for i in range(MAX):
		CheckState_T_P_template[basis_type](shift,fliplr,kblock,L,s,a,R,ns_pars)
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






cdef npy_uintp make_t_p_z_basis_template(shifter shift,bitop fliplr,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
												npy_uintp MAX,basis_type s,
												int L, int pblock, int zblock, int kblock, int a,
												NP_INT8_t*N, NP_INT16_t*m, object[basis_type,ndim=1,mode="c"] basis):
	cdef double k = 2.0*_np.pi*kblock*a/L
	cdef npy_uintp Ns=0
	cdef npy_uintp i
	cdef NP_INT8_t r,r_temp,mp,mz,mpz
	cdef int sigma,sigma_i,sigma_f
	cdef NP_INT8_t R[4]

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		sigma_i = 1
		sigma_f = 1
	else:
		sigma_i = -1
		sigma_f = 1

	cdef int j

	R[0] = 0
	R[1] = 0
	R[2] = 0
	R[3] = 0

	for i in range(MAX):
		CheckState_T_P_Z_template[basis_type](shift,fliplr,flip_all,kblock,L,s,a,R,ns_pars)
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

		s = next_state(s,ns_pars)

	return Ns





cdef npy_uintp make_t_pz_basis_template(shifter shift,bitop fliplr,bitop flip_all,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
											npy_uintp MAX,basis_type s,
											int L,int pzblock,int kblock,int a,
											NP_INT8_t*N,NP_INT8_t*m,object[basis_type,ndim=1,mode="c"] basis):
	cdef double k 
	
	cdef npy_uintp Ns
	cdef npy_uintp i
	cdef int sigma,sigma_i,sigma_f
	cdef NP_INT8_t r_temp,r,mpz
	cdef NP_INT8_t R[2]
	cdef int j
	
	k = 2.0*_np.pi*kblock*a/L

	if (2*kblock*a) % L == 0: #picks up k = 0, pi modes
		sigma_i = 1
		sigma_f = 1
	else:
		sigma_i = -1
		sigma_f = 1

	Ns = 0
	R[0] = 0
	R[1] = 0
	for i in range(MAX):
		CheckState_T_PZ_template[basis_type](shift,fliplr,flip_all,kblock,L,s,a,R,ns_pars)
		r = R[0]
		mpz = R[1]
		if r > 0:
			if mpz != -1:
				for sigma in range(sigma_i,sigma_f+1,2):
					r_temp = r
					if 1 + sigma*pzblock*cos(mpz*k) == 0:
						r_temp = -1
					if (sigma == -1) and (1 - sigma*pzblock*cos(mpz*k) != 0):
						r_temp = -1
	
					if r_temp > 0:
						m[Ns] = mpz
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






cdef npy_uintp make_t_zA_basis_template(shifter shift,bitop flip_sublat_A,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
												npy_uintp MAX,basis_type s,
												int L,int zAblock,int kblock,int a,
												NP_INT8_t*N, NP_INT8_t*m, object[basis_type,ndim=1,mode="c"] basis): 
	cdef double k 
	cdef npy_uintp i
	cdef npy_uintp Ns
	cdef NP_INT8_t mzA,r
	cdef NP_INT8_t R[2]
	cdef int j	

	k = 2.0*_np.pi*kblock*a/L
	Ns = 0
	R[0] = 0
	R[0] = 0	

	for i in range(MAX):
		CheckState_T_ZA_template[basis_type](shift,flip_sublat_A,kblock,L,s,a,R,ns_pars)
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






cdef npy_uintp make_t_zA_zB_basis_template(shifter shift,bitop flip_sublat_A,bitop flip_sublat_B,
												bitop flip_all,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
												npy_uintp MAX,basis_type s,
												int L,int zAblock,int zBblock,int kblock,int a,
												NP_INT8_t*N,NP_INT16_t*m,object[basis_type,ndim=1,mode="c"] basis): 
	cdef double k 
	cdef npy_uintp Ns
	cdef NP_INT8_t mzA,mzB,mz,r
	cdef npy_uintp i
	cdef NP_INT8_t R[4]
	cdef int j
	
	k = 2.0*_np.pi*kblock*a/L
	Ns = 0
	R[0] = 0
	R[1] = 0
	R[2] = 0
	R[3] = 0
	
	for i in range(MAX):
		CheckState_T_ZA_ZB_template[basis_type](shift,flip_sublat_A,flip_sublat_B,flip_all,kblock,L,s,a,R,ns_pars)
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





cdef npy_uintp make_t_z_basis_template(shifter shift,bitop flip_all,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
											npy_uintp MAX,basis_type s,
											int L,int zblock,int kblock,int a,
											NP_INT8_t*N,NP_INT8_t*m,object[basis_type,ndim=1,mode="c"] basis): 
	cdef double k 
	cdef npy_uintp Ns
	cdef npy_uintp i
	cdef NP_INT8_t mz,r
	cdef NP_INT8_t R[2]
	cdef int j
	

	k = 2.0*_np.pi*kblock*a/L


	Ns = 0
	R[0] = 0
	R[1] = 0

	for i in range(MAX):
		CheckState_T_Z_template[basis_type](shift,flip_all,kblock,L,s,a,R,ns_pars)
		r = R[0]
		mz = R[1]

		if r > 0:
			if mz != -1:
				if 1 + zblock*cos(k*mz) == 0:
					r = -1				

			if r > 0:
				m[Ns] = mz
				N[Ns] = r			
				basis[Ns] = s
				Ns += 1	
		
		s = next_state(s,ns_pars)

	return Ns




	

cdef npy_uintp make_t_zB_basis_template(shifter shift,bitop flip_sublat_B,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
											npy_uintp MAX,basis_type s,
											int L,int zBblock,int kblock,int a,
											NP_INT8_t*N, NP_INT8_t*m,object[basis_type,ndim=1,mode="c"] basis): 
	cdef double k 
	cdef npy_uintp Ns
	cdef npy_uintp i
	cdef NP_INT8_t mzB,r
	cdef NP_INT8_t R[2]
	cdef int j
	

	k = 2.0*_np.pi*kblock*a/L
	Ns = 0
	R[0] = 0
	R[1] = 0
	
	for i in range(MAX):
		CheckState_T_ZB_template[basis_type](shift,flip_sublat_B,kblock,L,s,a,R,ns_pars)
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





cdef npy_uintp make_zA_basis_template(bitop flip_sublat_A,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
											npy_uintp MAX,basis_type s,
											int L, object[basis_type,ndim=1,mode="c"] basis):
	cdef npy_uintp Ns
	cdef NP_INT8_t rzA
	cdef int j
	cdef npy_uintp i

	Ns = 0
	for i in range(MAX):
		rzA = CheckState_ZA_template[basis_type](flip_sublat_A,s,L,ns_pars)
		if rzA > 0:
			basis[Ns] = s
			Ns += 1

		s = next_state(s,ns_pars)

	return Ns






cdef npy_uintp make_zA_zB_basis_template(bitop flip_sublat_A,bitop flip_sublat_B,
											bitop flip_all,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
											npy_uintp MAX,basis_type s,
											int L,object[basis_type,ndim=1,mode="c"] basis):
	cdef npy_uintp Ns
	cdef npy_uintp i
	cdef NP_INT8_t r
	cdef int j

	Ns = 0
	for i in range(MAX):
		r = CheckState_ZA_ZB_template[basis_type](flip_sublat_A,flip_sublat_B,flip_all,s,L,ns_pars)
		if r > 0:
			basis[Ns] = s
			Ns += 1

		s = next_state(s,ns_pars)

	return Ns





cdef npy_uintp make_z_basis_template(bitop flip_all,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
										npy_uintp MAX,basis_type s,
										int L,object[basis_type,ndim=1,mode="c"] basis):
	cdef npy_uintp Ns
	cdef NP_INT8_t rz
	cdef int j
	cdef npy_uintp i

	Ns = 0
	for i in range(MAX):
		rz = CheckState_Z_template[basis_type](flip_all,s,L,ns_pars)
		if rz > 0:
			basis[Ns] = s
			Ns += 1

		s = next_state(s,ns_pars)

	return Ns





cdef npy_uintp make_zB_basis_template(bitop flip_sublat_B,ns_type next_state, object[basis_type,ndim=1,mode="c"] ns_pars,
											npy_uintp MAX,basis_type s,
											int L, object[basis_type,ndim=1,mode="c"] basis):
	cdef npy_uintp Ns
	cdef NP_INT8_t rzB
	cdef int j
	cdef npy_uintp i

	Ns = 0
	for i in range(MAX):
		rzB = CheckState_ZB_template(flip_sublat_B,s,L,ns_pars)
		if rzB > 0:
			basis[Ns] = s
			Ns += 1

		s = next_state(s,ns_pars)

	return Ns





