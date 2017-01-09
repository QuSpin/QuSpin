
cdef state_type make_pz_basis_template(bitop fliplr,bitop flip_all,ns_type next_state,
										state_type MAX,state_type s,int L, int pzblock,
										NP_INT8_t * N, basis_type * basis):
	cdef state_type Ns
	cdef NP_INT8_t rpz
	cdef state_type i

	Ns = 0
	for i in range(MAX):
		rpz = CheckState_PZ_template(fliplr,flip_all,pzblock,s,L)
		if rpz > 0:
			basis[Ns] = s
			N[Ns] = rpz
			Ns += 1
		
		s = next_state(s)

	return Ns