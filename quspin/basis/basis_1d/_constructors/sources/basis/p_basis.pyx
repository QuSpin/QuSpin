
cdef state_type make_p_basis_template(bitop fliplr,ns_type next_state,
											state_type MAX,state_type s,
											int L,int pblock, NP_INT8_t * N,basis_type * basis):
	cdef state_type Ns
	cdef NP_INT8_t rp

	cdef int j
	cdef state_type i

	Ns = 0

	for i in range(MAX):
		rp = CheckState_P_template(fliplr,pblock,s,L)
		if rp > 0:
			basis[Ns] = s
			N[Ns] = rp
			Ns += 1

		s = next_state(s)

	return Ns