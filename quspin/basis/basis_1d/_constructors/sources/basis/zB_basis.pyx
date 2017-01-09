cdef state_type make_zB_basis_template(bitop flip_sublat_B,ns_type next_state,
											state_type MAX,state_type s,
											int L, basis_type*basis):
	cdef state_type Ns
	cdef NP_INT8_t rzB
	cdef int j
	cdef state_type i

	Ns = 0
	for i in range(MAX):
		rzB = CheckState_ZB_template(flip_sublat_B,s,L)
		if rzB > 0:
			basis[Ns] = s
			Ns += 1

		s = next_state(s)

	return Ns
