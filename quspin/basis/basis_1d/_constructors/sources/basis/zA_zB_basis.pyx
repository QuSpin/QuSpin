
cdef state_type make_zA_zB_basis_template(bitop flip_sublat_A,bitop flip_sublat_B,
											bitop flip_all,ns_type next_state, void *ns_pars,
											state_type MAX,state_type s,
											int L,basis_type*basis):
	cdef state_type Ns
	cdef state_type i
	cdef NP_INT8_t r
	cdef int j

	Ns = 0
	for i in range(MAX):
		r = CheckState_ZA_ZB_template(flip_sublat_A,flip_sublat_B,flip_all,s,L,ns_pars)
		if r > 0:
			basis[Ns] = s
			Ns += 1

		s = next_state(s,ns_pars)

	return Ns
