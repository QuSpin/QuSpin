cdef state_type make_z_basis_template(bitop flip_all,ns_type next_state,
										state_type MAX,state_type s,
										int L,basis_type*basis):
	cdef state_type Ns
	cdef NP_INT8_t rz
	cdef int j
	cdef state_type i

	Ns = 0
	for i in range(MAX):
		rz = CheckState_Z_template(flip_all,s,L)
		if rz > 0:
			basis[Ns] = s
			Ns += 1

		s = next_state(s)

	return Ns
