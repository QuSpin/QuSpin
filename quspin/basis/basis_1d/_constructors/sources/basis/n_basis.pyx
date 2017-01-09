
cdef void make_n_basis_template(ns_type next_state,state_type MAX,state_type s,basis_type*basis):
	cdef state_type i
	for i in range(MAX):
		basis[i] = s
		s = next_state(s)	


