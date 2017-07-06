




cdef inline basis_type next_state_pcon_hcp(basis_type s,basis_type[:] pars):
	if s == 0 :
		return s

	cdef basis_type t = (s | (s - 1)) + 1
	return t | ((((t & -t) // (s & -s)) >> 1) - 1)




cdef inline basis_type next_state_inc_1(basis_type s,basis_type[:] pars):
	return s + 1