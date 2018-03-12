
cdef inline basis_type next_state_pcon(basis_type v,basis_type[:] pars):
	if v == 0 :
		return v

	cdef basis_type t = (v | (v - 1)) + 1
	return t | ((((t & -t) / (v & -v)) >> 1) - 1)


cdef basis_type next_state_pcon_spf(basis_type v, basis_type[:] pars):
	cdef basis_type L = pars[0]
	cdef basis_type MAX_right = pars[2]
	cdef basis_type MIN_right = pars[3]

	cdef basis_type s_right = v & pars[1]
	cdef basis_type s_left = v >> L
	if s_right < MAX_right:
		s_right = next_state_pcon[basis_type](s_right,None)
	else:
		s_left = next_state_pcon[basis_type](s_left,None)
		s_right = MIN_right
		
	return s_right+(s_left<<L)


cdef basis_type next_state_inc_1(basis_type v,basis_type[:] pars):
	return v + 1
