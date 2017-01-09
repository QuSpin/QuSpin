
cdef state_type make_t_basis_template(shifter shift,ns_type next_state,
										state_type MAX,state_type s,int L,
										int kblock,int a,NP_INT8_t * N, basis_type * basis):
	cdef state_type Ns
	cdef NP_INT8_t r
	cdef state_type i

	Ns = 0
	for i in range(MAX):
		r=CheckState_T_template(shift,kblock,L,s,a)
		if r > 0:
			N[Ns] = r				
			basis[Ns] = s
			Ns += 1		
		
		s = next_state(s)

	return Ns