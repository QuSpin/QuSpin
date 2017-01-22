
cdef state_type make_p_z_basis_template(bitop fliplr,bitop flip_all,ns_type next_state, void *ns_pars,
									state_type MAX,state_type s,int L, int pblock, int zblock,
									NP_INT8_t * N, basis_type * basis):
	cdef state_type Ns
	cdef NP_INT8_t rpz
	cdef state_type i

	Ns = 0
	for i in range(MAX):
		rpz = CheckState_P_Z_template(fliplr,flip_all,pblock,zblock,s,L,ns_pars)
		if rpz > 0:
			basis[Ns] = s
			N[Ns] = rpz
			Ns += 1
		s = next_state(s,ns_pars)

	return Ns