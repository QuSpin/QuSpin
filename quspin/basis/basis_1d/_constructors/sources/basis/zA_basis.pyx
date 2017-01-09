cdef state_type make_zA_basis_template(bitop flip_sublat_A,ns_type next_state,
											state_type MAX,state_type s,
											int L, basis_type*basis):
	cdef state_type Ns
	cdef NP_INT8_t rzA
	cdef int j
	cdef state_type i

	Ns = 0
	for i in range(MAX):
		rzA = CheckState_ZA_template(flip_sublat_A,s,L)
		if rzA > 0:
			basis[Ns] = s
			Ns += 1

		s = next_state(s)

	return Ns


"""
def make_m_zA_basis(int L, int Nup, _np.ndarray[NP_UINT32_t,ndim=1] basis):
	cdef unsigned int s,Ns
	cdef NP_INT8_t rzA
	cdef char stp
	cdef int j

	s = 0
	for j in range(Nup):
		s += ( 1ull << j )

	if (Nup == L) or (Nup == 0):
		return 0

	stp = 0
	Ns = 0
	while True:

		rzA = CheckState_ZA_template(flip_sublat_A,s,L)
		if rzA > 0:
			basis[Ns] = s
			Ns += 1

		stp = 1 & ( s >> (L-1) ) 
		for i in range(1,Nup):
			stp &= 1 & ( s >> (L-i-1) )

		if stp or (s == 0):
			break

		s = next_state(s)


	return Ns




def make_zA_basis(int L, _np.ndarray[NP_UINT32_t,ndim=1] basis):
	cdef state_type s
	cdef int Ns
	cdef NP_INT8_t rzA

	Ns = 0

	for s in range(1ull << L):
		rzA = CheckState_ZA_template(flip_sublat_A,s,L)
		if rzA > 0:
			basis[Ns] = s
			Ns += 1

	return Ns
"""