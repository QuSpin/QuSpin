


def make_m_basis(int L, int Nup, unsigned int Ns,_np.ndarray[NP_UINT32_t,ndim=1] basis):
	cdef unsigned int s,i
	cdef int j

	s = 0
	for j in range(Nup):
		s += ( 1ull << j )

	if (Nup == L) or (Nup == 0):
		basis[0] = s
		return basis

	for i in range(Ns):
		basis[i] = s
		s = next_state(s)


