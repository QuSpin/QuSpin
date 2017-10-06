


cdef basis_type next_state_pcon_boson(basis_type s,basis_type[:] pars):
	"""
	Returns integer representation of the next bosonic state.

	---arguments:

	s: integer representation of bosonic state to be shifted
	L: total number of lattice sites
	sps: number of states per site
	M = [sps**i for i in range(L)]
	"""
	cdef int L = pars[0]
	cdef basis_type[:] M = pars[1:]
	cdef basis_type N = 0
	cdef basis_type n = 0
	cdef basis_type sps = M[1]
	cdef basis_type b1,b2,N_left
	cdef basis_type t = s
	cdef int i,j,l
	cdef object o

	for i in range(L-1):
		b1 = (s//M[i])%sps
		if b1 > 0:
			N += b1
			b2 = (s//M[i+1])%sps
			if b2 < (sps-1):
				N -= 1
				# shift one particle right 
				s -= M[i] 
				s += M[i+1]

				if N > 0:
					# find the length of sub system which that number fits into (up to a remainder)
					l = N//(sps-1)
					N_left = N%(sps-1)
					for j in range(i+1):
						s -= ((s//M[j])%sps)*M[j] # subtract particles from site j

						if j < l:
							s += (sps-1)*M[j]
						elif j == l:
							s += N_left*M[j]

				return s


cdef inline basis_type next_state_inc_1(basis_type s,basis_type[:] pars):
	return s + 1