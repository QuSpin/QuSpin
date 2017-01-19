







cdef int pz_op_template(op_type op_func,bitop fliplr,bitop flip_all,int L,int pzblock, index_type Ns,
						NP_INT8_t *N, basis_type *basis, str opstr, NP_INT32_t *indx, scalar_type J,
						index_type *row, matrix_type *ME):
	cdef state_type s
	cdef index_type ss,i
	cdef int error,qg
	cdef basis_type[2] R
	cdef long double n

	error = op_func(Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0])

	if error != 0:
		return error

	for i in range(Ns):
		s = row[i]
		RefState_PZ_template(fliplr,flip_all,s,L,&R[0])

		s = R[0]
		qg = R[1]
		
		ss = findzstate(&basis[0],Ns,s)

		row[i] = ss
		if ss == -1: continue

		n = N[ss]
		n /= N[i]

		ME[i] *= (pzblock**qg)*sqrtl(n)

	return error





