

cdef int p_op_template(op_type op_func,bitop fliplr,int L,int pblock, state_type Ns,
				NP_INT8_t *N, basis_type *basis, str opstr, NP_INT32_t *indx, scalar_type J,
				index_type *row, matrix_type *ME):

	cdef state_type s,i
	cdef long long ss
	cdef int error = 0
	cdef int q
	cdef int N_op = len(opstr)
	cdef basis_type R[2]
	cdef long double n

	error = op_func(Ns,&basis[0],N_op,opstr,&indx[0],J,&row[0],&ME[0])

	if error != 0:
		return error

	for i in range(Ns):
		s = row[i]
		RefState_P_template(fliplr,s,L,&R[0])

		s = R[0]
		q = R[1]
		
		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss
		if ss == -1: continue

		n = N[ss]
		n /= N[i]
		ME[i] *= (pblock**q)*sqrtl(n)

	return error




