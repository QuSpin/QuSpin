






cdef int p_z_op_template(op_type op_func,void *op_pars,bitop fliplr,bitop flip_all,void *ref_pars,int L,int pblock,int zblock, index_type Ns,
						NP_INT8_t *N, basis_type *basis, str opstr, NP_INT32_t *indx, scalar_type J,
						index_type *row, index_type *col, matrix_type *ME):
	cdef state_type s
	cdef index_type ss,i
	cdef int error,q,g
	cdef basis_type R[3]
	cdef long double n


	error = op_func(Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0],op_pars)

	if error != 0:
		return error

	for i in range(Ns):
		col[i] = i

	for i in range(Ns):
		s = row[i]
		RefState_P_Z_template(fliplr,flip_all,s,L,&R[0],ref_pars)

		s = R[0]
		q = R[1]
		g = R[2]
		
		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[ss]
		n /= N[i]

		ME[i] *= sqrtl(n)*(pblock**q)*(zblock**g)

	return error







