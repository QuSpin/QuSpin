
cdef int zA_op_template(op_type op_func, void *op_pars,bitop flip_sublat_A,void *ref_pars,int L,int zAblock, index_type Ns,
						basis_type *basis, str opstr, NP_INT32_t *indx, scalar_type J,
						index_type *row, index_type *col, matrix_type *ME):
	cdef state_type s
	cdef index_type ss,i
	cdef int error,gA
	cdef basis_type R[2]

	error = op_func(Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0],op_pars)

	if error != 0:
		return error

	for i in range(Ns):
		col[i] = i

	for i in range(Ns):
		s = row[i]
		RefState_ZA_template(flip_sublat_A,s,L,&R[0],ref_pars)

		s = R[0]
		gA = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zAblock)**gA


	return error






