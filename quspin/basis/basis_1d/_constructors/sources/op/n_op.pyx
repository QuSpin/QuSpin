
cdef int n_op_template(op_type op_func,void *op_pars, index_type Ns, basis_type *basis,
			str opstr, NP_INT32_t *indx, scalar_type J, index_type *row, index_type *col, matrix_type *ME):
	cdef index_type i
	cdef state_type s
	cdef int error = 0
	error = op_func(Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0],op_pars)

	if error != 0:
		return error

	for i in range(Ns):
		col[i] = i

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(&basis[0],Ns,s)

	return error
