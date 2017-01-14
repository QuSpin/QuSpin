
cdef int n_op_template(op_type op_func, state_type Ns, basis_type *basis,
			str opstr, NP_INT32_t *indx, scalar_type J, index_type *row, matrix_type *ME):
	cdef long long s
	cdef int error = 0
	cdef int N_op = len(opstr)
	error = op_func(Ns,&basis[0],N_op,opstr,&indx[0],J,&row[0],&ME[0])

	if error != 0:
		return error

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(&basis[0],Ns,s)

	return error
