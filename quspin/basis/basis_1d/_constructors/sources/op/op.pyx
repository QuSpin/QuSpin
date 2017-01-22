
cdef int op_template(op_type op_func, void *op_pars, index_type Ns, basis_type *basis,
					 str opstr, NP_INT32_t *indx, scalar_type J, index_type *row, index_type *col, matrix_type *ME):
	cdef index_type i

	for i in range(Ns):
		col[i] = i

	return op_func(Ns,basis,opstr,indx,J,row,ME,op_pars)
