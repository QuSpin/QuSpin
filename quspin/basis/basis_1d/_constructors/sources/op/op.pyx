
cdef int op_template(op_type op_func, index_type Ns, basis_type *basis,
            str opstr, NP_INT32_t *indx, scalar_type J, index_type *row, matrix_type *ME):

    return op_func(Ns,basis,opstr,indx,J,row,ME)
