
cdef int op(op_type op_func, state_type Ns, basis_type *basis,
            str opstr, NP_INT32_t *indx, scalar_type J, index_type *row, matrix_type *ME):

    cdef int N_op = len(opstr)
    return op_func(Ns,basis,N_op,opstr,indx,J,row,ME)