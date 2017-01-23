


cdef int t_op_template(op_type op_func,void *op_pars,shifter shift,void *ref_pars,int L,int kblock,int a, index_type Ns,
						NP_INT8_t *N, basis_type *basis, str opstr, NP_INT32_t *indx, scalar_type J,
						index_type *row, index_type *col, matrix_type *ME):
	cdef state_type s
	cdef index_type ss,i
	cdef int error,l
	cdef basis_type R[2]
	cdef long double n,k

	k = (2.0*_np.pi*kblock*a)/L

	error = op_func(Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0],op_pars)

	if (matrix_type is float or matrix_type is double or matrix_type is longdouble) and ((2*a*kblock) % L) != 0:
		error = -1

	if error != 0:
		return error

	for i in range(Ns):
		col[i] = i

	for i in range(Ns):
		s = row[i]
		RefState_T_template(shift,s,L,a,&R[0],ref_pars)

		s = R[0]
		l = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[i]
		n /= N[ss]
		n = sqrtl(n)
		ME[i] *= n

		if (matrix_type is float or matrix_type is double or matrix_type is longdouble):
			ME[i] *= (-1.0)**(l*2*a*kblock/L)
		else:
			ME[i] *= (cosl(k*l) - 1.0j * sinl(k*l))

			

	return error




