


cdef int spin_op(state_type Ns, basis_type *basis,
				int N_indx, str opstr,NP_INT32_t *indx, complex_type J,
				index_type *row, matrix_type *ME, complex_type M_E):

	cdef state_type i
	cdef state_type r,b
	cdef int j,error
	cdef char a

	cdef unsigned char[:] c_opstr = bytearray(opstr,"utf-8")

	cdef char I = "I"
	cdef char x = "x"
	cdef char y = "y"
	cdef char z = "z"
	cdef char p = "+"
	cdef char m = "-"

	error = 0

	for i in range(Ns): #loop over basis
		M_E = J
		r = basis[i]
		
		for j in range(N_indx-1,-1,-1): #loop over the copstr

			b = ( 1ull << indx[j] ) #put the bit 1 at the place of the bit corresponding to the site indx[j]; ^b = flipbil
			a = ( r >> indx[j] ) & 1 #checks whether spin at site indx[j] is 1 ot 0; a = return of testbit

			if c_opstr[j] == I:
				continue
			elif c_opstr[j] == z:
				M_E *= (-1.0)**(a+1)
			elif c_opstr[j] == x:
				r = r ^ b
			elif c_opstr[j] == y:
				r = r ^ b
				M_E *= 1.0j*(-1.0)**(a+1)
			elif c_opstr[j] == p:
				if a == 1:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
				
			elif c_opstr[j] == m:
				if a == 0:
					r = -1
					M_E = 0.0
					break
				r = r ^ b
				M_E *= 2
		
			else:
				error = 1
				return error

		if matrix_type is NP_FLOAT32_t or matrix_type is NP_FLOAT64_t:
			if M_E.imag != 0.0:
				error = -1
				return error

			ME[i] = M_E.real
			row[i] = r
		else:
			ME[i] = M_E
			row[i] = r

	return error




cdef int op(op_type op_func, state_type Ns, basis_type *basis,
			str opstr, NP_INT32_t *indx, complex_type J, index_type *row, matrix_type *ME):

	cdef int N_op = len(opstr)
	return op_func(Ns,&basis[0],N_op,opstr,&indx[0],J,&row[0],&ME[0],1.0)



"""
cdef int n_op(op_type op_func, state_type Ns, basis_type *basis,
			str opstr, NP_INT32_t *indx, complex_type J, index_type *row, matrix_type *ME):
	cdef int error = 0
	cdef int N_op = len(opstr)
	error = op_func(Ns,&basis[0],N_op,opstr,&indx[0],J,&row[0],&ME[0],1.0)

	if error != 0:
		return error

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(basis,Ns,s)

	return error
"""