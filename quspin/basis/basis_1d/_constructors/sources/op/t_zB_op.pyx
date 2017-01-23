

cdef long double complex MatrixElement_ZB(int L,int zBblock, int kblock, int a, int l, long double k, int gB,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
	cdef long double nr,nc
	cdef long double complex ME
	

	if mr >=0:
		nr = (1 + zBblock*cosl(k*mr))/Nr
	else:
		nr = 1.0/Nr

	if mc >= 0:
		nc = (1 + zBblock*cosl(k*mc))/Nc
	else:
		nc = 1.0/Nc


	ME=sqrtl(nc/nr)*(zBblock**gB)

	if (2*kblock*a % L) == 0:
		ME *= (-1)**(2*l*a*kblock/L)
	else:
		ME *= (cosl(k*l) - 1.0j * sinl(k*l))

	return ME



cdef int t_zB_op_template(op_type op_func,void *op_pars,shifter shift,bitop flip_sublat_B,void *ref_pars,int L,int kblock,int zBblock,int a, index_type Ns,
						NP_INT8_t *N, NP_INT8_t *m, basis_type *basis, str opstr, NP_INT32_t *indx, scalar_type J,
						index_type *row, index_type *col, matrix_type *ME):
	cdef state_type s
	cdef index_type ss,i 
	cdef int error,l,gB
	cdef basis_type R[3]
	cdef long double k = (2.0*_np.pi*kblock*a)/L
	cdef long double complex me


	error = op_func(Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0],op_pars)

	if (matrix_type is float or matrix_type is double or matrix_type is longdouble) and ((2*a*kblock) % L) != 0:
		error = -1

	if error != 0:
		return error

	for i in range(Ns):
		col[i] = i

	for i in range(Ns):
		s = row[i]
		RefState_T_ZB_template(shift,flip_sublat_B,s,L,a,&R[0],ref_pars)

		s = R[0]
		l = R[1]
		gB = R[2]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss
		if ss == -1: continue

		if (matrix_type is float or matrix_type is double or matrix_type is longdouble):
			me = MatrixElement_ZB(L,zBblock,kblock,a,l,k,gB,N[i],N[ss],m[i],m[ss])
			ME[i] *= me.real
		else:
			ME[i] *= MatrixElement_ZB(L,zBblock,kblock,a,l,k,gB,N[i],N[ss],m[i],m[ss])


	return error


