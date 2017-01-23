



cdef long double complex MatrixElement_ZA_ZB(int L,int zAblock,int zBblock, int kblock, int a, int l, long double k, int gA, int gB,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef long double nr,nc
	cdef long double complex ME
	cdef NP_INT16_t mmr,cr,mmc,cc


	mmc = mc % (L+1)
	cc = mc/(L+1)

	mmr = mr % (L+1)
	cr = mr/(L+1)

	nr = 1.0
	nc = 1.0


	if cr == 1:
		nr = 1.0/Nr
	elif cr == 2:
		nr = (1.0 + zAblock*cos(k*mmr) )/Nr
	elif cr == 3:
		nr = (1.0 + zBblock*cos(k*mmr) )/Nr
	elif cr == 4:
		nr = (1.0 + zAblock*zBblock*cos(k*mmr) )/Nr	
	

	if cc == 1:
		nc = 1.0/Nc
	elif cc == 2:
		nc = (1.0 + zAblock*cos(k*mmc) )/Nc
	elif cc == 3:
		nc = (1.0 + zBblock*cos(k*mmc) )/Nc
	elif cc == 4:
		nc = (1.0 + zAblock*zBblock*cos(k*mmc) )/Nc	
	

	ME=sqrt(nc/nr)*(zAblock**gA)*(zBblock**gB)


	if ((2*a*kblock) % L) == 0:
		ME *= (-1)**(2*l*a*kblock/L)
	else:
		ME *= (cos(k*l) - 1.0j * sin(k*l))

	return ME



cdef int t_zA_zB_op_template(op_type op_func,void *op_pars,shifter shift,bitop flip_sublat_A,bitop flip_sublat_B,bitop flip_all,void *ref_pars,
						int L,int kblock,int zAblock,int zBblock,int a, index_type Ns, NP_INT8_t *N, NP_INT16_t *m,
						basis_type *basis, str opstr, NP_INT32_t *indx, scalar_type J,
						index_type *row, index_type *col, matrix_type *ME):
	cdef state_type s
	cdef index_type ss,i
	cdef int error,l,gA,gB
	cdef basis_type R[4]
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
		RefState_T_ZA_ZB_template(shift,flip_sublat_A,flip_sublat_B,flip_all,s,L,a,&R[0],ref_pars)

		s = R[0]
		l = R[1]
		gA = R[2]
		gB = R[3]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss
		if ss == -1: continue

		if (matrix_type is float or matrix_type is double or matrix_type is longdouble):
			me = MatrixElement_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
			ME[i] *= me.real
		else:
			ME[i] *= MatrixElement_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])


	return error







