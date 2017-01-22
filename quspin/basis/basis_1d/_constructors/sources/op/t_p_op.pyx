

cdef long double MatrixElement_P(int L,int pblock, int kblock, int a, int l, long double k, int q,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
	cdef long double nr,nc
	cdef long double ME
	cdef NP_INT8_t sr,sc

	if Nr > 0:
		sr = 1
	else:
		sr = -1

	if Nc > 0:
		sc = 1
	else:
		sc = -1


	if mr >= 0:
		nr = (1 + sr*pblock*cosl(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pblock*cosl(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc


	ME=sqrt(nc/nr)*(sr*pblock)**q

	if sr == sc :
		if mc < 0:
			ME *= cosl(k*l)
		else:
			ME *= (cosl(k*l)+sr*pblock*cosl((l-mc)*k))/(1+sr*pblock*cosl(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sinl(k*l)
		else:
			ME *= (-sr*sinl(k*l)+pblock*sinl((l-mc)*k))/(1-sr*pblock*cosl(k*mc))		


	return ME






cdef int t_p_op_template(op_type op_func,void *op_pars,shifter shift,bitop fliplr,void *ref_pars,int L,int kblock,int pblock,int a, index_type Ns,
						NP_INT8_t *N, NP_INT8_t *m, basis_type *basis, str opstr, NP_INT32_t *indx, scalar_type J,
						index_type *row, index_type *col, matrix_type *ME):
	cdef state_type s
	cdef index_type ss,i,j,o,p,c,b,l,q
	cdef int error
	cdef basis_type R[3]


	cdef long double k = (2.0*_np.pi*kblock*a)/L

	error = op_func(Ns,&basis[0],opstr,&indx[0],J,&row[0],&ME[0],op_pars)

	if error != 0:
		return error

	for i in range(Ns):
		ME[Ns+i] = ME[i]
		row[Ns+i] = -1
		col[i] = i
		col[Ns+i] = i


	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		for i in range(Ns):
			s = row[i]
			RefState_T_P_template(shift,fliplr,s,L,a,&R[0],ref_pars)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(&basis[0],Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P_template(shift,fliplr,s,L,a,&R[0],ref_pars)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(&basis[0],Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= MatrixElement_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

			else: # off diagonal ME

				if (ss > 0) and (basis[ss] == basis[ss-1]):
					ss -= 1; p = 2
				elif (ss < (Ns - 1)) and (basis[ss] == basis[ss+1]):
					p = 2
				else:
					p = 1

				for c in range(0,o,1):
					for b in range(0,p,1):
						j = i + c + Ns*b
						row[j] = ss + b
						ME[j] *= MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return error

