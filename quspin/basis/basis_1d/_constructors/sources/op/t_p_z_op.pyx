


cdef long double MatrixElement_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, long double k, int q, int g,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef long double nr,nc
	cdef long double ME
	cdef NP_INT8_t sr,sc
	cdef NP_INT16_t nnr,mmr,cr,nnc,mmc,cc

	# define sign function
	if Nr > 0:
		sr = 1
	else:
		sr = -1

	if Nc > 0:
		sc = 1
	else:
		sc = -1

	# unpack long integer, cf Anders' notes
	mmc = mc % (L+1)
	nnc = (mc/(L+1)) % (L+1)
	cc = mc/((L+1)*(L+1))

	mmr = mr % (L+1)
	nnr = (mr/(L+1)) % (L+1)
	cr = mr/((L+1)*(L+1))

	nr = 1.0
	nc = 1.0

	if cr == 1:
		nr = 1.0/Nr
	elif cr == 2:
		nr = (1.0 + pblock*sr*cosl(k*mmr))/Nr
	elif cr == 3:
		nr = (1.0 + zblock*cosl(k*nnr))/Nr
	elif cr == 4:
		nr = (1.0 + pblock*zblock*sr*cosl(k*mmr))/Nr	
	elif cr == 5:
		nr = (1.0 + pblock*sr*cosl(k*mmr))*(1.0 + zblock*cosl(k*nnr))/Nr

	nr *= sr

	if cc == 1:
		nc = 1.0/Nc
	elif cc == 2:
		nc = (1.0 + pblock*sc*cosl(k*mmc))/Nc
	elif cc == 3:
		nc = (1.0 + zblock*cosl(k*nnc))/Nc
	elif cc == 4:
		nc = (1.0 + pblock*zblock*sc*cosl(k*mmc))/Nc	
	elif cc == 5:
		nc = (1.0 + pblock*sc*cosl(k*mmc))*(1.0 + zblock*cosl(k*nnc))/Nc

	nc *= sc



	ME=sqrtl(nc/nr)*((sr*pblock)**q)*(zblock**g)

	if sr == sc :
		if (cc == 1) or (cc == 3):
			ME *= cosl(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (cosl(k*l)+sr*pblock*cosl((l-mmc)*k))/(1+sr*pblock*cosl(k*mmc))
		elif (cc == 4):
			ME *= (cosl(k*l)+sr*pblock*zblock*cosl((l-mmc)*k))/(1+sr*pblock*zblock*cosl(k*mmc))
	else:
		if (cc == 1) or (cc == 3):
			ME *= -sr*sinl(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (-sr*sinl(k*l) + pblock*sinl((l-mmc)*k))/(1-sr*pblock*cosl(k*mmc))
		elif (cc == 4):
			ME *= (-sr*sinl(k*l) + pblock*zblock*sinl((l-mmc)*k))/(1-sr*pblock*zblock*cosl(k*mmc))

	return ME






cdef int t_p_z_op_template(op_type op_func,void *op_pars,shifter shift,bitop fliplr,bitop flip_all,void *ref_pars,int L,int kblock,int pblock,int zblock,int a,
						index_type Ns, NP_INT8_t *N, NP_INT16_t *m, basis_type *basis, str opstr, NP_INT32_t *indx,
						scalar_type J, index_type *row, index_type *col, matrix_type *ME):
	cdef state_type s
	cdef index_type ss,i,j,o,p,c,b,l,q,g
	cdef int error
	cdef basis_type R[4]

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
			RefState_T_P_Z_template(shift,fliplr,flip_all,s,L,a,&R[0],ref_pars)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(&basis[0],Ns,s)
			row[i] = ss

			if ss == -1:
				continue

			ME[i] *= MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P_Z_template(shift,fliplr,flip_all,s,L,a,&R[0],ref_pars)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(&basis[0],Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

				for j in range(i,i+o,1):
					row[j] = j
					ME[j] *= MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

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
						ME[j] *= MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return error




