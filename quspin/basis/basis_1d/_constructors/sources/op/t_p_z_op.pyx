


cdef float f_MatrixElement_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, float k, int q, int g,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef float nr,nc
	cdef float ME
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
		nr = (1.0 + pblock*sr*cos(k*mmr))/Nr
	elif cr == 3:
		nr = (1.0 + zblock*cos(k*nnr))/Nr
	elif cr == 4:
		nr = (1.0 + pblock*zblock*sr*cos(k*mmr))/Nr	
	elif cr == 5:
		nr = (1.0 + pblock*sr*cos(k*mmr))*(1.0 + zblock*cos(k*nnr))/Nr

	nr *= sr

	if cc == 1:
		nc = 1.0/Nc
	elif cc == 2:
		nc = (1.0 + pblock*sc*cos(k*mmc))/Nc
	elif cc == 3:
		nc = (1.0 + zblock*cos(k*nnc))/Nc
	elif cc == 4:
		nc = (1.0 + pblock*zblock*sc*cos(k*mmc))/Nc	
	elif cc == 5:
		nc = (1.0 + pblock*sc*cos(k*mmc))*(1.0 + zblock*cos(k*nnc))/Nc

	nc *= sc



	ME=sqrt(nc/nr)*((sr*pblock)**q)*(zblock**g)

	if sr == sc :
		if (cc == 1) or (cc == 3):
			ME *= cos(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (cos(k*l)+sr*pblock*cos((l-mmc)*k))/(1+sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (cos(k*l)+sr*pblock*zblock*cos((l-mmc)*k))/(1+sr*pblock*zblock*cos(k*mmc))
	else:
		if (cc == 1) or (cc == 3):
			ME *= -sr*sin(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (-sr*sin(k*l) + pblock*sin((l-mmc)*k))/(1-sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (-sr*sin(k*l) + pblock*zblock*sin((l-mmc)*k))/(1-sr*pblock*zblock*cos(k*mmc))

	return ME






def f_t_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int pblock,int zblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float,ndim=1] ME

	cdef double n
	cdef float k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = f_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	row = _np.resize(row,(2*Ns,))
	row[Ns:] = -1
	ME = _np.resize(ME,(2*Ns,))

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		for i in range(Ns):
			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= f_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= f_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

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
						ME[j] *= f_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef double d_MatrixElement_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, double k, int q, int g,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef double nr,nc
	cdef double ME
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
		nr = (1.0 + pblock*sr*cos(k*mmr))/Nr
	elif cr == 3:
		nr = (1.0 + zblock*cos(k*nnr))/Nr
	elif cr == 4:
		nr = (1.0 + pblock*zblock*sr*cos(k*mmr))/Nr	
	elif cr == 5:
		nr = (1.0 + pblock*sr*cos(k*mmr))*(1.0 + zblock*cos(k*nnr))/Nr

	nr *= sr

	if cc == 1:
		nc = 1.0/Nc
	elif cc == 2:
		nc = (1.0 + pblock*sc*cos(k*mmc))/Nc
	elif cc == 3:
		nc = (1.0 + zblock*cos(k*nnc))/Nc
	elif cc == 4:
		nc = (1.0 + pblock*zblock*sc*cos(k*mmc))/Nc	
	elif cc == 5:
		nc = (1.0 + pblock*sc*cos(k*mmc))*(1.0 + zblock*cos(k*nnc))/Nc

	nc *= sc



	ME=sqrt(nc/nr)*((sr*pblock)**q)*(zblock**g)

	if sr == sc :
		if (cc == 1) or (cc == 3):
			ME *= cos(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (cos(k*l)+sr*pblock*cos((l-mmc)*k))/(1+sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (cos(k*l)+sr*pblock*zblock*cos((l-mmc)*k))/(1+sr*pblock*zblock*cos(k*mmc))
	else:
		if (cc == 1) or (cc == 3):
			ME *= -sr*sin(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (-sr*sin(k*l) + pblock*sin((l-mmc)*k))/(1-sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (-sr*sin(k*l) + pblock*zblock*sin((l-mmc)*k))/(1-sr*pblock*zblock*cos(k*mmc))

	return ME






def d_t_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int pblock,int zblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double,ndim=1] ME

	cdef double n
	cdef double k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = d_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	row = _np.resize(row,(2*Ns,))
	row[Ns:] = -1
	ME = _np.resize(ME,(2*Ns,))

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		for i in range(Ns):
			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= d_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= d_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

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
						ME[j] *= d_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef float F_MatrixElement_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, float k, int q, int g,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef float nr,nc
	cdef float ME
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
		nr = (1.0 + pblock*sr*cos(k*mmr))/Nr
	elif cr == 3:
		nr = (1.0 + zblock*cos(k*nnr))/Nr
	elif cr == 4:
		nr = (1.0 + pblock*zblock*sr*cos(k*mmr))/Nr	
	elif cr == 5:
		nr = (1.0 + pblock*sr*cos(k*mmr))*(1.0 + zblock*cos(k*nnr))/Nr

	nr *= sr

	if cc == 1:
		nc = 1.0/Nc
	elif cc == 2:
		nc = (1.0 + pblock*sc*cos(k*mmc))/Nc
	elif cc == 3:
		nc = (1.0 + zblock*cos(k*nnc))/Nc
	elif cc == 4:
		nc = (1.0 + pblock*zblock*sc*cos(k*mmc))/Nc	
	elif cc == 5:
		nc = (1.0 + pblock*sc*cos(k*mmc))*(1.0 + zblock*cos(k*nnc))/Nc

	nc *= sc



	ME=sqrt(nc/nr)*((sr*pblock)**q)*(zblock**g)

	if sr == sc :
		if (cc == 1) or (cc == 3):
			ME *= cos(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (cos(k*l)+sr*pblock*cos((l-mmc)*k))/(1+sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (cos(k*l)+sr*pblock*zblock*cos((l-mmc)*k))/(1+sr*pblock*zblock*cos(k*mmc))
	else:
		if (cc == 1) or (cc == 3):
			ME *= -sr*sin(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (-sr*sin(k*l) + pblock*sin((l-mmc)*k))/(1-sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (-sr*sin(k*l) + pblock*zblock*sin((l-mmc)*k))/(1-sr*pblock*zblock*cos(k*mmc))

	return ME






def F_t_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int pblock,int zblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float complex,ndim=1] ME

	cdef double n
	cdef float k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = F_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	row = _np.resize(row,(2*Ns,))
	row[Ns:] = -1
	ME = _np.resize(ME,(2*Ns,))

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		for i in range(Ns):
			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= F_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= F_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

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
						ME[j] *= F_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef double D_MatrixElement_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, double k, int q, int g,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef double nr,nc
	cdef double ME
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
		nr = (1.0 + pblock*sr*cos(k*mmr))/Nr
	elif cr == 3:
		nr = (1.0 + zblock*cos(k*nnr))/Nr
	elif cr == 4:
		nr = (1.0 + pblock*zblock*sr*cos(k*mmr))/Nr	
	elif cr == 5:
		nr = (1.0 + pblock*sr*cos(k*mmr))*(1.0 + zblock*cos(k*nnr))/Nr

	nr *= sr

	if cc == 1:
		nc = 1.0/Nc
	elif cc == 2:
		nc = (1.0 + pblock*sc*cos(k*mmc))/Nc
	elif cc == 3:
		nc = (1.0 + zblock*cos(k*nnc))/Nc
	elif cc == 4:
		nc = (1.0 + pblock*zblock*sc*cos(k*mmc))/Nc	
	elif cc == 5:
		nc = (1.0 + pblock*sc*cos(k*mmc))*(1.0 + zblock*cos(k*nnc))/Nc

	nc *= sc



	ME=sqrt(nc/nr)*((sr*pblock)**q)*(zblock**g)

	if sr == sc :
		if (cc == 1) or (cc == 3):
			ME *= cos(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (cos(k*l)+sr*pblock*cos((l-mmc)*k))/(1+sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (cos(k*l)+sr*pblock*zblock*cos((l-mmc)*k))/(1+sr*pblock*zblock*cos(k*mmc))
	else:
		if (cc == 1) or (cc == 3):
			ME *= -sr*sin(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (-sr*sin(k*l) + pblock*sin((l-mmc)*k))/(1-sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (-sr*sin(k*l) + pblock*zblock*sin((l-mmc)*k))/(1-sr*pblock*zblock*cos(k*mmc))

	return ME






def D_t_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int pblock,int zblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double complex,ndim=1] ME

	cdef double n
	cdef double k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = D_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	row = _np.resize(row,(2*Ns,))
	row[Ns:] = -1
	ME = _np.resize(ME,(2*Ns,))

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		for i in range(Ns):
			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= D_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= D_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

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
						ME[j] *= D_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef long double g_MatrixElement_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, long double k, int q, int g,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
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
		nr = (1.0 + pblock*sr*cos(k*mmr))/Nr
	elif cr == 3:
		nr = (1.0 + zblock*cos(k*nnr))/Nr
	elif cr == 4:
		nr = (1.0 + pblock*zblock*sr*cos(k*mmr))/Nr	
	elif cr == 5:
		nr = (1.0 + pblock*sr*cos(k*mmr))*(1.0 + zblock*cos(k*nnr))/Nr

	nr *= sr

	if cc == 1:
		nc = 1.0/Nc
	elif cc == 2:
		nc = (1.0 + pblock*sc*cos(k*mmc))/Nc
	elif cc == 3:
		nc = (1.0 + zblock*cos(k*nnc))/Nc
	elif cc == 4:
		nc = (1.0 + pblock*zblock*sc*cos(k*mmc))/Nc	
	elif cc == 5:
		nc = (1.0 + pblock*sc*cos(k*mmc))*(1.0 + zblock*cos(k*nnc))/Nc

	nc *= sc



	ME=sqrt(nc/nr)*((sr*pblock)**q)*(zblock**g)

	if sr == sc :
		if (cc == 1) or (cc == 3):
			ME *= cos(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (cos(k*l)+sr*pblock*cos((l-mmc)*k))/(1+sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (cos(k*l)+sr*pblock*zblock*cos((l-mmc)*k))/(1+sr*pblock*zblock*cos(k*mmc))
	else:
		if (cc == 1) or (cc == 3):
			ME *= -sr*sin(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (-sr*sin(k*l) + pblock*sin((l-mmc)*k))/(1-sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (-sr*sin(k*l) + pblock*zblock*sin((l-mmc)*k))/(1-sr*pblock*zblock*cos(k*mmc))

	return ME






def g_t_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int pblock,int zblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double,ndim=1] ME

	cdef double n
	cdef long double k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = g_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	row = _np.resize(row,(2*Ns,))
	row[Ns:] = -1
	ME = _np.resize(ME,(2*Ns,))

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		for i in range(Ns):
			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= g_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= g_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

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
						ME[j] *= g_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef long double G_MatrixElement_P_Z(int L,int pblock, int zblock, int kblock, int a, int l, long double k, int q, int g,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
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
		nr = (1.0 + pblock*sr*cos(k*mmr))/Nr
	elif cr == 3:
		nr = (1.0 + zblock*cos(k*nnr))/Nr
	elif cr == 4:
		nr = (1.0 + pblock*zblock*sr*cos(k*mmr))/Nr	
	elif cr == 5:
		nr = (1.0 + pblock*sr*cos(k*mmr))*(1.0 + zblock*cos(k*nnr))/Nr

	nr *= sr

	if cc == 1:
		nc = 1.0/Nc
	elif cc == 2:
		nc = (1.0 + pblock*sc*cos(k*mmc))/Nc
	elif cc == 3:
		nc = (1.0 + zblock*cos(k*nnc))/Nc
	elif cc == 4:
		nc = (1.0 + pblock*zblock*sc*cos(k*mmc))/Nc	
	elif cc == 5:
		nc = (1.0 + pblock*sc*cos(k*mmc))*(1.0 + zblock*cos(k*nnc))/Nc

	nc *= sc



	ME=sqrt(nc/nr)*((sr*pblock)**q)*(zblock**g)

	if sr == sc :
		if (cc == 1) or (cc == 3):
			ME *= cos(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (cos(k*l)+sr*pblock*cos((l-mmc)*k))/(1+sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (cos(k*l)+sr*pblock*zblock*cos((l-mmc)*k))/(1+sr*pblock*zblock*cos(k*mmc))
	else:
		if (cc == 1) or (cc == 3):
			ME *= -sr*sin(k*l)
		elif (cc == 2) or (cc == 5):
			ME *= (-sr*sin(k*l) + pblock*sin((l-mmc)*k))/(1-sr*pblock*cos(k*mmc))
		elif (cc == 4):
			ME *= (-sr*sin(k*l) + pblock*zblock*sin((l-mmc)*k))/(1-sr*pblock*zblock*cos(k*mmc))

	return ME






def G_t_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int pblock,int zblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double complex,ndim=1] ME

	cdef double n
	cdef long double k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = G_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	row = _np.resize(row,(2*Ns,))
	row[Ns:] = -1
	ME = _np.resize(ME,(2*Ns,))

	if ((2*kblock*a) % L) == 0: #picks up k = 0, pi modes
		for i in range(Ns):
			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= G_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P_Z(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]
			g = R[3]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (g == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= G_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[j],N[j],m[j],m[j])

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
						ME[j] *= G_MatrixElement_P_Z(L,pblock,zblock,kblock,a,l,k,q,g,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error




