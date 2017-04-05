


cdef float f_MatrixElement_P(int L,int pblock, int kblock, int a, int l, float k, int q,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
	cdef float nr,nc
	cdef float ME
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
		nr = (1 + sr*pblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc


	ME=sqrt(nc/nr)*(sr*pblock)**q

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pblock*cos((l-mc)*k))/(1+sr*pblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pblock*sin((l-mc)*k))/(1-sr*pblock*cos(k*mc))		


	return ME






def f_t_p_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int pblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float,ndim=1] ME

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
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= f_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= f_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

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
						ME[j] *= f_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef double d_MatrixElement_P(int L,int pblock, int kblock, int a, int l, double k, int q,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
	cdef double nr,nc
	cdef double ME
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
		nr = (1 + sr*pblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc


	ME=sqrt(nc/nr)*(sr*pblock)**q

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pblock*cos((l-mc)*k))/(1+sr*pblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pblock*sin((l-mc)*k))/(1-sr*pblock*cos(k*mc))		


	return ME






def d_t_p_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int pblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double,ndim=1] ME

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
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= d_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= d_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

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
						ME[j] *= d_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef float F_MatrixElement_P(int L,int pblock, int kblock, int a, int l, float k, int q,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
	cdef float nr,nc
	cdef float ME
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
		nr = (1 + sr*pblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc


	ME=sqrt(nc/nr)*(sr*pblock)**q

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pblock*cos((l-mc)*k))/(1+sr*pblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pblock*sin((l-mc)*k))/(1-sr*pblock*cos(k*mc))		


	return ME






def F_t_p_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int pblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float complex,ndim=1] ME

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
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= F_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= F_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

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
						ME[j] *= F_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef double D_MatrixElement_P(int L,int pblock, int kblock, int a, int l, double k, int q,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
	cdef double nr,nc
	cdef double ME
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
		nr = (1 + sr*pblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc


	ME=sqrt(nc/nr)*(sr*pblock)**q

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pblock*cos((l-mc)*k))/(1+sr*pblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pblock*sin((l-mc)*k))/(1-sr*pblock*cos(k*mc))		


	return ME






def D_t_p_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int pblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double complex,ndim=1] ME

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
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= D_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= D_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

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
						ME[j] *= D_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef long double g_MatrixElement_P(int L,int pblock, int kblock, int a, int l, long double k, int q,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
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
		nr = (1 + sr*pblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc


	ME=sqrt(nc/nr)*(sr*pblock)**q

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pblock*cos((l-mc)*k))/(1+sr*pblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pblock*sin((l-mc)*k))/(1-sr*pblock*cos(k*mc))		


	return ME






def g_t_p_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int pblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double,ndim=1] ME

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
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= g_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= g_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

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
						ME[j] *= g_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef long double G_MatrixElement_P(int L,int pblock, int kblock, int a, int l, long double k, int q,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
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
		nr = (1 + sr*pblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc


	ME=sqrt(nc/nr)*(sr*pblock)**q

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pblock*cos((l-mc)*k))/(1+sr*pblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pblock*sin((l-mc)*k))/(1-sr*pblock*cos(k*mc))		


	return ME






def G_t_p_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int pblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,q
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double complex,ndim=1] ME

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
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= G_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_P(s,L,a,R)

			s = R[0]
			l = R[1]
			q = R[2]

			ss = findzstate(basis,Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (q == 0) and (l == 0): #diagonal ME

				for j in xrange(i,i+o,1):
					row[j] = j
					ME[j] *= G_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[j],N[j],m[j],m[j])

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
						ME[j] *= G_MatrixElement_P(L,pblock,kblock,a,l,k,q,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error




