


cdef float f_MatrixElement_PZ(int L,int pzblock, int kblock, int a, int l, float k, int qg,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
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
		nr = (1 + sr*pzblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pzblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc

	ME=sqrt(nc/nr)*(sr*pzblock)**qg

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pzblock*cos((l-mc)*k))/(1+sr*pzblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pzblock*sin((l-mc)*k))/(1-sr*pzblock*cos(k*mc))		


	return ME






def f_t_pz_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int pzblock, int kblock, int a):
	cdef state_type s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,qg
	cdef _np.ndarray[basis_type,ndim=1] R = _np.zeros(3,dtype=basis.dtype)
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
			RefState_T_PZ_template(shift,fliplr,flip_all,s,L,a,&R[0])

			s = R[0]
			l = R[1]
			qg = R[2]

			t = s
			ss = findzstate(&basis[0],Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= f_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_PZ_template(shift,fliplr,flip_all,s,L,a,&R[0])

			s = R[0]
			l = R[1]
			qg = R[2]

			ss = findzstate(&basis[0],Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (qg == 0) and (l == 0): #diagonal ME

				for j in range(i,i+o,1):
					row[j] = j
					ME[j] *= f_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[j],N[j],m[j],m[j])

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
						ME[j] *= f_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef double d_MatrixElement_PZ(int L,int pzblock, int kblock, int a, int l, double k, int qg,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
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
		nr = (1 + sr*pzblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pzblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc

	ME=sqrt(nc/nr)*(sr*pzblock)**qg

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pzblock*cos((l-mc)*k))/(1+sr*pzblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pzblock*sin((l-mc)*k))/(1-sr*pzblock*cos(k*mc))		


	return ME






def d_t_pz_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int pzblock, int kblock, int a):
	cdef state_type s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,qg
	cdef _np.ndarray[basis_type,ndim=1] R = _np.zeros(3,dtype=basis.dtype)
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
			RefState_T_PZ_template(shift,fliplr,flip_all,s,L,a,&R[0])

			s = R[0]
			l = R[1]
			qg = R[2]

			t = s
			ss = findzstate(&basis[0],Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= d_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_PZ_template(shift,fliplr,flip_all,s,L,a,&R[0])

			s = R[0]
			l = R[1]
			qg = R[2]

			ss = findzstate(&basis[0],Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (qg == 0) and (l == 0): #diagonal ME

				for j in range(i,i+o,1):
					row[j] = j
					ME[j] *= d_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[j],N[j],m[j],m[j])

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
						ME[j] *= d_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef float F_MatrixElement_PZ(int L,int pzblock, int kblock, int a, int l, float k, int qg,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
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
		nr = (1 + sr*pzblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pzblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc

	ME=sqrt(nc/nr)*(sr*pzblock)**qg

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pzblock*cos((l-mc)*k))/(1+sr*pzblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pzblock*sin((l-mc)*k))/(1-sr*pzblock*cos(k*mc))		


	return ME






def F_t_pz_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int pzblock, int kblock, int a):
	cdef state_type s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,qg
	cdef _np.ndarray[basis_type,ndim=1] R = _np.zeros(3,dtype=basis.dtype)
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
			RefState_T_PZ_template(shift,fliplr,flip_all,s,L,a,&R[0])

			s = R[0]
			l = R[1]
			qg = R[2]

			t = s
			ss = findzstate(&basis[0],Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= F_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_PZ_template(shift,fliplr,flip_all,s,L,a,&R[0])

			s = R[0]
			l = R[1]
			qg = R[2]

			ss = findzstate(&basis[0],Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (qg == 0) and (l == 0): #diagonal ME

				for j in range(i,i+o,1):
					row[j] = j
					ME[j] *= F_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[j],N[j],m[j],m[j])

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
						ME[j] *= F_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error







cdef double D_MatrixElement_PZ(int L,int pzblock, int kblock, int a, int l, double k, int qg,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT8_t mr,NP_INT8_t mc):
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
		nr = (1 + sr*pzblock*cos(k*mr))/Nr
	else:
		nr = 1.0/Nr
	nr *= sr

	if mc >= 0:
		nc = (1 + sc*pzblock*cos(k*mc))/Nc
	else:
		nc = 1.0/Nc
	nc *= sc

	ME=sqrt(nc/nr)*(sr*pzblock)**qg

	if sr == sc :
		if mc < 0:
			ME *= cos(k*l)
		else:
			ME *= (cos(k*l)+sr*pzblock*cos((l-mc)*k))/(1+sr*pzblock*cos(k*mc))
	else:
		if mc < 0:
			ME *= -sr*sin(k*l)
		else:
			ME *= (-sr*sin(k*l)+pzblock*sin((l-mc)*k))/(1-sr*pzblock*cos(k*mc))		


	return ME






def D_t_pz_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT8_t,ndim=1] m, _np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int pzblock, int kblock, int a):
	cdef state_type s
	cdef int error,ss,i,j,Ns,o,p,c,b,l,qg
	cdef _np.ndarray[basis_type,ndim=1] R = _np.zeros(3,dtype=basis.dtype)
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
			RefState_T_PZ_template(shift,fliplr,flip_all,s,L,a,&R[0])

			s = R[0]
			l = R[1]
			qg = R[2]

			t = s
			ss = findzstate(&basis[0],Ns,s)
			row[i] = ss

			if ss == -1:
				continue



			ME[i] *= D_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[i],N[ss],m[i],m[ss])


	else:
		for i in range(Ns):
			if (i > 0) and (basis[i] == basis[i-1]): continue
			if (i < (Ns - 1)) and (basis[i] == basis[i+1]):
				o = 2
			else:
				o = 1

			s = row[i]
			RefState_T_PZ_template(shift,fliplr,flip_all,s,L,a,&R[0])

			s = R[0]
			l = R[1]
			qg = R[2]

			ss = findzstate(&basis[0],Ns,s)

			if ss == -1:
				for j in range(i,i+o,1):
					row[j] = -1
				continue


			if (ss == i) and (qg == 0) and (l == 0): #diagonal ME

				for j in range(i,i+o,1):
					row[j] = j
					ME[j] *= D_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[j],N[j],m[j],m[j])

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
						ME[j] *= D_MatrixElement_PZ(L,pzblock,kblock,a,l,k,qg,N[i+c],N[ss+b],m[i+c],m[ss+b])
		
		
	return row,ME,error




