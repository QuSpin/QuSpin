

cdef float f_MatrixElement_ZA_ZB(int L,int zAblock,int zBblock, int kblock, int a, int l, float k, int gA, int gB,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef float nr,nc
	cdef float ME
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


	if True:
		ME *= (-1)**(2*l*a*kblock/L)
	else:
		ME *= (cos(k*l) - 1.0j * sin(k*l))

	return ME



def f_t_zA_zB_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int zAblock, int zBblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,Ns,l,gA,gB
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float,ndim=1] ME
	cdef float k = (2.0*_np.pi*kblock*a)/L


	Ns = basis.shape[0]
	row,ME,error = f_spinop(basis,opstr,indx,J)

	if True and (2*kblock*a % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_T_ZA_ZB(s,L,a,R)

		s = R[0]
		l = R[1]
		gA = R[2]
		gB = R[3]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue

		ME[i] *= f_MatrixElement_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
		

	return row,ME,error









cdef double d_MatrixElement_ZA_ZB(int L,int zAblock,int zBblock, int kblock, int a, int l, double k, int gA, int gB,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef double nr,nc
	cdef double ME
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


	if True:
		ME *= (-1)**(2*l*a*kblock/L)
	else:
		ME *= (cos(k*l) - 1.0j * sin(k*l))

	return ME



def d_t_zA_zB_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int zAblock, int zBblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,Ns,l,gA,gB
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double,ndim=1] ME
	cdef double k = (2.0*_np.pi*kblock*a)/L


	Ns = basis.shape[0]
	row,ME,error = d_spinop(basis,opstr,indx,J)

	if True and (2*kblock*a % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_T_ZA_ZB(s,L,a,R)

		s = R[0]
		l = R[1]
		gA = R[2]
		gB = R[3]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue

		ME[i] *= d_MatrixElement_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
		

	return row,ME,error









cdef float complex F_MatrixElement_ZA_ZB(int L,int zAblock,int zBblock, int kblock, int a, int l, float k, int gA, int gB,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef float nr,nc
	cdef float complex ME
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


	if False:
		ME *= (-1)**(2*l*a*kblock/L)
	else:
		ME *= (cos(k*l) - 1.0j * sin(k*l))

	return ME



def F_t_zA_zB_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int zAblock, int zBblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,Ns,l,gA,gB
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float complex,ndim=1] ME
	cdef float k = (2.0*_np.pi*kblock*a)/L


	Ns = basis.shape[0]
	row,ME,error = F_spinop(basis,opstr,indx,J)

	if False and (2*kblock*a % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_T_ZA_ZB(s,L,a,R)

		s = R[0]
		l = R[1]
		gA = R[2]
		gB = R[3]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue

		ME[i] *= F_MatrixElement_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
		

	return row,ME,error









cdef double complex D_MatrixElement_ZA_ZB(int L,int zAblock,int zBblock, int kblock, int a, int l, double k, int gA, int gB,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef double nr,nc
	cdef double complex ME
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


	if False:
		ME *= (-1)**(2*l*a*kblock/L)
	else:
		ME *= (cos(k*l) - 1.0j * sin(k*l))

	return ME



def D_t_zA_zB_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int zAblock, int zBblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,Ns,l,gA,gB
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double complex,ndim=1] ME
	cdef double k = (2.0*_np.pi*kblock*a)/L


	Ns = basis.shape[0]
	row,ME,error = D_spinop(basis,opstr,indx,J)

	if False and (2*kblock*a % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_T_ZA_ZB(s,L,a,R)

		s = R[0]
		l = R[1]
		gA = R[2]
		gB = R[3]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue

		ME[i] *= D_MatrixElement_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
		

	return row,ME,error









cdef long double g_MatrixElement_ZA_ZB(int L,int zAblock,int zBblock, int kblock, int a, int l, long double k, int gA, int gB,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
	cdef long double nr,nc
	cdef long double ME
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


	if True:
		ME *= (-1)**(2*l*a*kblock/L)
	else:
		ME *= (cos(k*l) - 1.0j * sin(k*l))

	return ME



def g_t_zA_zB_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int zAblock, int zBblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,Ns,l,gA,gB
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double,ndim=1] ME
	cdef long double k = (2.0*_np.pi*kblock*a)/L


	Ns = basis.shape[0]
	row,ME,error = g_spinop(basis,opstr,indx,J)

	if True and (2*kblock*a % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_T_ZA_ZB(s,L,a,R)

		s = R[0]
		l = R[1]
		gA = R[2]
		gB = R[3]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue

		ME[i] *= g_MatrixElement_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
		

	return row,ME,error









cdef long double complex G_MatrixElement_ZA_ZB(int L,int zAblock,int zBblock, int kblock, int a, int l, long double k, int gA, int gB,NP_INT8_t Nr,NP_INT8_t Nc,NP_INT16_t mr,NP_INT16_t mc):
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


	if False:
		ME *= (-1)**(2*l*a*kblock/L)
	else:
		ME *= (cos(k*l) - 1.0j * sin(k*l))

	return ME



def G_t_zA_zB_op(_np.ndarray[NP_INT8_t,ndim=1] N,_np.ndarray[NP_INT16_t,ndim=1] m, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int zAblock, int zBblock, int kblock, int a):
	cdef unsigned int s
	cdef int error,ss,i,Ns,l,gA,gB
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(4,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double complex,ndim=1] ME
	cdef long double k = (2.0*_np.pi*kblock*a)/L


	Ns = basis.shape[0]
	row,ME,error = G_spinop(basis,opstr,indx,J)

	if False and (2*kblock*a % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_T_ZA_ZB(s,L,a,R)

		s = R[0]
		l = R[1]
		gA = R[2]
		gB = R[3]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue

		ME[i] *= G_MatrixElement_ZA_ZB(L,zAblock,zBblock,kblock,a,l,k,gA,gB,N[i],N[ss],m[i],m[ss])
		

	return row,ME,error







