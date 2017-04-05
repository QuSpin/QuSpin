
def f_zA_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int zAblock):
	cdef unsigned int s
	cdef int error,ss,i,Ns,gA
	cdef _np.ndarray[NP_UINT32_t,ndim = 1,mode='c'] R = _np.zeros(2,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float,ndim=1,mode='c'] ME

	Ns = basis.shape[0]

	row,ME,error = f_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZA(s,L,R)

		s = R[0]
		gA = R[1]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zAblock)**gA


	return row,ME,error







def d_zA_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int zAblock):
	cdef unsigned int s
	cdef int error,ss,i,Ns,gA
	cdef _np.ndarray[NP_UINT32_t,ndim = 1,mode='c'] R = _np.zeros(2,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double,ndim=1,mode='c'] ME

	Ns = basis.shape[0]

	row,ME,error = d_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZA(s,L,R)

		s = R[0]
		gA = R[1]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zAblock)**gA


	return row,ME,error







def F_zA_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int zAblock):
	cdef unsigned int s
	cdef int error,ss,i,Ns,gA
	cdef _np.ndarray[NP_UINT32_t,ndim = 1,mode='c'] R = _np.zeros(2,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float complex,ndim=1,mode='c'] ME

	Ns = basis.shape[0]

	row,ME,error = F_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZA(s,L,R)

		s = R[0]
		gA = R[1]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zAblock)**gA


	return row,ME,error







def D_zA_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int zAblock):
	cdef unsigned int s
	cdef int error,ss,i,Ns,gA
	cdef _np.ndarray[NP_UINT32_t,ndim = 1,mode='c'] R = _np.zeros(2,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double complex,ndim=1,mode='c'] ME

	Ns = basis.shape[0]

	row,ME,error = D_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZA(s,L,R)

		s = R[0]
		gA = R[1]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zAblock)**gA


	return row,ME,error







def g_zA_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int zAblock):
	cdef unsigned int s
	cdef int error,ss,i,Ns,gA
	cdef _np.ndarray[NP_UINT32_t,ndim = 1,mode='c'] R = _np.zeros(2,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double,ndim=1,mode='c'] ME

	Ns = basis.shape[0]

	row,ME,error = g_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZA(s,L,R)

		s = R[0]
		gA = R[1]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zAblock)**gA


	return row,ME,error







def G_zA_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int zAblock):
	cdef unsigned int s
	cdef int error,ss,i,Ns,gA
	cdef _np.ndarray[NP_UINT32_t,ndim = 1,mode='c'] R = _np.zeros(2,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double complex,ndim=1,mode='c'] ME

	Ns = basis.shape[0]

	row,ME,error = G_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZA(s,L,R)

		s = R[0]
		gA = R[1]

		ss = findzstate(basis,Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zAblock)**gA


	return row,ME,error






