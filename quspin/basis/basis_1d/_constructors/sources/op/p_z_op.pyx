

def f_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int pblock, int zblock):
	cdef unsigned int s
	cdef int Ns,error,ss,i,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float,ndim=1] ME
	cdef float n


	Ns = basis.shape[0]
	row,ME,error = f_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_P_Z(s,L,R)

		s = R[0]
		q = R[1]
		g = R[2]
		
		ss = findzstate(basis,Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[ss]
		n /= N[i]

		ME[i] *= sqrt(n)*(pblock**q)*(zblock**g)

	return row,ME,error









def d_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int pblock, int zblock):
	cdef unsigned int s
	cdef int Ns,error,ss,i,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double,ndim=1] ME
	cdef double n


	Ns = basis.shape[0]
	row,ME,error = d_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_P_Z(s,L,R)

		s = R[0]
		q = R[1]
		g = R[2]
		
		ss = findzstate(basis,Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[ss]
		n /= N[i]

		ME[i] *= sqrt(n)*(pblock**q)*(zblock**g)

	return row,ME,error









def F_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int pblock, int zblock):
	cdef unsigned int s
	cdef int Ns,error,ss,i,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float complex,ndim=1] ME
	cdef float n


	Ns = basis.shape[0]
	row,ME,error = F_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_P_Z(s,L,R)

		s = R[0]
		q = R[1]
		g = R[2]
		
		ss = findzstate(basis,Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[ss]
		n /= N[i]

		ME[i] *= sqrt(n)*(pblock**q)*(zblock**g)

	return row,ME,error









def D_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int pblock, int zblock):
	cdef unsigned int s
	cdef int Ns,error,ss,i,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double complex,ndim=1] ME
	cdef double n


	Ns = basis.shape[0]
	row,ME,error = D_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_P_Z(s,L,R)

		s = R[0]
		q = R[1]
		g = R[2]
		
		ss = findzstate(basis,Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[ss]
		n /= N[i]

		ME[i] *= sqrt(n)*(pblock**q)*(zblock**g)

	return row,ME,error









def g_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int pblock, int zblock):
	cdef unsigned int s
	cdef int Ns,error,ss,i,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double,ndim=1] ME
	cdef long double n


	Ns = basis.shape[0]
	row,ME,error = g_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_P_Z(s,L,R)

		s = R[0]
		q = R[1]
		g = R[2]
		
		ss = findzstate(basis,Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[ss]
		n /= N[i]

		ME[i] *= sqrt(n)*(pblock**q)*(zblock**g)

	return row,ME,error









def G_p_z_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, long double complex J, int L, int pblock, int zblock):
	cdef unsigned int s
	cdef int Ns,error,ss,i,q,g
	cdef _np.ndarray[NP_UINT32_t,ndim = 1] R = _np.zeros(3,NP_UINT32)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double complex,ndim=1] ME
	cdef long double n


	Ns = basis.shape[0]
	row,ME,error = G_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_P_Z(s,L,R)

		s = R[0]
		q = R[1]
		g = R[2]
		
		ss = findzstate(basis,Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[ss]
		n /= N[i]

		ME[i] *= sqrt(n)*(pblock**q)*(zblock**g)

	return row,ME,error







