
def f_zB_op(_np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int zBblock):
	cdef state_type s
	cdef int error,ss,i,Ns,gB
	cdef _np.ndarray[basis_type,ndim = 1] R = _np.zeros(2,dtype=basis.dtype)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float,ndim=1] ME

	Ns = basis.shape[0]

	row,ME,error = f_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZB_template(flip_sublat_B,s,L,&R[0])

		s = R[0]
		gB = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zBblock)**gB


	return row,ME,error







def d_zB_op(_np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int zBblock):
	cdef state_type s
	cdef int error,ss,i,Ns,gB
	cdef _np.ndarray[basis_type,ndim = 1] R = _np.zeros(2,dtype=basis.dtype)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double,ndim=1] ME

	Ns = basis.shape[0]

	row,ME,error = d_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZB_template(flip_sublat_B,s,L,&R[0])

		s = R[0]
		gB = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zBblock)**gB


	return row,ME,error







def F_zB_op(_np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, float complex J, int L, int zBblock):
	cdef state_type s
	cdef int error,ss,i,Ns,gB
	cdef _np.ndarray[basis_type,ndim = 1] R = _np.zeros(2,dtype=basis.dtype)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float complex,ndim=1] ME

	Ns = basis.shape[0]

	row,ME,error = F_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZB_template(flip_sublat_B,s,L,&R[0])

		s = R[0]
		gB = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zBblock)**gB


	return row,ME,error







def D_zB_op(_np.ndarray[basis_type,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx, double complex J, int L, int zBblock):
	cdef state_type s
	cdef int error,ss,i,Ns,gB
	cdef _np.ndarray[basis_type,ndim = 1] R = _np.zeros(2,dtype=basis.dtype)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double complex,ndim=1] ME

	Ns = basis.shape[0]

	row,ME,error = D_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		RefState_ZB_template(flip_sublat_B,s,L,&R[0])

		s = R[0]
		gB = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss
		if ss == -1: continue
		ME[i] *= (zBblock)**gB


	return row,ME,error






