


def f_t_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type] basis, str opstr, _np.ndarray[NP_INT32_t] indx, float complex J, int L, int kblock, int a):
	cdef state_type s
	cdef int error,ss,i,Ns,l
	cdef _np.ndarray[basis_type,ndim=1] R = _np.zeros(2,dtype=basis.dtype)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float,ndim=1] ME
	cdef float n,k

	k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = f_spinop(basis,opstr,indx,J)

	if True and ((2*a*kblock) % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error


	for i in range(Ns):
		s = row[i]
		RefState_T_template(shift,s,L,a,&R[0])

		s = R[0]
		l = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[i]
		n /= N[ss]
		n = sqrt(n)
		ME[i] *= n

		if True:
			ME[i] *= (-1.0)**(l*2*a*kblock/L)
		else:
			ME[i] *= (cos(k*l) - 1.0j * sin(k*l))

			

	return row,ME,error







def d_t_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type] basis, str opstr, _np.ndarray[NP_INT32_t] indx, double complex J, int L, int kblock, int a):
	cdef state_type s
	cdef int error,ss,i,Ns,l
	cdef _np.ndarray[basis_type,ndim=1] R = _np.zeros(2,dtype=basis.dtype)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double,ndim=1] ME
	cdef double n,k

	k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = d_spinop(basis,opstr,indx,J)

	if True and ((2*a*kblock) % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error


	for i in range(Ns):
		s = row[i]
		RefState_T_template(shift,s,L,a,&R[0])

		s = R[0]
		l = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[i]
		n /= N[ss]
		n = sqrt(n)
		ME[i] *= n

		if True:
			ME[i] *= (-1.0)**(l*2*a*kblock/L)
		else:
			ME[i] *= (cos(k*l) - 1.0j * sin(k*l))

			

	return row,ME,error







def F_t_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type] basis, str opstr, _np.ndarray[NP_INT32_t] indx, float complex J, int L, int kblock, int a):
	cdef state_type s
	cdef int error,ss,i,Ns,l
	cdef _np.ndarray[basis_type,ndim=1] R = _np.zeros(2,dtype=basis.dtype)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float complex,ndim=1] ME
	cdef float n,k

	k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = F_spinop(basis,opstr,indx,J)

	if False and ((2*a*kblock) % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error


	for i in range(Ns):
		s = row[i]
		RefState_T_template(shift,s,L,a,&R[0])

		s = R[0]
		l = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[i]
		n /= N[ss]
		n = sqrt(n)
		ME[i] *= n

		if False:
			ME[i] *= (-1.0)**(l*2*a*kblock/L)
		else:
			ME[i] *= (cos(k*l) - 1.0j * sin(k*l))

			

	return row,ME,error







def D_t_op(_np.ndarray[NP_INT8_t,ndim=1] N, _np.ndarray[basis_type] basis, str opstr, _np.ndarray[NP_INT32_t] indx, double complex J, int L, int kblock, int a):
	cdef state_type s
	cdef int error,ss,i,Ns,l
	cdef _np.ndarray[basis_type,ndim=1] R = _np.zeros(2,dtype=basis.dtype)
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double complex,ndim=1] ME
	cdef double n,k

	k = (2.0*_np.pi*kblock*a)/L

	Ns = basis.shape[0]

	row,ME,error = D_spinop(basis,opstr,indx,J)

	if False and ((2*a*kblock) % L) != 0:
		error = -1

	if error != 0:
		return row,ME,error


	for i in range(Ns):
		s = row[i]
		RefState_T_template(shift,s,L,a,&R[0])

		s = R[0]
		l = R[1]

		ss = findzstate(&basis[0],Ns,s)
		row[i] = ss

		if ss == -1: continue

		n = N[i]
		n /= N[ss]
		n = sqrt(n)
		ME[i] *= n

		if False:
			ME[i] *= (-1.0)**(l*2*a*kblock/L)
		else:
			ME[i] *= (cos(k*l) - 1.0j * sin(k*l))

			

	return row,ME,error




