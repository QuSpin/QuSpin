
def f_m_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,float complex J):

	cdef int i,Ns 
	cdef unsigned int s
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float,ndim=1] ME

	Ns = basis.shape[0]
	row,ME,error = f_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(basis,Ns,s)


		
	return row,ME,error







def d_m_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,double complex J):

	cdef int i,Ns 
	cdef unsigned int s
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double,ndim=1] ME

	Ns = basis.shape[0]
	row,ME,error = d_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(basis,Ns,s)


		
	return row,ME,error







def F_m_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,float complex J):

	cdef int i,Ns 
	cdef unsigned int s
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[float complex,ndim=1] ME

	Ns = basis.shape[0]
	row,ME,error = F_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(basis,Ns,s)


		
	return row,ME,error







def D_m_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,double complex J):

	cdef int i,Ns 
	cdef unsigned int s
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[double complex,ndim=1] ME

	Ns = basis.shape[0]
	row,ME,error = D_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(basis,Ns,s)


		
	return row,ME,error







def g_m_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,long double complex J):

	cdef int i,Ns 
	cdef unsigned int s
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double,ndim=1] ME

	Ns = basis.shape[0]
	row,ME,error = g_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(basis,Ns,s)


		
	return row,ME,error







def G_m_op(_np.ndarray[NP_UINT32_t,ndim=1] basis, str opstr, _np.ndarray[NP_INT32_t,ndim=1] indx,long double complex J):

	cdef int i,Ns 
	cdef unsigned int s
	cdef _np.ndarray[NP_INT32_t,ndim=1] row
	cdef _np.ndarray[long double complex,ndim=1] ME

	Ns = basis.shape[0]
	row,ME,error = G_spinop(basis,opstr,indx,J)

	if error != 0:
		return row,ME,error

	for i in range(Ns):
		s = row[i]
		row[i] = findzstate(basis,Ns,s)


		
	return row,ME,error






