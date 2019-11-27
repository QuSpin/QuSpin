




@njit
def _get_W(O,W,data,indices,indptr):
	nrow = O.shape[0]
	nvec = O.shape[1]

	for i in range(nrow):
		w = O[i,:].copy()
		for k in range(indptr[i],indptr[i+1],1):
			j = indices[k]
			for l in range(nvec)
				w[l] += data[k]*W[indices[k],l]

		W[i,:] = w


def _get_Sn(W,L,N):
	Ncl = N[-1]

	Nsum = s.shape[0]
	Nobs = s.shape[1]

	Sn = np.zeros((Ncl,Nobs))
	for i in range(Nsum):
		n = N[i]
		for j in range(Nobs):
			Sn[n,j] += W[i,j]*L[i]

	return Sn




class _ncle(object):
	def __init__(self,N_cl,N_lat,
					  nn_list,cluster_list,
					  L_list,Ncl_list,Y):
		self._N_cl = N_cl
		self._N_lat = N_lat
		self._cluster_list = cluster_list
		self._L_list = L_list
		self._n_list = n_list
		self._Y = Y

	def get_W(self,O,out=None):
		result_dtype = _np.result_type(self._Y.dtype,O.dtype)
		

		if O.shape[0] != self._L_lat.shape[0]:
			raise ValueError

		if out is not None:
			if out.dtype != result_dtype:
				raise ValueError

			if out.shape != shape0:
				raise ValueError
		else:
			out = np.zeros(shape0,dtype=result_dtype)

		shape0 = O.shape
		shape = shape0[:1] + (-1,)

		O = O.reshape(shape)
		out = out.reshape(shape)

		_get_W(O,out,self._Y.data,self._Y.indices,self._Y.indptr)

		return out.reshape(shape0)

	def partial_sums(self,O):
		W = self.get_W(O)
		Sn = _get_Sn(W,self._L_list,self._n_list)
		return Sn

	def bare_sums(self,O):
		return self.partial_sums(O).cumsum(axis=0)

	def wynn_sums(self,O,ncycle):
		if 2*ncycle >= O:
			raise ValueError

		p = self.bare_sums(O)

		nmax = p.shape[0]

		e0 = np.zeros_like(p)
		e1 = p.copy()
		e2 = np.zeros_like(e1)

		for k in range(1,2*ncycle+1,1):
			e2[0:nmax-k,...] = e0[1:nmax-k+1,...] + 1/(np.diff(e1,axis=0)[0:nmax-k,...]+1.1e-15)

			e0[:] = e1[:]
			e1[:] = e2[:]
			e2[:] = 0

		return e2[:nmax-2*ncycle,...]

	def get_cluster(self,ic):
		graph = []
		stack = []
		
		sites = list(self._cluster_list[ic,:])
		stack.append(sites[0])

		while(stack):
			i = stack.pop()
			a = sites.index(i)

			for nn in self._nn_list[i,:]:
				if nn in sites:
					b = sites.index(nn)
					graph.append([a,b])
					stack.append(nn)

			sites.remove(i)

		return graph




def get_period(p):
	s = np.arange(p.shape[0])
	i = np.unique(p)

	if not _np.add(i==s):
		raise ValueError

	r = s[p]
	T_p = 1
	while(not _np.all(r==s)):
		r = r[p]
		T_p += 1

	return T_p



class NLCE(_ncle):
	def __init__(self,N_cl,N_lat,nn_list,tr,pg):
		
		nt_point = pg.shape[0]
		nt_trans = tr.shape[0]

		maps = np.vstack((pg,tr))

		pg_list = []

		nlce_core = nlce_core_wrap(N_cl,nt_point,nt_trans)
		















