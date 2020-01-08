from ._basis_general_core import nlce_site_core_wrap,nlce_plaquet_core_wrap
from ._perm_checks import process_map
import numpy as _np
from numba import njit
from builtins import range

@njit
def _get_Nc(nmax,N):
	for i in range(N.size):
		if nmax > N[i]:
			return i
	return N.size


@njit
def _get_W(O,W,data,indices,indptr):
	nrow = O.shape[0]
	nvec = O.shape[1]
	w = O[0,:].copy()
	for i in range(nrow):
		w[:] = O[i,:]
		for k in range(indptr[i],indptr[i+1],1):
			j = indices[k]
			for l in range(nvec):
				w[l] += data[k]*W[j,l]

		W[i,:] = w

@njit
def _get_Sn(Ncl,W,L,N):
	
	index_shift = N[0]

	Nsum = W.shape[0]
	Nobs = W.shape[1]

	Sn = W[:Ncl,:].copy()
	Sn[:,:] = 0
	for i in range(Nsum):
		n = N[i] - index_shift
		l = L[i]
		for j in range(Nobs):
			Sn[n,j] += W[i,j]*l

	return Sn


def wynn_eps_method(p,ncycle):
	nmax = p.shape[0]

	if 2*ncycle >= nmax:
		raise ValueError

	e0 = _np.zeros_like(p)
	e1 = p.copy()
	e2 = _np.zeros_like(e1)

	for k in range(1,2*ncycle+1,1):
		de = _np.diff(e1,axis=0)[0:nmax-k,...]
		e2[0:nmax-k,...] = e0[1:nmax-k+1,...] + de/(de**2+1e-300)

		e0[:] = e1[:]
		e1[:] = e2[:]
		e2[:] = 0

	return e1[:nmax-2*ncycle,...]

class _nlce(object):
	def __init__(self,N_cl,
				 cluster_list,L_list,Ncl_list,Y):
		self._N_cl = N_cl
		self._cluster_list = cluster_list
		self._L_list = L_list
		self._Ncl_list = Ncl_list
		self._Y = Y


	@property
	def Nc(self):
		return self._L_list.shape[0]


	def get_Nc(self,Ncl):
		return _get_Nc(Ncl,self._Ncl_list)

	def get_W(self,O,out=None,Ncl_max=None):
		if Ncl_max is not None:
			if Ncl_max > self._N_cl:
				raise ValueError

			Nc = self.get_Nc(Ncl_max)

		else:
			Nc = self._L_list.shape[0]

		result_dtype = _np.result_type(self._Y.dtype,O.dtype)

		if O.shape[0] != Nc:
			raise ValueError

		shape0 = O.shape
		shape = shape0[:1] + (-1,)

		if out is not None:
			if out.dtype != result_dtype:
				raise ValueError

			if out.shape != shape0:
				raise ValueError
		else:
			out = _np.zeros(shape0,dtype=result_dtype)

		O = O.reshape(shape)
		out = out.reshape(shape)

		_get_W(O,out,self._Y.data,self._Y.indices,self._Y.indptr)

		return out.reshape(shape0)

	def partial_sums(self,O,Ncl_max=None):
		W = self.get_W(O,Ncl_max=Ncl_max)
		shape = W.shape[:1]+(-1,)
		Nc = W.shape[0]
		Sn = _get_Sn(self._N_cl,W.reshape(shape),self._L_list[:Nc],self._Ncl_list[:Nc])
		return Sn

	def bare_sums(self,O,Ncl_max=None):
		return self.partial_sums(O,Ncl_max=Ncl_max).cumsum(axis=0)

	def wynn_sums(self,O,ncycle,Ncl_max=None):
		p = self.bare_sums(O,Ncl_max=Ncl_max)
		return wynn_eps_method(p,ncycle)

	def __getitem__(self,key=None):
		if type(key) is int:
			yield self.get_cluster_graph(key)
		elif type(key) is slice:
			if key.start is None:
				start = 0
			else:
				start = (key.start)%len(self._L_list)

			if key.stop is None:
				stop = len(self._L_list)
			else:
				stop = (key.stop)%len(self._L_list)

			if key.step is None:
				step = 1
			else:
				step = key.step

			for i in range(start,stop,step):
				yield self.get_cluster_graph(i)
		else:
			try:
				iter_key = iter(key)
			except:
				raise ValueError("cannot interpret input: {}".format(key))

			for i in iter_key:
				yield self.get_cluster_graph(i)

	def get_cluster_graph(self,ic):
		raise NotImplementedError


class _ncle_site(_nlce):
	def __init__(self,N_cl,N_lat,
				 nn_list,nn_weight,cluster_list,
				 L_list,Ncl_list,Y):
		
		self._N_lat = N_lat
		self._nn_list = nn_list
		self._nn_weight = nn_weight
		_nlce.__init__(self,N_cl,cluster_list,L_list,Ncl_list,Y)


	def get_cluster_graph(self,ic):
		if type(ic) is not int:
			raise ValueError

		if ic < 0 or ic >= self.Nc:
			raise ValueError

		graph = []
		stack = []
		
		sites = self._cluster_list[ic,:].compressed()
		sites.sort()
		visited = set([])
		stack.append(sites[0])
		if self._nn_weight is not None:
			while(stack):
				i = stack.pop()
				a = _np.searchsorted(sites,i)

				for j,nn in enumerate(self._nn_list[i,:]):
					if nn not in visited and nn in sites:
						b = _np.searchsorted(sites,nn)
						graph.append((self._nn_weight[j],a,b))
						stack.append(nn)

				visited.add(i)
		else:
			while(stack):
				i = stack.pop()
				a = _np.searchsorted(sites,i)

				for nn in self._nn_list[i,:]:
					if nn not in visited and nn in sites:
						b = _np.searchsorted(sites,nn)
						graph.append((a,b))
						stack.append(nn)

				visited.add(i)			

		return ic,_np.array(sites),self._Ncl_list[ic],frozenset(graph)

class NLCE_site(_ncle_site):
	def __init__(self,N_cl,N_lat,nn_list,tr,pg,nn_weight=None):
	
		if nn_list.shape[0] != N_lat:
			raise ValueError

		if tr.shape[1] != N_lat:
			raise ValueError

		if pg.shape[1] != N_lat:
			raise ValueError

		if nn_weight is not None and nn_weight.shape[0] != nn_list.shape[1]:
			raise ValueError

		nt_point = pg.shape[0]
		nt_trans = tr.shape[0]

		symm_list = ([process_map(p,0) for p in pg[:]]+
					 [process_map(p,0) for p in tr[:]] )

		maps,pers,qs,_ = zip(*symm_list)

		maps = _np.vstack(maps).astype(_np.int32)
		pers = _np.array(pers,dtype=_np.int32)
		qs   = _np.array(qs,dtype=_np.int32)

		n_maps = maps.shape[0]

		for j in range(n_maps-1):
			for i in range(j+1,n_maps,1):
				if _np.all(maps[j]==maps[i]):
					ValueError("repeated transformations in list of permutations for point group/translations.")

		nlce_core = nlce_site_core_wrap(N_cl,nt_point,nt_trans,maps,pers,qs,nn_list,nn_weight)

		clusters_list,L_list,Ncl_list,Y = nlce_core.calc_clusters()

		_ncle_site.__init__(self,N_cl,N_lat,nn_list,nn_weight,clusters_list,L_list,Ncl_list,Y)
		
class _ncle_plaquet(_nlce):
	def __init__(self,N_cl,N_plaquet,
				 plaquet_sites,plaquet_edges,edge_weights,
				 cluster_list,L_list,Ncl_list,Y):

		self._N_plaquet = N_plaquet
		self._plaquet_sites = plaquet_sites
		self._plaquet_edges = plaquet_edges
		self._edge_weights = edge_weights

		_nlce.__init__(self,N_cl,cluster_list,L_list,Ncl_list,Y)

	def get_cluster_graph(self,ic):
		if type(ic) is not int:
			raise ValueError

		if ic < 0 or ic >= self.Nc:
			raise ValueError

		graph = []
		stack = []

		plaquets = list(self._cluster_list[ic,:].compressed())

		if ic==0:
			sites = set([0])
		else:
			sites = set([])
			
		for plaquet in plaquets:
			for site in self._plaquet_sites[plaquet]:
				sites.add(site)

		sites = _np.array(list(sites))
		sites.sort()

		visited = set([])

		try:
			stack.append(plaquets[0])
		except IndexError:
			pass

		if self._edge_weights is not None:
			while(stack):
				pos = stack.pop()

				for new_pos,edge_set in self._plaquet_edges[pos].items():
					if new_pos not in visited and new_pos in plaquets:
						for i,j in edge_set:
							ww = self._edge_weights[i][j]
							ii = _np.searchsorted(sites,i)
							jj = _np.searchsorted(sites,j)
							graph.append((ww,ii,jj))

						stack.append(new_pos)

				visited.add(pos)

		else:
			while(stack):
				pos = stack.pop()

				for new_pos,edge_set in self._plaquet_edges[pos].items():
					if new_pos not in visited and new_pos in plaquets:
						for i,j in edge_set:
							ii = _np.searchsorted(sites,i)
							jj = _np.searchsorted(sites,j)
							graph.append((ii,jj))

						stack.append(new_pos)

				visited.add(pos)		

		return ic,sites,self._Ncl_list[ic],frozenset(graph)




class NLCE_plaquet(_ncle_plaquet):
	def __init__(self,N_cl,plaquet_sites,plaquet_edges,tr,pg,edge_weights=None):
		
		plaquet_sites = _np.asanyarray(plaquet_sites)
		plaquet_sites = plaquet_sites.astype(_np.int32,order="C",copy=True)

		tr = _np.asanyarray(tr)
		pg = _np.asanyarray(pg)

		N_plaquet = len(plaquet_edges)

		if not isinstance(plaquet_edges,dict):
			raise TypeError
		else:
			for a,edge_dict in plaquet_edges.items():
				if not isinstance(edge_dict,dict):
					raise TypeError

				for b,edge_set in edge_dict.items():
					if type(edge_set) not in [set,frozenset]:
						raise TypeError


		if plaquet_sites.shape[0] != N_plaquet:
			raise ValueError

		if tr.shape[1] != N_plaquet:
			raise ValueError

		if pg.shape[1] != N_plaquet:
			raise ValueError

		nt_point = pg.shape[0]
		nt_trans = tr.shape[0]

		symm_list = ([process_map(p,0) for p in pg[:]]+
					 [process_map(p,0) for p in tr[:]] )

		maps,pers,qs,_ = zip(*symm_list)

		maps = _np.vstack(maps).astype(_np.int32)
		pers = _np.array(pers,dtype=_np.int32)
		qs   = _np.array(qs,dtype=_np.int32)

		n_maps = maps.shape[0]

		for j in range(n_maps-1):
			for i in range(j+1,n_maps,1):
				if _np.all(maps[j]==maps[i]):
					ValueError("repeated transformations in list of permutations for point group/translations.")

		nlce_core = nlce_plaquet_core_wrap(N_cl,nt_point,nt_trans,maps,pers,qs,
			plaquet_sites,plaquet_edges,edge_weights)

		cluster_list,L_list,Ncl_list,Y = nlce_core.calc_clusters()

		_ncle_plaquet.__init__(self,N_cl,N_plaquet,
				 plaquet_sites,plaquet_edges,edge_weights,
				 cluster_list,L_list,Ncl_list,Y)
		


