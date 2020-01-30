from ._basis_general_core import nlce_site_core_wrap,nlce_plaquet_core_wrap
from ._perm_checks import process_map
import numpy as _np
from numba import njit
from builtins import range


@njit
def _get_Nc(nmax,N):
	# helper function to get index of the last cluster of size nmax
	for i in range(N.size):
		if nmax < N[i]:
			return i
	return N.size


@njit
def _get_W(O,W,data,indices,indptr):
	# helper function to get the weights for the bare sum 
	# given the expecation values for the clusters.
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
	# helper function to get the bare partial sums
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
	"""Perform the wynn extrapoation method on a series.

	This function performs the wynn epsilon method to a given series.

	Parameters
	----------
	p : array_like, (n,...)
		input of one or more series to extrapolate

	ncycle : integer, 
		number of cycles of extrapolation to perform, must have: 2*ncycle < n

	returns
	-------

	array_like, (n-2*ncycle,...)
		the extrapolated series to the requested cycle.

	"""
	p = _np.asanyarray(p)

	nmax = p.shape[0]

	if 2*ncycle >= nmax:
		raise ValueError("the number of cycles must satisfy: 2*ncycle < p.shape[0].")

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
	def Ncl_max(self):
		"""Maximum cluster size calculated."""
		return self._N_cl
	

	@property
	def Nc_max(self):
		"""Total number of clusters for this particular object."""
		return self._L_list.shape[0]

	@property
	def L_list(self):
		view =  self._L_list[:]
		view.flags["WRITEABLE"]=False
		return view

	@property
	def Ncl_list(self):
		view =  self._Ncl_list[:]
		view.flags["WRITEABLE"]=False
		return view

	def get_Nc(self,Ncl=None):
		"""Get the total number of cluster for cluster sizes up to requested cluster size. 
		
		Get the number of clusters up to cluster size `Ncl`.

		Parameters
		----------
		Ncl : optional, integer
			maximum cluster size. 

		returns
		-------

		integer :
			total number of clusters up to cluster size `Ncl`.

		"""

		if Ncl is not None:
			if Ncl > self.Ncl_max:
				raise ValueError("'Ncl' must be smaller or equal to the cluster size calculated for the NLCE object.")
			elif Ncl == self.Ncl_max:
				return self.Nc_max
			else:
				return _get_Nc(Ncl,self._Ncl_list)
		else:
			return self.Nc_max

	def get_W(self,O,Ncl_max=None,out=None):
		""" Calculate Weights for expectation values over clusters. 

		Calculate the weights for a given set of expectation values over clusters. The weights are calculated by removing contributions from subclusters. 

		Parameters
		----------

		O : array_like, (M,...)
			expectation values over clusters.

		Ncl_max : optional, integer
			Maximum cluster size, can be smaller than the maximum custer size of the calling object.

		out : optional, array_like, (M,...)
			output array for the results of this function.

		returns
		-------

		array_like, (M,...)
			The weights the expecation values `O`.

		"""

		O = _np.asanyarray(O)

		Nc = self.get_Nc(Ncl_max)

		result_dtype = _np.result_type(self._Y.dtype,O.dtype)

		if O.shape[0] != Nc:
			raise ValueError("'O' must be array with shape (M,...) with M equal to the number of clusters for given 'Ncl_max'")

		shape0 = O.shape
		shape = shape0[:1] + (-1,)

		if out is not None:
			if not isinstance(out,_np.ndarray):
				raise TypeError("'out' must be a numpy ndarray.")

			if out.dtype != result_dtype:
				raise ValueError("'out' must have dtype: {}".format(result_dtype))

			if out.shape != shape0:
				raise ValueError("'out' must have shape: {}".format(shape0))

		else:
			out = _np.zeros(shape0,dtype=result_dtype)

		O = O.reshape(shape)
		out = out.reshape(shape)

		_get_W(O,out,self._Y.data,self._Y.indices,self._Y.indptr)

		return out.reshape(shape0)

	def cluster_sums(self,O,Ncl_max=None):
		"""Calculate sums over cluster of a given size given the expecation values over clusters. 

		Parameters
		----------

		O : array_like, (M,...)
			expectation values over clusters.

		Ncl_max : optional, integer
			Maximum cluster size, can be smaller than the maximum custer size of the calling object. 
			Default value is the maximum cluster size of the calling object. 

		returns
		-------

		array_like, (Ncl_max,...)
			The cluster sums for expectation values of `O`.

		"""

		W = self.get_W(O,Ncl_max=Ncl_max)
		shape = W.shape[:1]+(-1,)
		Nc = W.shape[0]
		Sn = _get_Sn(self._N_cl,W.reshape(shape),self._L_list[:Nc],self._Ncl_list[:Nc])
		return Sn

	def partial_sums(self,O,Ncl_max=None):
		"""Calculate the bare sums given the expecation values over clusters. 

		equivilant to calculating cumulative sums over the cluster sums. 

		Parameters
		----------

		O : array_like, (M,...)
			expectation values over clusters.

		Ncl_max : optional, integer
			Maximum cluster size, can be smaller than the maximum custer size of the calling object. 
			Default value is the maximum cluster size of the calling object. 

		returns
		-------

		array_like, (Ncl_max,...)
			The partial sums for expectation values of `O`.

		"""
		return self.cluster_sums(O,Ncl_max=Ncl_max).cumsum(axis=0)

	def wynn_sums(self,O,ncycle,Ncl_max=None):
		"""Calculate the bare sums and perform wynn extrapolation. 

		calculates the partial sums and performs wynn extrapolation. 

		Parameters
		----------

		O : array_like, (M,...)
			expectation values over clusters.
		
		ncycle : integer, 
			number of cycles of extrapolation to perform, must have: 2*ncycle < Ncl_max

		Ncl_max : optional, integer
			Maximum cluster size, can be smaller than the maximum custer size of the calling object. 
			Default value is the maximum cluster size of the calling object. 

		returns
		-------

		array_like, (Ncl_max-2*ncycle,...)
			The extrapolated series to the requested cycle of the partial sums for expectation values of `O`.

		"""
		p = self.partial_sums(O,Ncl_max=Ncl_max)
		return wynn_eps_method(p,ncycle)

	def __getitem__(self,key=None):
		if type(key) is int:
			yield self.get_cluster_graph(key)
		elif type(key) is slice:

			if key.start is None:
				start = 0
			elif key.start > self.Nc_max:
				raise IndexError
			elif key.start < -(self.Nc_max+1):
				raise IndexError
			elif key.start < 0:
				start = self.Nc_max + key.start + 1
			else:
				start = key.start

			if key.stop is None:
				stop = len(self._L_list)
			elif key.stop > self.Nc_max:
				raise IndexError
			elif key.stop < -(self.Nc_max+1):
				raise IndexError
			elif key.stop < 0:
				stop = self.Nc_max + key.stop + 1
			else:
				stop = key.stop

			if key.step is None:
				step = 1
			else:
				step = key.step

			print(start,stop,step)

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
		"""Get connectivity list for a give cluster.

		Parameters
		----------

		ic : integer
			index for the requested cluster.

		returns
		-------

		ic : integer
			Same value as input `ic`.

		sites : numpy.ndarray, (Ncl,)
			The sites on the lattice that are included in this cluster.

		graph : frozenset of tuples
			The connectivity of the graph, w/ or w/o weights. The vertices of this graph are mapped to the full lattice
			via `sites`. 

		"""
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

		if ic < 0 or ic >= self.Nc_max:
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

	get_cluster_graph.__doc__ = _nlce.get_cluster_graph.__doc__



class NLCE_site(_ncle_site):
	"""Site based Numerical Linked Cluster Expansions. 

	This class is a specifically implements an optimized calculation of the site based Numerical Linked Cluster Expansion (NLCE).

	This particular type of NCLE calculation is over the infinite lattice. As the expansion has to be truncated at a finite order
	and so the representation of the cluster is over a finite lattice that has to be defined by the user. This user-defined 
	lattice must be periodic and must be large enough such that a cluster will not wrap around the boundaries of the system. As 
	an example we show the cluster expansion over the infinite square lattice with nearest neighbor connections. 


	"""
	def __init__(self,N_cl,N_lat,nn_list,tr,pg,nn_weight=None):
		"""Initialize the `NLCE_site` object.
		
		Parameters
		----------

		N_cl: integer
			Maximum cluster size to calculate to in the expansion.

		N_lat: integer
			Number of sites on the embedding lattice.

		nn_list : array_like, (N_lat,N_nn)
			array containing list of nearest neighbors for the sites. The row index corresponds to the given site

		tr : array_like, (n_trans,N_lat)
			array containing the permutation of the embedding sites that generates all the independent translations. 

		pg : array_like, (n_point,N_lat)
			array containing the permutation of the embedding sites that generates all of the independent point-group
			symmetries of the embedding lattice. 

		nn_weight : array_like, (N_lat,N_nn), optional
			array containing the weights for the bonds connecting nearest neighbors.


		Notes
		-----

		This class does not check if the input graph is consistent with the given translations or 
		point-group symmetries, this has to be handled by the user to ensure these are correct. 


		"""
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

		if ic < 0 or ic >= self.Nc_max:
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

	get_cluster_graph.__doc__ = _nlce.get_cluster_graph.__doc__


class NLCE_plaquet(_ncle_plaquet):
	""" Generic user defined Numerical Linked Cluster Expansion.

	This class can be used to generated a general linked cluster expansions on an infinite lattice with arbitrary building blocks
	generically called plaquets.

	As with the site based expansion, the finite lattice used to embed the clusters must be large enough to avoid wrapping. The 
	expansion is defined through a dictionary that gives the connectivity of the individual plaquets on the finite lattice
	as well as the which sites belong to given plquet(s). The symmetries of the lattice are define through the transformation
	of the plaquets, but should represent the underlying symmetry of the sites. For translational symmetry one should generate 
	translations that tranlate the plaquets, not the sites. As an example we discuss the square plaquet expansion of the 
	square lattice. 


	"""
	def __init__(self,N_cl,plaquet_sites,plaquet_edges,tr,pg,edge_weights=None,plaquet_per_site=None):
		"""Initialize the `NLCE_site` object.
		
		Parameters
		----------

		N_cl: integer
			Maximum cluster size to calculate to in the expansion.

		plaquet_sites: array_like, (N_lat,N_sp)
			Number of sites on the embedding lattice.

		plaquet_edges : dictionary
			dictionary of dictionaries of sets

		tr : array_like, (n_trans,N_lat)
			array containing the permutation of the plaquets over the embedding sites that generates all the independent translations. 

		pg : array_like, (n_point,N_lat)
			array containing the permutation of the plaquets over the embedding sites that generates all of the independent point-group
			symmetries of the embedding lattice. 

		edge_weights : array_like, (N_lat,N_nn), optional
			dictionary of dictionaries that contains the weights of the bonds as indexed through the sites. 

		plaquet_per_site : int, optional
			number that specifies the scaling of the multiplicity of a plaquet cluster. If not specified this number is deduced from inputs. 

		Notes
		-----

		This class does not check if the input graph is consistent with the given translations or 
		point-group symmetries, this has to be handled by the user to ensure these are correct.

		"""

		if plaquet_per_site is None:
			_,counts = _np.unique(plaquet_sites,return_counts=True)

			self._plaquets_per_site = _np.sum(counts)//len(counts)
		else:
			self._plaquets_per_site = plaquet_per_site

		if _np.any(counts != self._plaquets_per_site):
			raise ValueError("Number of plaquets per site is not equal for all plaquets")

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

		plaquet_sites = _np.asanyarray(plaquet_sites)
		plaquet_sites = plaquet_sites.astype(_np.int32,order="C",copy=True)

	def get_W(self,O,Ncl_max=None,out=None):
		res = _nlce.get_W(self,O,Ncl_max=Ncl_max,out=out)
		res[1:] /= self._plaquets_per_site
		return res

	get_W.__doc__ = _nlce.get_W.__doc__
