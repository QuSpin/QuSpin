import numpy as _np
import scipy.sparse as _sp
import os
from ..lattice import lattice_basis
import warnings

class GeneralBasisWarning(Warning):
	pass


def process_map(map,q):
	map = _np.asarray(map,dtype=_np.int32)
	i_map = map.copy()
	i_map[map<0] = -(i_map[map<0] + 1) # site mapping
	s_map = map < 0 # sites with spin-inversion

	sites = _np.arange(len(map),dtype=_np.int32)
	order = sites.copy()

	if _np.any(_np.sort(i_map)-order):
		raise ValueError("map must be a one-to-one site mapping.")

	per = 0
	group = [tuple(order)]
	while(True):
		sites[s_map] = -(sites[s_map]+1)
		sites = sites[i_map]
		per += 1
		group.append(tuple(sites))
		if _np.array_equal(order,sites):
			break

	if per == 1:
		warnings.warn("identity mapping found in set of transformations.",GeneralBasisWarning,stacklevel=5)

	return map,per,q,set(group)

def check_symmetry_maps(item1,item2):
	grp1 = item1[1][-1]
	map1 = item1[1][0]
	block1 = item1[0]

	i_map1 = map1.copy()
	i_map1[map1<0] = -(i_map1[map1<0] + 1) # site mapping
	s_map1 = map1 < 0 # sites with spin-inversion		

	grp2 = item2[1][-1]
	map2 = item2[1][0]
	block2 = item2[0]

	i_map2 = map2.copy()
	i_map2[map2<0] = -(i_map2[map2<0] + 1) # site mapping
	s_map2 = map2 < 0 # sites with spin-inversion

	if grp1 == grp2:
		warnings.warn("mappings for block {} and block {} produce the same symmetry.".format(block1,block2),GeneralBasisWarning,stacklevel=5)

	sites1 = _np.arange(len(map1))
	sites2 = _np.arange(len(map2))

	sites1[s_map1] = -(sites1[s_map1]+1)
	sites1 = sites1[i_map1]
	sites1[s_map2] = -(sites1[s_map2]+1)
	sites1 = sites1[i_map2]

	sites2[s_map2] = -(sites2[s_map2]+1)
	sites2 = sites2[i_map2]
	sites2[s_map1] = -(sites2[s_map1]+1)
	sites2 = sites2[i_map1]

	if not _np.array_equal(sites1,sites2):
		warnings.warn("using non-commuting symmetries can lead to unwanted behaviour of general basis, make sure that quantum numbers are invariant under non-commuting symmetries!",GeneralBasisWarning,stacklevel=5)

class basis_general(lattice_basis):
	def __init__(self,N,**kwargs):
		self._unique_me = True
		self._check_pcon = None

		if self.__class__ is basis_general:
			raise TypeError("general_basis class is not to be instantiated.")

		kwargs = {key:value for key,value in kwargs.items() if value is not None}
		
		# if not kwargs:
		# 	raise ValueError("require at least one map.")

		n_maps = len(kwargs)

		if n_maps > 32:
			raise ValueError("general basis can only support up to 32 symmetries.")

		if n_maps>0:
			self._conserved='custom symmeries'
		else:
			self._conserved=''

		if any((type(map) is not tuple) and (len(map)!=2) for map in kwargs.values()):
			raise ValueError("blocks must contain tuple: (map,q).")

		kwargs = {block:process_map(*item) for block,item in kwargs.items()}

		sorted_items = sorted(kwargs.items(),key=lambda x:x[1][1])
		sorted_items.reverse()

		self._blocks = {block:((-1)**q if per==2 else q) for block,(_,per,q,_) in sorted_items}
		self._maps_dict = {block:map for block,(map,_,_,_) in sorted_items}
		remove_index = []
		for i,item1 in enumerate(sorted_items[:-1]):
			if item1[1][1] == 1:
				remove_index.append(i)
			for j,item2 in enumerate(sorted_items[i+1:]):
				check_symmetry_maps(item1,item2)

		remove_index.sort()

		if sorted_items:
			blocks,items = zip(*sorted_items)
			items = list(items)

			for i in remove_index:
				items.pop(i)

			n_maps = len(items)
			maps,pers,qs,_ = zip(*items)

			self._maps = _np.vstack(maps)
			self._qs   = _np.asarray(qs,dtype=_np.int32)
			self._pers = _np.asarray(pers,dtype=_np.int32)

			if any(map.ndim != 1 for map in self._maps[:]):
				raise ValueError("maps must be a 1-dim array/list of integers.")

			if any(map.shape[0] != N for map in self._maps[:]):
				raise ValueError("size of map is not equal to N.")

			if self._maps.shape[0] != self._qs.shape[0]:
				raise ValueError("number of maps must be the same as the number of quantum numbers provided.")

			for j in range(n_maps-1):
				for i in range(j+1,n_maps,1):
					if _np.all(self._maps[j]==self._maps[i]):
						ValueError("repeated map in maps list.")
		else:
			self._maps = _np.array([[]],dtype=_np.int32)
			self._qs   = _np.array([],dtype=_np.int32)
			self._pers = _np.array([],dtype=_np.int32)

		nmax = self._pers.prod()
		self._n_dtype = _np.min_scalar_type(nmax)

	@property
	def description(self):
		"""str: information about `basis` object."""
		blocks = ""

		for symm in self._blocks:
			blocks += symm+" = {"+symm+"}, "

		blocks = blocks.format(**self._blocks)

		if len(self._conserved) == 0:
			symm = "no symmetry"
		elif len(self._conserved) == 1:
			symm = "symmetry"
		else:
			symm = "symmetries"

		string = """general basis for lattice of N = {0} sites containing {5} states \n\t{1}: {2} \n\tquantum numbers: {4} \n\n""".format(self._N,symm,self._conserved,'',blocks,self._Ns)
		string += self.operators
		return string


	def _reduce_n_dtype(self):
		if len(self._n)>0:
			self._n_dtype = _np.min_scalar_type(self._n.max())
			self._n = self._n.astype(self._n_dtype)

	def _Op(self,opstr,indx,J,dtype):

		indx = _np.asarray(indx,dtype=_np.int32)

		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')

		if _np.any(indx >= self._N) or _np.any(indx < 0):
			raise ValueError('values in indx falls outside of system')

		extra_ops = set(opstr) - self._allowed_ops
		if extra_ops:
			raise ValueError("unrecognized characters {} in operator string.".format(extra_ops))

		if self._Ns <= 0:
			return _np.array([],dtype=dtype),_np.array([],dtype=self._index_type),_np.array([],dtype=self._index_type)
	
		col = _np.zeros(self._Ns,dtype=self._index_type)
		row = _np.zeros(self._Ns,dtype=self._index_type)
		ME = _np.zeros(self._Ns,dtype=dtype)

		self._core.op(row,col,ME,opstr,indx,J,self._basis,self._n)

		mask = _np.logical_not(_np.logical_or(_np.isnan(ME),_np.abs(ME)==0.0))
		col = col[mask]
		row = row[mask]
		ME = ME[mask]

		return ME,row,col

	def get_proj(self,dtype):
		"""Calculates transformation/projector from symmetry-reduced basis to full (symmetry-free) basis.

		Notes
		-----
		Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
		in a straightforward manner.

		Parameters
		-----------
		dtype : 'type'
			Data type (e.g. numpy.float64) to construct the projector with.
		sparse : bool, optional
			Whether or not the output should be in sparse format. Default is `True`.
		
		Returns
		--------
		scipy.sparse.csr_matrix
			Transformation/projector between the symmetry-reduced and the full basis.

		Examples
		--------

		>>> P = get_proj(np.float64,pcon=False)
		>>> print(P.shape)

		"""
		c = _np.ones_like(self._basis,dtype=dtype)
		sign = _np.ones_like(self._basis,dtype=_np.int8)
		c[:] = self._n[:]
		c *= self._pers.prod()
		_np.sqrt(c,out=c)
		_np.power(c,-1,out=c)
		index_type = _np.min_scalar_type(-(self._sps**self._N))
		col = _np.arange(self._Ns,dtype=index_type)
		row = _np.arange(self._Ns,dtype=index_type)
		return self._core.get_proj(self._basis,dtype,sign,c,row,col)

	def get_vec(self,v0,sparse=True):
		"""Transforms state from symmetry-reduced basis to full (symmetry-free) basis.

		Notes
		-----
		Particularly useful when a given operation canot be carried away in the symmetry-reduced basis
		in a straightforward manner.

		Supports parallelisation to multiple states listed in the columns.

		Parameters
		-----------
		v0 : numpy.ndarray
			Contains in its columns the states in the symmetry-reduced basis.
		sparse : bool, optional
			Whether or not the output should be in sparse format. Default is `True`.
		
		Returns
		--------
		numpy.ndarray
			Array containing the state `v0` in the full basis.

		Examples
		--------

		>>> v_full = get_vec(v0)
		>>> print(v_full.shape, v0.shape)

		"""

		if not hasattr(v0,"shape"):
			v0 = _np.asanyarray(v0)

		squeeze = False

		if v0.ndim == 1:
			shape = (self._sps**self._N,1)
			v0 = v0.reshape((-1,1))
			squeeze = True
		elif v0.ndim == 2:
			shape = (self._sps**self._N,v0.shape[1])
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")

		if self._Ns <= 0:
			if sparse:
				return _sp.csr_matrix(([],([],[])),shape=(self._sps**self._N,0),dtype=v0.dtype)
			else:
				return _np.zeros((self._sps**self._N,0),dtype=v0.dtype)

		if v0.shape[0] != self._Ns:
			raise ValueError("v0 shape {0} not compatible with Ns={1}".format(v0.shape,self._Ns))

		if _sp.issparse(v0): # current work around for sparse states.
			# return self.get_proj(v0.dtype).dot(v0)
			raise ValueError

		if not v0.flags["C_CONTIGUOUS"]:
			v0 = _np.ascontiguousarray(v0)

		if sparse:
			# current work-around for sparse
			return self.get_proj(v0.dtype).dot(_sp.csr_matrix(v0))
		else:
			v_out = _np.zeros(shape,dtype=v0.dtype,)
			self._core.get_vec_dense(self._basis,self._n,v0,v_out)
			if squeeze:
				return  _np.squeeze(v_out)
			else:
				return v_out	

	def _check_symm(self,static,dynamic,photon_basis=None):
		if photon_basis is None:
			basis_sort_opstr = self._sort_opstr
			static_list,dynamic_list = self._get_local_lists(static,dynamic)
		else:
			basis_sort_opstr = photon_basis._sort_opstr
			static_list,dynamic_list = photon_basis._get_local_lists(static,dynamic)


		static_blocks = {}
		dynamic_blocks = {}
		for block,map in self._maps_dict.items():
			key = block+" symm"
			odd_ops,missing_ops = _check_symm_map(map,basis_sort_opstr,static_list)
			if odd_ops or missing_ops:
				static_blocks[key] = (tuple(odd_ops),tuple(missing_ops))

			odd_ops,missing_ops = _check_symm_map(map,basis_sort_opstr,dynamic_list)
			if odd_ops or missing_ops:
				dynamic_blocks[key] = (tuple(odd_ops),tuple(missing_ops))


		return static_blocks,dynamic_blocks


def _check_symm_map(map,sort_opstr,operator_list):
	missing_ops=[]
	odd_ops=[]
	for op in operator_list:
		opstr = str(op[0])
		indx  = list(op[1])
		J     = op[2]
		for j,ind in enumerate(op[1]):
			i = map[ind]
			if i < 0:
				if opstr[j] == "n":
					odd_ops.append(op)

				J *= (-1 if opstr[j] in ["z","y"] else 1)
				opstr = opstr.replace("+","#").replace("-","+").replace("#","-")
				i = -(i+1)

			indx[j] = i

		new_op = list(op)
		new_op[0] = opstr
		new_op[1] = indx
		new_op[2] = J

		new_op = sort_opstr(new_op)
		if not (new_op in operator_list):
			missing_ops.append(new_op)

	return odd_ops,missing_ops







