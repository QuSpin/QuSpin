from .base import basis,MAXPRINT

import numpy as _np
from scipy import sparse as _sp
from scipy.sparse import linalg as _sla
from scipy import linalg as _la
from numpy.linalg import eigvalsh
import warnings

# gives the basis for the kronecker/Tensor product of two basis: |basis_left> (x) |basis_right>
class tensor_basis(basis):

	def __init__(self,basis_left,basis_right):
		if not isinstance(basis_left,basis):
			raise ValueError("basis_left must be instance of basis class")
		if not isinstance(basis_right,basis):
			raise ValueError("basis_right must be instance of basis class")
		if isinstance(basis_left,tensor_basis): 
			raise TypeError("Can only create tensor basis with non-tensor type basis")
		if isinstance(basis_right,tensor_basis): 
			raise TypeError("Can only create tensor basis with non-tensor type basis")
		self._basis_left=basis_left
		self._basis_right=basis_right

		self._Ns = basis_left.Ns*basis_right.Ns
		self._dtype = _np.min_scalar_type(-self._Ns)

		self._blocks = self._basis_left._blocks.copy()
		self._blocks.update(self._basis_right._blocks)

		self._unique_me = basis_left.unique_me and basis_left.unique_me
		self._operators = self._basis_left._operators +"\n"+ self._basis_right._operators


	@property
	def basis_left(self):
		return self._basis_left

	@property
	def basis_right(self):
		return self._basis_right

	def Op(self,opstr,indx,J,dtype):
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("not enough indices for opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx_left = indx[:i]
		indx_right = indx[i:]

		opstr_left,opstr_right=opstr.split("|")

		if self._basis_left._Ns < self._basis_right._Ns:
			ME_left,row_left,col_left = self._basis_left.Op(opstr_left,indx_left,J,dtype)
			ME_right,row_right,col_right = self._basis_right.Op(opstr_right,indx_right,1.0,dtype)
		else:
			ME_left,row_left,col_left = self._basis_left.Op(opstr_left,indx_left,1.0,dtype)
			ME_right,row_right,col_right = self._basis_right.Op(opstr_right,indx_right,J,dtype)
			

		n1 = row_left.shape[0]
		n2 = row_right.shape[0]


		if n1 > 0 and n2 > 0:
			row_left = row_left.astype(self._dtype)
			row_left *= self._basis_right.Ns
			row = _np.kron(row_left,_np.ones_like(row_right,dtype=_np.int8))
			row += _np.kron(_np.ones_like(row_left,dtype=_np.int8),row_right)

			del row_left,row_right

			col_left = col_left.astype(self._dtype)
			col_left *= self._basis_right.Ns
			col = _np.kron(col_left,_np.ones_like(col_right,dtype=_np.int8))
			col += _np.kron(_np.ones_like(col_left,dtype=_np.int8),col_right)

			del col_left,col_right

			ME = _np.kron(ME_left,ME_right)

			del ME_left,ME_right
		else:
			row = _np.array([])
			col = _np.array([])
			ME = _np.array([])


		return ME,row,col




	def get_vec(self,v0,sparse=True,full_left=True,full_right=True):
		if self._Ns <= 0:
			return _np.array([])

		if v0.shape[0] != self._Ns:
			raise ValueError("v0 has incompatible dimensions with basis")

		if v0.ndim == 1:
			v0 = v0.reshape((-1,1))
			if sparse:
				return _combine_get_vecs(self,v0,sparse,full_left,full_right)
			else:
				return _combine_get_vecs(self,v0,sparse,full_left,full_right).reshape((-1,))
		elif v0.ndim == 2:
			return _combine_get_vecs(self,v0,sparse,full_left,full_right)
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")



	def index(self,s_left,s_right):
		s_left = self.basis_left.index(s_left)
		s_right = self.basis_right.index(s_right)
		return s_right + self.basis_right.Ns*s_left


	def get_proj(self,dtype,full_left=True,full_right=True):
		if full_left:
			proj1 = self._basis_left.get_proj(dtype)
		else:
			proj1 = _sp.identity(self._basis_left.Ns,dtype=dtype)

		if full_right:
			proj2 = self._basis_right.get_proj(dtype)
		else:
			proj2 = _sp.identity(self._basis_right.Ns,dtype=dtype)


		return _sp.kron(proj1,proj2,format="csr")



	def partial_trace(self,state,sub_sys_A="left",state_type="pure",sparse=False):
		"""
		This function calculates the reduced density matrix (DM), performing a partial trace 
		of a quantum state.

		RETURNS: reduced DM

		--- arguments ---

		state: (required) the state of the quantum system. Can be a:

				-- pure state (default) [numpy array of shape (Ns,)].

				-- density matrix [numpy array of shape (Ns,Ns)].

				-- collection of states [dictionary {'V_states':V_states}] containing the states
					in the columns of V_states [shape (Ns,Nvecs)]

		sub_sys_A: (optional) flag to define the subsystem retained after the partial trace is taken.

				-- 'left': str, subsystem corresponds to the first tensor basis 

				-- 'right': str, subsystem corresponds to second tensor basis

				-- 'both': str, DM corresponding to both subsystems are returned

		state_type: (optional) flag to determine if 'state' is a collection of pure states or
						a density matrix

				-- 'pure': (default) (a collection of) pure state(s)

				-- 'mixed': mixed state (i.e. a density matrix)

		sparse: (optional) flag to enable usage of sparse linear algebra algorithms.

		"""

		if sub_sys_A not in set(["left","right","both"]):
			raise ValueError("sub_sys_A must be 'left' or 'right' or 'both'.")

		Ns = self._basis_left.Ns*self._basis_right.Ns

		if _sp.issparse(state):
			if state_type == "pure":
				if state.shape[-1] != Ns:
					raise ValueError("state shape {0} not compatible with Ns={1}".format(state.shape,self._Ns))

				if state.shape[0] == 1:
					return _tensor_partial_trace_sparse_pure(state,self._basis_left.Ns,self._basis_right.Ns,sub_sys_A=sub_sys_A)
				else:
					state = state.tocsr()
					try:
						gen = (_tensor_partial_trace_sparse_pure(state.getrow(i),self._basis_left.Ns,self._basis_right.Ns,sub_sys_A=sub_sys_A) for i in xrange(state.shape[0]))
					except NameError:
						gen = gen = (_tensor_partial_trace_sparse_pure(state.getrow(i),self._basis_left.Ns,self._basis_right.Ns,sub_sys_A=sub_sys_A) for i in range(state.shape[0]))
					return _np.stack(gen)

			elif state_type == "mixed":
				raise NotImplementedError("only pure state calculation implemeted for sparse arrays")

			else:
				raise ValueError("state_type '{}' not recognized.".format(state_type))

		else:
			state = _np.asanyarray(state)
			if state.shape[-1] != Ns:
				raise ValueError("state shape {0} not compatible with Ns={1}".format(state.shape,self._Ns))

			if state_type == "pure":

				return _tensor_partial_trace_pure(state,self._basis_left.Ns,self._basis_right.Ns,sub_sys_A=sub_sys_A)
				
			elif state_type == "mixed":
				if state.ndim < 2:
					raise ValueError("mixed state input must be a single or collection of 2-d square array(s)")

				if state.shape[-1] != state.shape[-2]:
					raise ValueError("mixed state input must be a single or collection of 2-d square array(s)")

				return _tensor_partial_trace_mixed(state,self._basis_left.Ns,self._basis_right.Ns,sub_sys_A=sub_sys_A)

			else:
				raise ValueError("state_type '{}' not recognized.".format(state_type))


	def ent_entropy(self,state,return_rdm=None,state_type="pure",alpha=1.0,sparse=False):
		"""
		This function calculates the entanglement entropy of the two chains used to 
		construct the tensor basis, and the corresponding reduced density matrix. In the following,
		the two chains are denoted 'left' and 'right'.

		RETURNS: dictionary with keys:

		'Sent': entanglement entropy.
		'rdm_left': (optional) reduced density matrix of subsystem A
		'rdm_right': (optional) reduced density matrix of subsystem B

		--- arguments ---

		state: (required) the state of the quantum system. Can be a:

				-- pure state (default) [numpy array of shape (Ns,)].

				-- density matrix [numpy array of shape (Ns,Ns)].

				-- collection of states [dictionary {'V_states':V_states}] containing the states
					in the columns of V_states [shape (Ns,Nvecs)]

		return_rdm: (optional) flag to return the reduced density matrix. Default is 'None'.

				-- 'left': str, returns reduced DM of photon 

				-- 'right': str, returns reduced DM of chain

				-- 'both': str, returns reduced DM of both photon and chain

		state_type: (optional) flag to determine if 'state' is a collection of pure states or
						a density matrix

				-- 'pure': (default) (a collection of) pure state(s)

				-- 'mixed': mixed state (i.e. a density matrix)

		sparse: (optional) flag to enable usage of sparse linear algebra algorithms.

		alpha: (optional) Renyi alpha parameter. Default is '1.0'.

		"""

		if return_rdm is None:
			if self._basis_left.Ns <= self._basis_right.Ns:
				rdm = self.partial_trace(state,sub_sys_A="left",state_type=state_type,sparse=sparse)
			else:
				rdm = self.partial_trace(state,sub_sys_A="right",state_type=state_type,sparse=sparse)

		elif return_rdm == "left" and self._basis_left.Ns <= self._basis_right.Ns:
			rdm_left = self.partial_trace(state,sub_sys_A="left",state_type=state_type,sparse=sparse)
			rdm = rdm_left

		elif return_rdm == "right" and self._basis_right.Ns <= self._basis_left.Ns:
			rdm_right = self.partial_trace(state,sub_sys_A="right",state_type=state_type,sparse=sparse)
			rdm = rdm_right

		else:
			rdm_left,rdm_right = self.partial_trace(state,sub_sys_A="both",state_type=state_type,sparse=sparse)

			if self._basis_left.Ns < self._basis_right.Ns:
				rdm = rdm_left
			else:
				rdm = rdm_right

		try:
			E = eigvalsh(rdm.todense()) + _np.finfo(rdm.dtype).eps
		except AttributeError:
			if rdm.dtype == _np.object:
				E_gen = (eigvalsh(dm.todense())+_np.finfo(dm.dtype).eps for dm in rdm[:])
				E = _np.stack(E_gen)
			else:
				E = eigvalsh(rdm) + _np.finfo(rdm.dtype).eps

		if alpha == 1.0:
			Sent = - (E * _np.log(E)).sum(axis=-1)
		elif alpha >= 0.0:
			Sent = (_np.log(_np.power(E,alpha))/(1-alpha)).sum(axis=-1)
		else:
			raise ValueError("alpha >= 0")

		
		if return_rdm is None:
			return dict(Sent=Sent)
		elif return_rdm == "left":
			return dict(Sent=Sent,rdm_left=rdm_left)
		elif return_rdm == "right":
			return dict(Sent=Sent,rdm_right=rdm_right)
		elif return_rdm == "both":
			return dict(Sent=Sent,rdm_left=rdm_left,rdm_right=rdm_right)




	def __name__(self):
		return "<type 'qspin.basis.tensor_basis'>"

	def _sort_opstr(self,op):
		op = list(op)
		opstr = op[0]
		indx  = op[1]

		if opstr.count("|") == 0: 
			raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr,indx))
	
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx_left = indx[:i]
		indx_right = indx[i:]

		opstr_left,opstr_right=opstr.split("|")

		op1 = list(op)
		op1[0] = opstr_left
		op1[1] = tuple(indx_left)

		op2 = list(op)
		op2[0] = opstr_right
		op2[1] = tuple(indx_right)
		
		op1 = self._basis_left._sort_opstr(op1)
		op2 = self._basis_right._sort_opstr(op2)

		op[0] = "|".join((op1[0],op2[0]))
		op[1] = op1[1] + op2[1]
		
		return tuple(op)




	def _hc_opstr(self,op):
		op = list(op)
		opstr = op[0]
		indx  = op[1]
	
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx_left = indx[:i]
		indx_right = indx[i:]

		opstr_left,opstr_right=opstr.split("|")

		op1 = list(op)
		op1[0] = opstr_left
		op1[1] = indx_left
		op1[2] = op[2]

		op2 = list(op)
		op2[0] = opstr_right
		op2[1] = indx_right
		op2[2] = complex(1.0)
		
		op1 = self._basis_left._hc_opstr(op1)
		op2 = self._basis_right._hc_opstr(op2)

		op[0] = "|".join((op1[0],op2[0]))
		op[1] = op1[1] + op2[1]

		op[2] = op1[2]*op2[2]


		return tuple(op)
	

	def _non_zero(self,op):
		op = list(op)
		opstr = op[0]
		indx  = op[1]
	
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("number of indices doesn't match opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx_left = indx[:i]
		indx_right = indx[i:]

		opstr_left,opstr_right=opstr.split("|")

		op1 = list(op)
		op1[0] = opstr_left
		op1[1] = indx_left

		op2 = list(op)
		op2[0] = opstr_right
		op2[1] = indx_right

		return (self._basis_left._non_zero(op1) and self._basis_right._non_zero(op2))



	def _expand_opstr(self,op,num):
		op = list(op)
		opstr = op[0]
		indx  = op[1]
	
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("not enough indices for opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx_left = indx[:i]
		indx_right = indx[i:]

		opstr_left,opstr_right=opstr.split("|")

		op1 = list(op)
		op1[0] = opstr_left
		op1[1] = indx_left

		op2 = list(op)
		op2[0] = opstr_right
		op2[1] = indx_right

		op1_list = self._basis_left.expand_opstr(op1,num)
		op2_list = self._basis_right.expand_opstr(op2,num)

		op_list = []
		for new_op1 in op1_list:
			for new_op2 in op2_list:
				new_op = list(new_op1)
				new_op[0] = "|".join((new_op1[0],new_op2[0]))
				new_op[1] += tuple(new_op2[1])
				new_op[2] *= new_op2[2]


				op_list.append(tuple(new_op))

		return tuple(op_list)
			




	def _get__str__(self):
		if not hasattr(self._basis_left,"_get__str__"):
			warnings.warn("basis class {0} missing _get__str__ function, can not print out basis representatives.".format(type(self._basis_left)),UserWarning,stacklevel=3)
			return "reference states: \n\t not availible"

		if not hasattr(self._basis_right,"_get__str__"):
			warnings.warn("basis class {0} missing _get__str__ function, can not print out basis representatives.".format(type(self._basis_right)),UserWarning,stacklevel=3)
			return "reference states: \n\t not availible"

		n_digits = int(_np.ceil(_np.log10(self._Ns)))

		str_list_1 = self._basis_left._get__str__()
		str_list_2 = self._basis_right._get__str__()
		Ns2 = self._basis_right.Ns
		temp = "\t{0:"+str(n_digits)+"d}.  "
		str_list=[]
		for basis_left in str_list_1:
			basis_left,s1 = basis_left.split(".  ")
			i1 = int(basis_left)
			for basis_right in str_list_2:
				basis_right,s2 = basis_right.split(".  ")
				i2 = int(basis_right)
				str_list.append((temp.format(i2+Ns2*i1))+s1+s2)

		if self._Ns > MAXPRINT:
			half = MAXPRINT//2
			str_list_1 = str_list[:half]
			str_list_2 = str_list[-half:]

			str_list = str_list_1
			str_list.extend(str_list_2)	

		return str_list		




def _combine_get_vecs(basis,v0,sparse,full_left,full_right):
	Ns1=basis._basis_left.Ns
	Ns2=basis._basis_right.Ns

	Nvecs = v0.shape[1]

	Ns = min(Ns1,Ns2)

	# reshape vector to matrix to rewrite vector as an outer product.
	v0=v0.T.reshape((Nvecs,Ns1,Ns2))
	# take singular value decomposition to get which decomposes the matrix into separate parts.
	# the outer/tensor product of the cols of V1 and V2 are the product states which make up the original vector 

	V1,S,V2 = _np.linalg.svd(v0,full_matrices=False)
	S = S.T
	V1 = V1.transpose((2,1,0))
	V2 = V2.transpose((1,2,0))

	# combining all the vectors together with the tensor product as opposed to the outer product
	if sparse:
		# take the vectors and convert them to their full hilbert space
		v1 = V1[-1]
		v2 = V2[-1]

		if full_left:
			v1 = basis._basis_left.get_vec(v1,sparse=True)
			
		if full_right:
			v2 = basis._basis_right.get_vec(v2,sparse=True)


		temp1 = _np.ones((v1.shape[0],1),dtype=_np.int8)
		temp2 = _np.ones((v2.shape[0],1),dtype=_np.int8)

		v1 = _sp.kron(v1,temp2,format="csr")
		v2 = _sp.kron(temp1,v2,format="csr")

		s = _np.array(S[-1])
		s = _np.broadcast_to(s,v1.shape)

		v0 = v1.multiply(v2).multiply(s)
		
		for i,s in enumerate(S[:-1]):
			v1 = V1[i]
			v2 = V2[i]

			if full_left:
				v1 = basis._basis_left.get_vec(v1,sparse=True)
			
			if full_right:
				v2 = basis._basis_right.get_vec(v2,sparse=True)


			v1 = _sp.kron(v1,temp2,format="csr")  
			v2 = _sp.kron(temp1,v2,format="csr")

			s = _np.broadcast_to(s,v1.shape)
			v = v1.multiply(v2).multiply(s)

			v0 = v0 + v
		
		
	else:
		# take the vectors and convert them to their full hilbert space
		v1 = V1[-1]
		v2 = V2[-1]

		if full_left:
			v1 = basis._basis_left.get_vec(v1,sparse=False)
			
		if full_right:
			v2 = basis._basis_right.get_vec(v2,sparse=False)


		temp1 = _np.ones((v1.shape[0],1),dtype=_np.int8)
		temp2 = _np.ones((v2.shape[0],1),dtype=_np.int8)

		v1 =  _np.kron(v1,temp2)
		v2 = _np.kron(temp1,v2)
		v0 = _np.multiply(v1,v2)
		v0 *= S[-1]

		for i,s in enumerate(S[:-1]):
			v1 = V1[i]
			v2 = V2[i]

			if full_left:
				v1 = basis._basis_left.get_vec(v1,sparse=False)
			
			if full_right:
				v2 = basis._basis_right.get_vec(v2,sparse=False)

			v1 =  _np.kron(v1,temp2)
			v2 = _np.kron(temp1,v2)
			v = _np.multiply(v1,v2)
			v0 += s*v



	return v0




def _tensor_partial_trace_pure(psi,Ns_l,Ns_r,sub_sys_A="left"):
	extra_dims = psi.shape[:-1]
	psi_v = psi.reshape(extra_dims+(Ns_l,Ns_r))

	if sub_sys_A == "left":
		return _np.squeeze(_np.einsum("...ij,...kj->...ik",psi_v,psi_v.conj()))
	elif sub_sys_A == "right":
		return _np.squeeze(_np.einsum("...ji,...jk->...ik",psi_v,psi_v.conj()))
	elif sub_sys_A == "both":
		return _np.squeeze(_np.einsum("...ij,...kj->...ik",psi_v,psi_v.conj())),_np.squeeze(_np.einsum("...ji,...jk->...ik",psi_v,psi_v.conj()))

	

def _tensor_partial_trace_sparse_pure(psi,Ns_l,Ns_r,sub_sys_A="left"):
	if not _sp.issparse(psi):
		raise Exception

	if psi.shape[0] > 1:
		raise Exception

	psi = psi.tocoo()
	# make shift way of reshaping array
	# j = j_l + Ns_r * j_l
	# j_l = j / Ns_r
	# j_r = j % Ns_r 
	psi._shape = (Ns_l,Ns_r)
	psi.row[:] = psi.col / Ns_r
	psi.col[:] = psi.col % Ns_r

	psi = psi.tocsr()

	if sub_sys_A == "left":
		return psi.dot(psi.H)
	elif sub_sys_A == "right":
		return psi.H.dot(psi)
	elif sub_sys_A == "both":
		return psi.dot(psi.H),psi.H.dot(psi)


def _tensor_partial_trace_mixed(rho,Ns_l,Ns_r,sub_sys_A="left"):
	extra_dims = rho.shape[:-2]
	psi_v = rho.reshape(extra_dims+(Ns_l,Ns_r,Ns_l,Ns_r))

	if sub_sys_A == "left":
		return _np.squeeze(_np.einsum("...ijkj->...ik",psi_v))
	elif sub_sys_A == "right":
		return _np.squeeze(_np.einsum("...jijk->...ik",psi_v))
	elif sub_sys_A == "both":
		return _np.squeeze(_np.einsum("...ijkj->...ik",psi_v)),_np.squeeze(_np.einsum("...jijk->...ik",psi_v))






		





"""
def _combine_get_vec(basis,v0,sparse,full_left,full_right):
	Ns1=basis._basis_left.Ns
	Ns2=basis._basis_right.Ns

	Ns = min(Ns1,Ns2)

	# reshape vector to matrix to rewrite vector as an outer product.
	v0=_np.reshape(v0,(Ns1,Ns2))
	# take singular value decomposition to get which decomposes the matrix into separate parts.
	# the outer/tensor product of the cols of V1 and V2 are the product states which make up the original vector 

	if sparse:
		V1,S,V2=_sla.svds(v0,k=Ns-1,which='SM',maxiter=1E10)
		V12,[S2],V22=_sla.svds(v0,k=1,which='LM',maxiter=1E10)

		S.resize((Ns,))
		S[-1] = S2
		V1.resize((Ns1,Ns))
		V1[:,-1] = V12[:,0]
		V2.resize((Ns,Ns2))
		V2[-1,:] = V22[0,:]
	else:
		V1,S,V2=_la.svd(v0)
		
	# svd returns V2.H so take the hc to reverse that
	V2=V2.T.conj()
	eps = _np.finfo(S.dtype).eps
	# for any values of s which are 0, remove those vectors because they do not contribute.
	mask=(S >= 10*eps)
	V1=V1[:,mask]
	V2=V2[:,mask]
	S=S[mask]


	# Next thing to do is take those vectors and convert them to their full hilbert space
	if full_left:
		V1=basis._basis_left.get_vec(V1,sparse)

	if full_right:
		V2=basis._basis_right.get_vec(V2,sparse)


	# calculate the dimension total hilbert space with no symmetries
	Ns=V2.shape[0]*V1.shape[0]		


	if sparse:
		v0=_sp.csr_matrix((Ns,1),dtype=V2.dtype)
		# combining all the vectors together with the tensor product as opposed to the outer product
		for i,s in enumerate(S):
			v1=V1.getcol(i)
			v2=V2.getcol(i)
			v=_sp.kron(v1,v2)
			v0 = v0 + s*v
		n=_np.sqrt(v0.multiply(v0.conj()).sum())
#		v0=v0/n
		v0=v0.astype(V1.dtype)
		
		
	else:
		v0=_np.zeros((Ns,),dtype=V2.dtype)
		for i,s in enumerate(S):
			v1=V1[:,i]
			v2=V2[:,i]
			v=_np.kron(v1,v2)
			v0 += s*v
		v0 /= _la.norm(v0)


	return v0




def _combine_get_vecs(basis,V0,sparse,full_left,full_right):
	v0_list=[]
	V0=V0.T
	for v0 in V0:
		v0_list.append(_combine_get_vec(basis,v0,sparse,full_left,full_right))

	if sparse:
		V0=_sp.hstack(v0_list)
	else:
		V0=_np.hstack(v0_list)

	return V0
"""

