from .base import basis,MAXPRINT

import numpy as _np
from scipy import sparse as _sp
from scipy.sparse import linalg as _sla
from scipy import linalg as _la
from numpy.linalg import eigvalsh
import warnings


# gives the basis for the kronecker/Tensor product of two basis: b1 (x) b2
class tensor_basis(basis):

	def __init__(self,b1,b2):
		if not isinstance(b1,basis):
			raise ValueError("b1 must be instance of basis class")
		if not isinstance(b2,basis):
			raise ValueError("b2 must be instance of basis class")
		if isinstance(b1,tensor_basis): 
			raise TypeError("Can only create tensor basis with non-tensor type basis")
		if isinstance(b2,tensor_basis): 
			raise TypeError("Can only create tensor basis with non-tensor type basis")
		self._b1=b1
		self._b2=b2

		self._Ns = b1.Ns*b2.Ns
		self._dtype = _np.min_scalar_type(-self._Ns)

		self._blocks = self._b1._blocks.copy()
		self._blocks.update(self._b2._blocks)

		self._unique_me = b1.unique_me and b1.unique_me
		self._operators = self._b1._operators +"\n"+ self._b2._operators
#		self._check_pcon = self._b1._check_pcon and self._b2._check_pcon


	def Op(self,opstr,indx,J,dtype):
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("not enough indices for opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx1 = indx[:i]
		indx2 = indx[i:]

		opstr1,opstr2=opstr.split("|")

		if self._b1._Ns < self._b2._Ns:
			ME1,row1,col1 = self._b1.Op(opstr1,indx1,J,dtype)
			ME2,row2,col2 = self._b2.Op(opstr2,indx2,1.0,dtype)
		else:
			ME1,row1,col1 = self._b1.Op(opstr1,indx1,1.0,dtype)
			ME2,row2,col2 = self._b2.Op(opstr2,indx2,J,dtype)
			

		n1 = row1.shape[0]
		n2 = row2.shape[0]


		if n1 > 0 and n2 > 0:
			row1 = row1.astype(self._dtype)
			row1 *= self._b2.Ns
			row = _np.kron(row1,_np.ones_like(row2,dtype=_np.int8))
			row += _np.kron(_np.ones_like(row1,dtype=_np.int8),row2)

			del row1,row2

			col1 = col1.astype(self._dtype)
			col1 *= self._b2.Ns
			col = _np.kron(col1,_np.ones_like(col2,dtype=_np.int8))
			col += _np.kron(_np.ones_like(col1,dtype=_np.int8),col2)

			del col1,col2

			ME = _np.kron(ME1,ME2)

			del ME1,ME2
		else:
			row = _np.array([])
			col = _np.array([])
			ME = _np.array([])


		return ME,row,col




	def get_vec(self,v0,sparse=True,full_1=True,full_2=True):
		if self._Ns <= 0:
			return _np.array([])

		if v0.shape[0] != self._Ns:
			raise ValueError("v0 has incompatible dimensions with basis")

		if v0.ndim == 1:
			v0 = v0.reshape((-1,1))
			if sparse:
				return _combine_get_vecs(self,v0,sparse,full_1,full_2)
			else:
				return _combine_get_vecs(self,v0,sparse,full_1,full_2).reshape((-1,))
		elif v0.ndim == 2:
			return _combine_get_vecs(self,v0,sparse,full_1,full_2)
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")




	def get_proj(self,dtype,full_1=True,full_2=True):
		if full_1:
			proj1 = self._b1.get_proj(dtype)
		else:
			proj1 = _sp.identity(self._b1.Ns,dtype=dtype)

		if full_2:
			proj2 = self._b2.get_proj(dtype)
		else:
			proj2 = _sp.identity(self._b2.Ns,dtype=dtype)


		return _sp.kron(proj1,proj2,format="csr")



	def partial_trace(self,state,sub_sys_A="left",state_type="pure"):
		if sub_sys_A not in set(["left","right","both"]):
			raise ValueError("sub_sys_A must be 'left' or 'right' or 'both'.")

		if _sp.issparse(state):
			if state_type == "pure":
				if state.shape[0] != self.Ns:
					raise ValueError("pure state must have dimension equal to basis size.")

				if state.shape[1] == 1:
					return _tensor_partial_trace_sparse_pure(state,self._b1.Ns,self._b2.Ns,sub_sys_A=sub_sys_A)
				else:
					state = state.tocsc()
					return _np.array([_tensor_partial_trace_sparse_pure(state.getcol(i),self._b1.Ns,self._b2.Ns,sub_sys_A=sub_sys_A)],dtype=_np.object)

			elif state_type == "mixed":
				raise NotImplementedError("only pure state calculation implemeted for sparse arrays")

			else:
				raise ValueError("state_type '{}' not recognized.".format(state_type))

		else:
			state = _np.asanyarray(state)
			if state_type == "pure":

				if state.ndim == 1:
					if state.shape[0] != self.Ns:
						raise ValueError("expecting 1-d array containing pure state.")

					return _tensor_partial_trace_pure(state,self._b1.Ns,self._b2.Ns,sub_sys_A=sub_sys_A)

				elif state.ndim == 2:
					if state.shape[0] != self.Ns:
						raise ValueError("expecting 2-d array containing pure states in columns of array.")

					return _tensor_partial_trace_pure(state.T,self._b1.Ns,self._b2.Ns,sub_sys_A=sub_sys_A)

				else:
					raise ValueError("pure state calculation expecting arrays with ndim < 3.")

					
			elif state_type == "mixed":
				if state.ndim != 2 or state.shape[0] != state.shape[1]:
					raise ValueError("mixed state input must be a 2-d square array")
				
				return _tensor_partial_trace_mixed(state,self._b1.Ns,self._b2.Ns,sub_sys_A=sub_sys_A)

			else:
				raise ValueError("state_type '{}' not recognized.".format(state_type))



	def ent_entropy(self,state,return_rdm=None,state_type="pure",alpha=1.0):

		if return_rdm is None:
			if self._b1.Ns <= self._b2.Ns:
				rdm = self.partial_trace(state,sub_sys_A="left",state_type=state_type)
			else:
				rdm = self.partial_trace(state,sub_sys_A="right",state_type=state_type)

		elif return_rdm == "left" and self._b1.Ns <= self._b2.Ns:
			rdm_left = self.partial_trace(state,sub_sys_A="left",state_type=state_type)
			rdm = rdm_left

		elif return_rdm == "right" and self._b2.Ns <= self._b1.Ns:
			rdm_right = self.partial_trace(state,sub_sys_A="right",state_type=state_type)
			rdm = rdm_right

		else:
			rdm_left,rdm_right = self.partial_trace(state,sub_sys_A=return_rdm,state_type=state_type)

			if self._b1.Ns < self._b2.Ns:
				rdm = rdm_left
			else:
				rdm = rdm_right

		try:
			E = eigvalsh(rdm.todense()) + _np.finfo(rdm.dtype).eps
		except AttributeError:
			E = eigvalsh(rdm) + _np.finfo(rdm.dtype).eps

		if alpha == 1.0:
			Sent = - (E * _np.log(E)).sum(axis=-1)
		elif alpha >= 0.0:
			Sent = (_np.log(_np.power(E,alpha))/(1-alpha)).sum(axis=-1)
		else:
			raise ValueError("alpha >= 0")

		
		if return_rdm is None:
			return Sent
		elif return_rdm == "left":
			return Sent,rdm_left
		elif return_rdm == "right":
			return Sent,rdm_right
		elif return_rdm == "both":
			return Sent,rdm_left,rdm_right





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
		indx1 = indx[:i]
		indx2 = indx[i:]

		opstr1,opstr2=opstr.split("|")

		op1 = list(op)
		op1[0] = opstr1
		op1[1] = tuple(indx1)

		op2 = list(op)
		op2[0] = opstr2
		op2[1] = tuple(indx2)
		
		op1 = self._b1._sort_opstr(op1)
		op2 = self._b2._sort_opstr(op2)

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
		indx1 = indx[:i]
		indx2 = indx[i:]

		opstr1,opstr2=opstr.split("|")

		op1 = list(op)
		op1[0] = opstr1
		op1[1] = indx1

		op2 = list(op)
		op2[0] = opstr2
		op2[1] = indx2
		
		op1 = self._b1.hc_opstr(op1)
		op2 = self._b2.hc_opstr(op2)

		op[0] = "|".join((op1[0],op2[0]))
		op[1] = op1[1] + op2[1]

		op[2] = op[2].conjugate()

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
		indx1 = indx[:i]
		indx2 = indx[i:]

		opstr1,opstr2=opstr.split("|")

		op1 = list(op)
		op1[0] = opstr1
		op1[1] = indx1

		op2 = list(op)
		op2[0] = opstr2
		op2[1] = indx2

		return (self._b1.non_zero(op1) and self._b2.non_zero(op2))



	def _expand_opstr(self,op,num):
		op = list(op)
		opstr = op[0]
		indx  = op[1]
	
		if opstr.count("|") > 1: 
			raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

		if len(opstr)-1 != len(indx):
			raise ValueError("not enough indices for opstr in: {0}, {1}".format(opstr,indx))

		i = opstr.index("|")
		indx1 = indx[:i]
		indx2 = indx[i:]

		opstr1,opstr2=opstr.split("|")

		op1 = list(op)
		op1[0] = opstr1
		op1[1] = indx1

		op2 = list(op)
		op2[0] = opstr2
		op2[1] = indx2

		op1_list = self._b1.expand_opstr(op1,num)
		op2_list = self._b2.expand_opstr(op2,num)

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
		if not hasattr(self._b1,"_get__str__"):
			warnings.warn("basis class {0} missing _get__str__ function, can not print out basis representatives.".format(type(self._b1)),UserWarning,stacklevel=3)
			return "reference states: \n\t not availible"

		if not hasattr(self._b2,"_get__str__"):
			warnings.warn("basis class {0} missing _get__str__ function, can not print out basis representatives.".format(type(self._b2)),UserWarning,stacklevel=3)
			return "reference states: \n\t not availible"

		n_digits = int(_np.ceil(_np.log10(self._Ns)))

		str_list_1 = self._b1._get__str__()
		str_list_2 = self._b2._get__str__()
		Ns2 = self._b2.Ns
		temp = "\t{0:"+str(n_digits)+"d}.  "
		str_list=[]
		for b1 in str_list_1:
			b1,s1 = b1.split(".  ")
			i1 = int(b1)
			for b2 in str_list_2:
				b2,s2 = b2.split(".  ")
				i2 = int(b2)
				str_list.append((temp.format(i2+Ns2*i1))+s1+s2)

		if self._Ns > MAXPRINT:
			half = MAXPRINT//2
			str_list_1 = str_list[:half]
			str_list_2 = str_list[-half:]

			str_list = str_list_1
			str_list.extend(str_list_2)	

		return str_list		




def _combine_get_vecs(basis,v0,sparse,full_1,full_2):
	Ns1=basis._b1.Ns
	Ns2=basis._b2.Ns

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

		if full_1:
			v1 = basis._b1.get_vec(v1,sparse=True)
			
		if full_2:
			v2 = basis._b2.get_vec(v2,sparse=True)


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

			if full_1:
				v1 = basis._b1.get_vec(v1,sparse=True)
			
			if full_2:
				v2 = basis._b2.get_vec(v2,sparse=True)


			v1 = _sp.kron(v1,temp2,format="csr")  
			v2 = _sp.kron(temp1,v2,format="csr")

			s = _np.broadcast_to(s,v1.shape)
			v = v1.multiply(v2).multiply(s)

			v0 = v0 + v
		
		
	else:
		# take the vectors and convert them to their full hilbert space
		v1 = V1[-1]
		v2 = V2[-1]

		if full_1:
			v1 = basis._b1.get_vec(v1,sparse=False)
			
		if full_2:
			v2 = basis._b2.get_vec(v2,sparse=False)


		temp1 = _np.ones((v1.shape[0],1),dtype=_np.int8)
		temp2 = _np.ones((v2.shape[0],1),dtype=_np.int8)

		v1 =  _np.kron(v1,temp2)
		v2 = _np.kron(temp1,v2)
		v0 = _np.multiply(v1,v2)
		v0 *= S[-1]

		for i,s in enumerate(S[:-1]):
			v1 = V1[i]
			v2 = V2[i]

			if full_1:
				v1 = basis._b1.get_vec(v1,sparse=False)
			
			if full_2:
				v2 = basis._b2.get_vec(v2,sparse=False)

			v1 =  _np.kron(v1,temp2)
			v2 = _np.kron(temp1,v2)
			v = _np.multiply(v1,v2)
			v0 += s*v



	return v0




def _tensor_partial_trace_pure(psi,Ns_l,Ns_r,sub_sys_A="left"):
	psi_v = psi.reshape((-1,Ns_r,Ns_l))

	if sub_sys_A == "left":
		return _np.squeeze(_np.einsum("...ji,...jk->...ik",psi_v,psi_v))
	elif sub_sys_A == "right":
		return _np.squeeze(_np.einsum("...ij,...kj->...ik",psi_v,psi_v))
	elif sub_sys_A == "both":
		return _np.squeeze(_np.einsum("...ji,...jk->...ik",psi_v,psi_v)),_np.squeeze(_np.einsum("...ij,...kj->...ik",psi_v,psi_v))
	else:
		raise Exception
	

def _tensor_partial_trace_sparse_pure(psi,Ns_l,Ns_r,sub_sys_A="left"):
	if not _sp.issparse(psi):
		raise Exception

	if psi.shape[1] > 1:
		raise Exception

	# make shift way of reshaping array 
	psi = psi.tocoo()
	psi._shape = (Ns_r,Ns_l)
	psi.col[:] = psi.row / Ns_r
	psi.row[:] = psi.row % Ns_r
	psi = psi.tocsr()

	if sub_sys_A == "left":
		return psi.T.dot(psi)
	elif sub_sys_A == "right":
		return psi.dot(psi.T)
	elif sub_sys_A == "both":
		return psi.T.dot(psi),psi.dot(psi.T)
	else:
		raise Exception

def _tensor_partial_trace_mixed(rho,Ns_l,Ns_r,sub_sys_A="left"):
	psi_v = psi.reshape((-1,Ns_r,Ns_l,Ns_r,Ns_l))

	if sub_sys_A == "left":
		return _np.squeeze(_np.einsum("...jijk->...ik",psi_v))
	elif sub_sys_A == "right":
		return _np.squeeze(_np.einsum("...ijkj->...ik",psi_v))
	elif sub_sys_A == "both":
		return _np.squeeze(_np.einsum("...jijk->...ik",psi_v)),_np.squeeze(_np.einsum("...ijkj->...ik",psi_v))
	else:
		raise Exception





		





"""
def _combine_get_vec(basis,v0,sparse,full_1,full_2):
	Ns1=basis._b1.Ns
	Ns2=basis._b2.Ns

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
	if full_1:
		V1=basis._b1.get_vec(V1,sparse)

	if full_2:
		V2=basis._b2.get_vec(V2,sparse)


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




def _combine_get_vecs(basis,V0,sparse,full_1,full_2):
	v0_list=[]
	V0=V0.T
	for v0 in V0:
		v0_list.append(_combine_get_vec(basis,v0,sparse,full_1,full_2))

	if sparse:
		V0=_sp.hstack(v0_list)
	else:
		V0=_np.hstack(v0_list)

	return V0
"""

