import numpy as _np
from scipy import sparse as _sp
from scipy.sparse import linalg as _sla
from scipy import linalg as _la


# this file stores

class basis(object):
	def __init__(self):
		self._Ns = 0
		self._basis = _np.asarray([])
		self._operators = ""
		if self.__class__.__name__ == 'basis':
			raise ValueError("This class is not intended"
                             " to be instantiated directly.")



	@property
	def Ns(self):
		return self._Ns

	@property
	def operators(self):
		return self._operators

	def __repr__(self):
		return "< exact_diag_py basis with {0} states >".format(self._Ns)




# gives the basis for the kronecker/Tensor product of two basis: b1 (x) b2
class tensor(basis):
	def __init__(self,b1,b2):
		if not isbasis(b1):
			raise ValueError("b1 must be instance of basis class")
		if not isbasis(b2):
			raise ValueError("b2 must be instance of basis class")

		self._b1=b1
		self._b2=b2

		self._Ns = b1.Ns*b2.Ns
		self.dtype = _np.min_scalar_type(-self._Ns)

		self._operators = self._b1._operators + self._b2._operators


	def __str__(self):
		return "basis 1:\n"+self._b1.__str__() + "\n basis 2: \n" + self._b2.__str__()

		


	def get_vec(self,v0,sparse=True):
		if self._Ns <= 0:
			return _np.array([])

		if v0.ndim == 1:
			if v0.shape[0] != self._Ns:
				raise ValueError("v0 has incompatible dimensions with basis")
			v0 = v0.reshape((-1,1))
			return _combine_get_vec(self,v0,sparse)
		elif v0.ndim == 2:
			if v0.shape[0] != self._Ns:
				raise ValueError("v0 has incompatible dimensions with basis")
			return _combine_get_vecs(self,v0,sparse)
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")



	def get_proj(self,dtype):
		proj1 = self._b1.get_proj(dtype)
		proj2 = self._b2.get_proj(dtype)

		return _sp.kron(proj1,proj2)


	def Op(self,dtype,J,opstr,indx,pauli):
		n=opstr.count("|")
		if n > 1: 
			raise ValueError("only one '|' charactor allowed")
		i = opstr.index("|")
		indx1 = indx[:i]
		indx2 = indx[i:]

		opstr1,opstr2=opstr.split("|")
#		print opstr1,indx1
#		print opstr2,indx2

		ME1,row1,col1 = self._b1.Op(dtype,1.0,opstr1,indx1,pauli)
		ME2,row2,col2 = self._b2.Op(dtype,1.0,opstr2,indx2,pauli)

		n1 = row1.shape[0]
		n2 = row2.shape[0]

		row1 = _np.array(_np.broadcast_to(row1,(n2,n1)).T,dtype=self.dtype).ravel()
		row2 = _np.array(_np.broadcast_to(row2,(n1,n2)),dtype=self.dtype).ravel()
		row1 *= self._b2.Ns
		row = row1+row2

		del row1,row2

		col1 = _np.array(_np.broadcast_to(col1,(n2,n1)).T,dtype=self.dtype).ravel()
		col2 = _np.array(_np.broadcast_to(col2,(n1,n2)),dtype=self.dtype).ravel()
		col1 *= self._b2.Ns
		col = col1+col2

		del col1,col2

		ME = _np.kron(ME1,ME2)
		ME = J*ME

		del ME1,ME2

		return ME,row,col


		






def _combine_get_vec(basis,v0,sparse):
	Ns1=basis.b1.Ns
	Ns2=basis.b2.Ns

	Ns = min(Ns1,Ns2)

	# reshape vector to matrix to rewrite vector as an outer product.
	v0=_np.reshape(v0,(Ns1,Ns2))
	# take singular value decomposition to get which decomposes the matrix into separate parts.
	# the outer/tensor product of the cols of V1 and V2 are the product states which make up the original vector 
	if k<=0:
		V1,S,V2=_la.svd(v0)
	else:
		V1,S,V2=_sla.svds(v0,k=Ns-1,which='LM',maxiter=10**10)
		V12,S,V22=_sla.svds(v0,k=1,which='SM',maxiter=10**10)
		V1 = _np.hstack(V1,V12)
		V2 = _np.hstack(V2,V22)
	# svd returns V2.H so take the hc to reverse that
	V2=V2.T.conj()
	eps = _np.finfo(S.dtype).eps
	print S
	# for any values of s which are 0, remove those vectors because they do not contribute.
	mask=(S >= 10*eps)
	V1=V1[:,mask]
	V2=V2[:,mask]
	S=S[mask]


	# Next thing to do is take those vectors and convert them to their full hilbert space
	V1=basis.b1.get_vec(V1,sparse)
	V2=basis.b2.get_vec(V2,sparse)


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




def _combine_get_vecs(basis,V0,sparse):
	v0_list=[]
	V0=V0.T
	for v0 in V0:
		v0_list.append(_combine_get_vec(basis,v0,sparse))

	if sparse:
		V0=_sp.hstack(v0_list)
	else:
		V0=_np.hstack(v0_list)

	return V0













def isbasis(x):
	return isinstance(x,basis)


