from __future__ import print_function
from .base import basis,MAXPRINT

import numpy as _np
from scipy import sparse as _sp
from scipy.sparse import linalg as _sla
from scipy import linalg as _la
from scipy.sparse.linalg import eigsh
from numpy.linalg import eigvalsh,svd
import warnings

_dtypes={"f":_np.float32,"d":_np.float64,"F":_np.complex64,"D":_np.complex128}

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

		if not hasattr(v0,"shape"):
			v0 = _np.asanyarray(v0)

		if v0.shape[0] != self._Ns:
			raise ValueError("v0 has incompatible dimensions with basis")

		if v0.ndim == 1:
			v0 = v0.reshape((-1,1))
			if sparse:
				return _combine_get_vecs(self,v0,sparse,full_left,full_right)
			else:
				return _combine_get_vecs(self,v0,sparse,full_left,full_right).reshape((-1,))
		elif v0.ndim == 2:

			if _sp.issparse(v0):
				return self.get_proj(v0.dtype,full_left=full_left,full_right=full_right).dot(v0)

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

	def _p_pure(self,state,sub_sys_A,return_rdm=None):
		
		# put states in rows
		state=state.T
		# reshape state according to sub_sys_A
		Ns_left = self._basis_left.Ns
		Ns_right = self._basis_right.Ns
		v=_tensor_reshape_pure(state,Ns_left,Ns_right)


		rdm_A=None
		rdm_B=None

		# perform SVD	
		if return_rdm is None:
			lmbda = svd(v, compute_uv=False) 
		else:
			U, lmbda, V = svd(v, full_matrices=False)
			if return_rdm=='A':
				rdm_A = _np.einsum('...ij,...j,...kj->...ik',U,lmbda**2,U.conj() )
			elif return_rdm=='B':
				rdm_B = _np.einsum('...ji,...j,...jk->...ik',V.conj(),lmbda**2,V )
			elif return_rdm=='both':
				rdm_A = _np.einsum('...ij,...j,...kj->...ik',U,lmbda**2,U.conj() )
				rdm_B = _np.einsum('...ji,...j,...jk->...ik',V.conj(),lmbda**2,V )

		return (lmbda**2) + _np.finfo(lmbda.dtype).eps, rdm_A, rdm_B

	def _p_pure_sparse(self,state,sub_sys_A,return_rdm=None,sparse_diag=True,maxiter=None):

		"""
		# THE FOLLOWING LINES HAVE BEEN DEPRECATED

		if svds: # patchy sparse svd

			# calculate full H-space representation of state
			state=self.get_vec(state.T,sparse=True).T
			# reshape state according to sub_sys_A
			v=_lattice_reshape_sparse_pure(state,sub_sys_A,self._L,self._sps)
			#print(v.todense())
			n=min(v.shape)

			# perform SVD	
			if return_rdm is None:
				lmbda_SM = _spla.svds(v, k=n//2, which='SM',return_singular_vectors=False)
				lmbda_LM = _spla.svds(v, k=n//2+n%2, which='LM',return_singular_vectors=False)
				#_, lmbda_dense, _ = _npla.svd(v.todense(),full_matrices=False)
				# concatenate lower and upper part
				lmbda=_np.concatenate((lmbda_LM,lmbda_SM),axis=0)
				lmbda.sort()
				return lmbda[::-1]**2 + _np.finfo(lmbda.dtype).eps
			else:
				
				if return_rdm=='A':
					U_SM, lmbda_SM, V_SM = _spla.svds(v, k=n//2, which='SM',return_singular_vectors='u')
					U_LM, lmbda_LM, V_LM = _spla.svds(v, k=n//2+n%2, which='LM',return_singular_vectors='u')
					#ua,lmbdas,va = _npla.svd(v.todense())

					# concatenate lower and upper part
					lmbda=_np.concatenate((lmbda_LM,lmbda_SM),axis=0)
					arg = _np.argsort(lmbda)
					lmbda = lmbda[arg]**2

					U=_np.concatenate((U_LM,U_SM),axis=1)
					U=U[...,arg]
					#V=_np.concatenate((V_LM,V_SM[...,::-1,:]),axis=0)

					# check and orthogonalise VF in degenerate subspaces
					if _np.any( _np.diff(lmbda) < 1E3*_np.finfo(lmbda.dtype).eps):
						U,_ = _sla.qr(U, overwrite_a=True)
						#V,_ = _sla.qr(V.T, overwrite_a=True)
						#V = V.T
	
					# calculate reduced DM
					rdm_A = _np.einsum('...ij,...j,...kj->...ik',U,lmbda,U.conj() )

					return lmbda[::-1] + _np.finfo(lmbda.dtype).eps, rdm_A

				elif return_rdm=='B':
					U_SM, lmbda_SM, V_SM = _spla.svds(v, k=n//2, which='SM',return_singular_vectors='vh')
					U_LM, lmbda_LM, V_LM = _spla.svds(v, k=n//2+n%2, which='LM',return_singular_vectors='vh')
					#ua,lmbdas,va = _npla.svd(v.todense())

					# concatenate lower and upper part
					lmbda=_np.concatenate((lmbda_LM,lmbda_SM),axis=0)
					#U=_np.concatenate((U_LM,U_SM[...,::-1]),axis=1)
					V=_np.concatenate((V_LM,V_SM),axis=0)

					arg = _np.argsort(lmbda)
					lmbda = lmbda[arg]**2
					V = V[...,arg,:]

					

					# check and orthogonalise VF in degenerate subspaces
					if _np.any( _np.diff(lmbda) < 1E3*_np.finfo(lmbda.dtype).eps):
						#U,_ = _sla.qr(U, overwrite_a=True)
						V,_ = _sla.qr(V.T, overwrite_a=True)
						V = V.T
					
					# calculate reduced DM
					rdm_B = _np.einsum('...ji,...j,...jk->...ik',V.conj(),lmbda,V )

					return lmbda[::-1] + _np.finfo(lmbda.dtype).eps, rdm_B

				elif return_rdm=='both':
					U_SM, lmbda_SM, V_SM = _spla.svds(v, k=n//2, which='SM',return_singular_vectors=True)
					U_LM, lmbda_LM, V_LM = _spla.svds(v, k=n//2+n%2, which='LM',return_singular_vectors=True)

					# concatenate lower and upper part
					lmbda=_np.concatenate((lmbda_LM,lmbda_SM),axis=0)
					U=_np.concatenate((U_LM,U_SM),axis=1)
					V=_np.concatenate((V_LM,V_SM),axis=0)
					arg = _np.argsort(lmbda)
					lmbda = lmbda[arg]**2
					V = V[...,arg,:]
					U = U[...,arg]
					# check and orthogonalise VF in degenerate subspaces
					if _np.any( _np.diff(lmbda) < 1E3*_np.finfo(lmbda.dtype).eps):
						U,_ = _sla.qr(U, overwrite_a=True)
						V,_ = _sla.qr(V.T, overwrite_a=True)
						V = V.T
					# calculate reduced DM
					rdm_A = _np.einsum('...ij,...j,...kj->...ik',U,lmbda,U.conj() )
					rdm_B = _np.einsum('...ji,...j,...jk->...ik',V.conj(),lmbda,V )

					return lmbda[::-1] + _np.finfo(lmbda.dtype).eps, rdm_A, rdm_B
		"""

		partial_trace_args = dict(sub_sys_A=sub_sys_A,sparse=True,enforce_pure=True)

		Ns_A=self._basis_left.Ns
		Ns_B=self._basis_right.Ns

		rdm_A=None
		rdm_B=None

		if return_rdm is None:
			if Ns_A <= Ns_B:
				partial_trace_args["return_rdm"] = "A"
				rdm = tensor_basis.partial_trace(self,state,**partial_trace_args)
			else:
				partial_trace_args["return_rdm"] = "B"
				rdm = tensor_basis.partial_trace(self,state,**partial_trace_args)

		elif return_rdm=='A' and Ns_A <= Ns_B:
			partial_trace_args["return_rdm"] = "A"
			rdm_A = tensor_basis.partial_trace(self,state,**partial_trace_args)
			rdm = rdm_A

		elif return_rdm=='B' and Ns_B <= Ns_A:
			partial_trace_args["return_rdm"] = "B"
			rdm_B = tensor_basis.partial_trace(self,state,**partial_trace_args)
			rdm = rdm_B

		else:
			partial_trace_args["return_rdm"] = "both"
			rdm_A,rdm_B = tensor_basis.partial_trace(self,state,**partial_trace_args)

			if Ns_A <= Ns_B:
				rdm = rdm_A
			else:
				rdm = rdm_B

		if sparse_diag and rdm.shape[0] > 16:

			def get_p_patchy(rdm):
				n = rdm.shape[0]
				p_LM = eigsh(rdm,k=n//2+n%2,which="LM",maxiter=maxiter,return_eigenvectors=False) # get upper half
				p_SM = eigsh(rdm,k=n//2,which="SM",maxiter=maxiter,return_eigenvectors=False) # get lower half
				p = _np.concatenate((p_LM[::-1],p_SM)) + _np.finfo(p_LM.dtype).eps
				return p

			if _sp.issparse(rdm):
				p = get_p_patchy(rdm)
			else:
				p_gen = (get_p_patchy(dm) for dm in rdm[:])
				p = _np.stack(p_gen)

		else:
			if _sp.issparse(rdm):
				p = eigvalsh(rdm.todense())[::-1] + _np.finfo(rdm.dtype).eps
			else:
				p_gen = (eigvalsh(dm.todense())[::-1] + _np.finfo(dm.dtype).eps for dm in rdm[:])
				p = _np.stack(p_gen)

		return p,rdm_A,rdm_B
	
	def _p_mixed(self,state,sub_sys_A,return_rdm=None):
		"""
		This function calculates the eigenvalues of the reduced density matrix.
		It will first calculate the partial trace of the full density matrix and
		then diagonalizes it to get the eigenvalues. It will automatically choose
		the subsystem with the smaller hilbert space to do the diagonalization in order
		to reduce the calculation time but will only return the desired reduced density
		matrix. 
		"""

		state = state.transpose((2,0,1))

		Ns_left = self._basis_left.Ns
		Ns_right = self._basis_right.Ns

		rdm_A,p_A=None,None
		rdm_B,p_B=None,None
		
		if return_rdm=='both':
			rdm_A,rdm_B = _tensor_partial_trace_mixed(state,Ns_left,Ns_right,sub_sys_A,return_rdm="both")
			
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

		elif return_rdm=='A':
			rdm_A,rdm_B = _tensor_partial_trace_mixed(state,Ns_left,Ns_right,sub_sys_A,return_rdm="A")
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			
		elif return_rdm=='B':
			rdm_A,rdm_B = _tensor_partial_trace_mixed(state,Ns_left,Ns_right,sub_sys_A,return_rdm="B")
			p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

		else:
			rdm_A,rdm_B = _tensor_partial_trace_mixed(state,Ns_left,Ns_right,sub_sys_A,return_rdm="A")
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			
		return p_A, p_B, rdm_A, rdm_B

	def partial_trace(self,state,sub_sys_A="left",return_rdm=None,enforce_pure=False,sparse=False):
		if sub_sys_A is None:
			sub_sys_A = "left"

		if return_rdm not in set(["A","B","both",None]):
			raise ValueError("return_rdm must be: 'A','B','both' or None")

		if sub_sys_A not in set(["left","right","both",None]):
			raise ValueError("sub_sys_A must be 'left' or 'right' or 'both'.")

		if not hasattr(state,"shape"):
			state = _np.asanyarray(state)
			state = state.squeeze() # avoids artificial higher-dim reps of ndarray

		Ns_left = self._basis_left.Ns
		Ns_right = self._basis_right.Ns
		tensor_Ns =  Ns_left*Ns_right

		if state.shape[0] != tensor_Ns:
			raise ValueError("state shape {0} not compatible with Ns={1}".format(state.shape,tensor_Ns))

		if _sp.issparse(state) or sparse:
			if not _sp.issparse(state):
				state = _sp.csr_matrix(state)

			state = state.T
			if state.shape[0] == 1:
				# sparse_pure partial trace
				rdm_A,rdm_B = _tensor_partial_trace_sparse_pure(state,Ns_left,Ns_right,sub_sys_A,return_rdm=return_rdm)
			else:
				if state.shape[0] != state.shape[1] or enforce_pure:
					# vectorize sparse_pure partial trace 
					state = state.tocsr()
					try:
						state_gen = (_tensor_partial_trace_sparse_pure(state.getrow(i),Ns_left,Ns_right,sub_sys_A,return_rdm=return_rdm) for i in xrange(state.shape[0]))
					except NameError:
						state_gen = (_tensor_partial_trace_sparse_pure(state.getrow(i),Ns_left,Ns_right,sub_sys_A,return_rdm=return_rdm) for i in range(state.shape[0]))

					left,right = zip(*state_gen)

					rdm_A,rdm_B = _np.stack(left),_np.stack(right)

					if any(rdm is None for rdm in rdm_A):
						rdm_A = None

					if any(rdm is None for rdm in rdm_B):
						rdm_B = None
				else: 
					raise ValueError("Expecting a dense array for mixed states.")

		else:
			if state.ndim==1:
				rdm_A,rdm_B = _tensor_partial_trace_pure(state.T,Ns_left,Ns_right,sub_sys_A,return_rdm=return_rdm)

			elif state.ndim==2: 
				if state.shape[0]!=state.shape[1] or enforce_pure:
					rdm_A,rdm_B = _tensor_partial_trace_pure(state.T,Ns_left,Ns_right,sub_sys_A,return_rdm=return_rdm)

				else: 
					shape0 = state.shape
					state = state.reshape((1,)+shape0)					

					rdm_A,rdm_B = _tensor_partial_trace_mixed(state,Ns_left,Ns_right,sub_sys_A,return_rdm=return_rdm)

			elif state.ndim==3: #3D DM 
				rdm_A,rdm_B = _tensor_partial_trace_mixed(state,Ns_left,Ns_right,sub_sys_A,return_rdm=return_rdm)
			else:
				raise ValueError("state must have ndim < 4")

		if return_rdm == "A":
			return rdm_A
		elif return_rdm == "B":
			return rdm_B
		else:
			return rdm_A,rdm_B

	def ent_entropy(self,state,sub_sys_A="left",return_rdm=None,enforce_pure=False,return_rdm_EVs=False,sparse=False,alpha=1.0,sparse_diag=True,maxiter=None):
		"""
		This function calculates the entanglement entropy of subsystem A and the corresponding reduced 
		density matrix.

		RETURNS: dictionary with keys:

		'Sent_A': entanglement entropy of subystem A.
		'Sent_B': (optional) entanglement entropy of subystem B.
		'rdm_A': (optional) reduced density matrix of subsystem A
		'rdm_B': (optional) reduced density matrix of subsystem B
		'p_A': (optional) eigenvalues of reduced density matrix of subsystem A
		'p_B': (optional) eigenvalues of reduced density matrix of subsystem B

		--- arguments ---

		state: (required) the state of the quantum system. Can be a:

				-- pure state (default) [numpy array of shape (Ns,)].

				-- density matrix [numpy array of shape (Ns,Ns)].

				-- collection of states containing the states in the columns of state

		sub_sys_A: (optional) tuple or list to define the sites contained in subsystem A 
						[by python convention the first site of the chain is labelled j=0]. 
						Default is tuple(range(L//2)).

		return_rdm: (optional) flag to return the reduced density matrix. Default is 'None'.

				-- 'A': str, returns reduced DM of subsystem A

				-- 'B': str, returns reduced DM of subsystem B

				-- 'both': str, returns reduced DM of both subsystems A and B

		return_rdm_EVs: (optional) boolean to return eigenvalues of reduced DM. If `return_rdm` is specified,
						the eigenvalues of the corresponding DM are returned. If `return_rdm` is NOT specified, 
						the spectrum of `rdm_A` is terurned. Default is `False`.

		enforce_pure: (optional) boolean to determine if 'state' is a collection of pure states or
						a density matrix

		sparse: (optional) flag to enable usage of sparse linear algebra algorithms.

		alpha: (optional) Renyi alpha parameter. Default is '1.0'.

		"""
		if sub_sys_A is None:
			sub_sys_A = "left"

		if return_rdm not in set(["A","B","both",None]):
			raise ValueError("return_rdm must be: 'A','B','both' or None")

		if sub_sys_A not in set(["left","right","both"]):
			raise ValueError("sub_sys_A must be 'left' or 'right' or 'both'.")

		if not hasattr(state,"shape"):
			state = _np.asanyarray(state)
			state = state.squeeze() # avoids artificial higher-dim reps of ndarray

		tensor_Ns =  self._basis_left.Ns*self._basis_right.Ns

		if state.shape[0] != tensor_Ns:
			raise ValueError("state shape {0} not compatible with Ns={1}".format(state.shape,tensor_Ns))

		pure=True # set pure state parameter to True
		if _sp.issparse(state) or sparse:
			if not _sp.issparse(state):
				if state.ndim == 1:
					state = _sp.csr_matrix(state).T
				else:
					state = _sp.csr_matrix(state)

			if state.shape[1] == 1:
				p, rdm_A, rdm_B = self._p_pure_sparse(state,sub_sys_A,return_rdm=return_rdm,sparse_diag=sparse_diag,maxiter=maxiter)
			else:
				if state.shape[0]!=state.shape[1] or enforce_pure:
					p, rdm_A, rdm_B = self._p_pure_sparse(state,sub_sys_A,return_rdm=return_rdm)
				else: 
					raise ValueError("Expecting a dense array for mixed states.")
					
		else:
			if state.ndim==1:
				state = state.reshape((-1,1))
				p, rdm_A, rdm_B = self._p_pure(state,sub_sys_A,return_rdm=return_rdm)
			
			elif state.ndim==2: 
				if state.shape[0]!=state.shape[1] or enforce_pure:
					p, rdm_A, rdm_B = self._p_pure(state,sub_sys_A,return_rdm=return_rdm)
				else: # 2D mixed
					pure=False
					"""
					# check if DM's are positive definite
					try:
						_np.linalg.cholesky(state)
					except:
						raise ValueError("LinAlgError: (collection of) DM(s) not positive definite")
					# check oif trace of DM is unity
					if _np.any( abs(_np.trace(state) - 1.0 > 1E3*_np.finfo(state.dtype).eps)  ):
						raise ValueError("Expecting eigenvalues of DM to sum to unity!")
					"""
					shape0 = state.shape
					state = state.reshape(shape0+(1,))
					p_A, p_B, rdm_A, rdm_B = self._p_mixed(state,sub_sys_A,return_rdm=return_rdm)
				
			elif state.ndim==3: #3D DM 
				pure=False

				"""
				# check if DM's are positive definite
				try:
					_np.linalg.cholesky(state)
				except:
					raise ValueError("LinAlgError: (collection of) DM(s) not positive definite")

				# check oif trace of DM is unity
				if _np.any( abs(_np.trace(state, axis1=1,axis2=2) - 1.0 > 1E3*_np.finfo(state.dtype).eps)  ):
					raise ValueError("Expecting eigenvalues of DM to sum to unity!")
				"""
				p_A, p_B, rdm_A, rdm_B = self._p_mixed(state,sub_sys_A,return_rdm=return_rdm)

			else:
				raise ValueError("state must have ndim < 4")

		
		if pure:
			p_A, p_B = p, p

		Sent_A, Sent_B = None, None
		if alpha == 1.0:
			if p_A is not None:
				Sent_A = - (p_A * _np.log(p_A)).sum(axis=-1)
			if p_B is not None:
				Sent_B = - (p_B * _np.log(p_B)).sum(axis=-1)
		elif alpha >= 0.0:
			if p_A is not None:
				Sent_A = (_np.log(_np.power(p_A,alpha).sum(axis=-1))/(1.0-alpha))
			if p_B is not None:
				Sent_B = (_np.log(_np.power(p_B,alpha).sum(axis=-1))/(1.0-alpha))
		else:
			raise ValueError("alpha >= 0")

		# initiate variables
		variables = ["Sent_A"]
		
		if return_rdm_EVs:
			variables.append("p_A")

		if return_rdm == "A":
			variables.append("rdm_A")
			
		elif return_rdm == "B":
			variables.extend(["Sent_B","rdm_B"])
			if return_rdm_EVs:
				variables.append("p_B")
			
		elif return_rdm == "both":
			variables.extend(["rdm_A","Sent_B","rdm_B"])
			if return_rdm_EVs:
				variables.extend(["p_A","p_B"])
	

		# store variables to dictionar
		return_dict = {}
		for i in variables:
			if locals()[i] is not None:
				if sparse and 'rdm' in i:
					return_dict[i] = locals()[i] # don't squeeze sparse matrix
				else:
					return_dict[i] = _np.squeeze( locals()[i] )

		return return_dict

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

		op1_list = self._basis_left._expand_opstr(op1,num)
		op2_list = self._basis_right._expand_opstr(op2,num)

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


def _tensor_reshape_pure(psi,Ns_l,Ns_r):
	extra_dims = psi.shape[:-1]
	psi_v = psi.reshape(extra_dims+(Ns_l,Ns_r))

	return psi_v	


def _tensor_reshape_sparse_pure(psi,Ns_l,Ns_r):
	psi = psi.tocoo()
	# make shift way of reshaping array
	# j = j_l + Ns_r * j_l
	# j_l = j / Ns_r
	# j_r = j % Ns_r 
	psi._shape = (Ns_l,Ns_r)
	psi.row[:] = psi.col / Ns_r
	psi.col[:] = psi.col % Ns_r

	return psi.tocsr()


def _tensor_reshape_mixed(rho,Ns_l,Ns_r):
	extra_dims = rho.shape[:-2]
	psi_v = rho.reshape(extra_dims+(Ns_l,Ns_r,Ns_l,Ns_r))

	return psi_v


def _tensor_partial_trace_pure(psi,Ns_l,Ns_r,sub_sys_A="left",return_rdm="A"):
	psi_v = _tensor_reshape_pure(psi,Ns_l,Ns_r)

	if return_rdm == "A":
		return _np.squeeze(_np.einsum("...ij,...kj->...ik",psi_v,psi_v.conj())),None
	elif return_rdm == "B":
		return None,_np.squeeze(_np.einsum("...ji,...jk->...ik",psi_v,psi_v.conj()))
	elif return_rdm == "both":
		return _np.squeeze(_np.einsum("...ij,...kj->...ik",psi_v,psi_v.conj())),_np.squeeze(_np.einsum("...ji,...jk->...ik",psi_v,psi_v.conj()))


def _tensor_partial_trace_sparse_pure(psi,Ns_l,Ns_r,sub_sys_A="left",return_rdm="A"):
	psi = _tensor_reshape_sparse_pure(psi,Ns_l,Ns_r)

	if return_rdm == "A":
		return psi.dot(psi.H),None
	elif return_rdm == "B":
		return None,psi.H.dot(psi)
	elif return_rdm == "both":
		return psi.dot(psi.H),psi.H.dot(psi)


def _tensor_partial_trace_mixed(rho,Ns_l,Ns_r,sub_sys_A="left",return_rdm="A"):
	rho_v = _tensor_reshape_mixed(rho,Ns_l,Ns_r)

	if return_rdm == "A":
		return _np.squeeze(_np.einsum("...ijkj->...ik",rho_v)),None
	elif return_rdm == "B":
		return None,_np.squeeze(_np.einsum("...jijk->...ki",rho_v))
	elif return_rdm == "both":
		return _np.squeeze(_np.einsum("...ijkj->...ik",rho_v)),_np.squeeze(_np.einsum("...jijk->...ki",rho_v))

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
	#	v0=v0/n
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