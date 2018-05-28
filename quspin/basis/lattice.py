from .base import basis,MAXPRINT
from ._reshape_subsys import _lattice_partial_trace_pure,_lattice_reshape_pure
from ._reshape_subsys import _lattice_partial_trace_mixed,_lattice_reshape_mixed
from ._reshape_subsys import _lattice_partial_trace_sparse_pure,_lattice_reshape_sparse_pure
import numpy as _np
import scipy.sparse as _sp
from numpy.linalg import norm,eigvalsh,svd
from scipy.sparse.linalg import eigsh
import warnings

_dtypes={"f":_np.float32,"d":_np.float64,"F":_np.complex64,"D":_np.complex128}

class lattice_basis(basis):
	def __init__(self):
		self._Ns = 0
		self._basis = _np.asarray([])
		self._operators = "no operators for base."
		self._unique_me = True
		self._check_symm = None
		self._check_pcon = None
		if self.__class__.__name__ == 'lattice_basis':
			raise ValueError("This class is not intended"
							 " to be instantiated directly.")

	def __getitem__(self,key):
		return self._basis.__getitem__(key)

	def __iter__(self):
		return self._basis.__iter__()

	def index(self,s):
		"""Finds the index of user-defined Fock state in any lattice basis.

		Notes
		-----
		Particularly useful for defining initial Fock states through a unit vector in the direction specified
		by `index()`. 

		Parameters
		-----------
		s : {str, int}
			Defines the Fock state with number of particles (spins) per site in underlying lattice `basis`.

		Returns
		--------
		int
			Position of the Fock state in the lattice basis.

		Examples
		--------
		
		>>> i0 = index("111000") # pick state from basis set
		>>> print(basis)
		>>> print(i0)
		>>> psi = np.zeros(basis.Ns,dtype=np.float64)
		>>> psi[i0] = 1.0 # define state corresponding to the string "111000"

		"""
		if type(s) is int:
			pass
		elif type(s) is str:
			s = int(s,self.sps)
		else:
			raise ValueError("s must be integer or state")

		indx = _np.argwhere(self._basis == s)

		if len(indx) != 0:
			return _np.squeeze(indx)
		else:
			raise ValueError("s must be representive state in basis. ")

	def _partial_trace(self,state,sub_sys_A=None,subsys_ordering=True,return_rdm="A",enforce_pure=False,sparse=False):
		"""Calculates reduced density matrix, through a partial trace of a quantum state in a lattice `basis`.

		Parameters
		-----------
		state : obj
			State of the quantum system. Can be either one of:

				* numpy.ndarray [shape (Ns,)]: pure state (default).
				* numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
				* dict('V_states',V_states) [shape (Ns,Nvecs)]: collection of `Nvecs` states stored in the columns of `V_states`.
		sub_sys_A : tuple/list, optional
			Defines the sites contained in subsystem A [by python convention the first site of the chain is labelled j=0].
			Default is `tuple(range(N//2))` with `N` the number of lattice sites.
		return_rdm : str, optional
			Toggles returning the reduced DM. Can be tierh one of:

				* "A": returns reduced DM of subsystem A.
				* "B": returns reduced DM of subsystem B.
				* "both": returns reduced DM of both A and B subsystems.
		subsys_ordering : bool, optional
			Whether or not to reorder the sites in `sub_sys_A` in ascending order. Default is `True`.
		enforce_pure : bool, optional
			Whether or not to assume `state` is a collection of pure states or a mixed density matrix, if
			it is a square array. Default is `False`.
		sparse : bool, optional
			Whether or not to return a sparse DM. Default is `False`.

		Returns
		--------
		numpy.ndarray
			Density matrix associated with `state`. Depends on optional arguments.

		Examples
		--------

		>>> partial_trace(state,sub_sys_A=tuple(range(basis.N//2),return_rdm="A",enforce_pure=False,sparse=False,subsys_ordering=True)

		"""

		if sub_sys_A is None:
			sub_sys_A = tuple(range(self.N//2))
		elif len(sub_sys_A)==self.N:
			raise ValueError("Size of subsystem must be strictly smaller than total system size N!")


		N_A = len(sub_sys_A)
		N_B = self.N - N_A

		if sub_sys_A is None:
			sub_sys_A = tuple(range(self.N//2))

		sub_sys_A = tuple(sub_sys_A)

		if any(not _np.issubdtype(type(s),_np.integer) for s in sub_sys_A):
			raise ValueError("sub_sys_A must iterable of integers with values in {0,...,N-1}!")

		if any(s < 0 or s > self.N for s in sub_sys_A):
			raise ValueError("sub_sys_A must iterable of integers with values in {0,...,N-1}")

		doubles = tuple(s for s in sub_sys_A if sub_sys_A.count(s) > 1)
		if len(doubles) > 0:
			raise ValueError("sub_sys_A contains repeated values: {}".format(doubles))

		if return_rdm not in set(["A","B","both"]):
			raise ValueError("return_rdm must be: 'A','B','both' or None")

		if subsys_ordering:
			sub_sys_A = sorted(sub_sys_A)

		sps = self.sps
		N = self.N

		if not hasattr(state,"shape"):
			state = _np.asanyarray(state)
			state = state.squeeze() # avoids artificial higher-dim reps of ndarray


		if state.shape[0] != self.Ns:
			raise ValueError("state shape {0} not compatible with Ns={1}".format(state.shape,self._Ns))

		if _sp.issparse(state) or sparse:
			state=self.get_vec(state,sparse=True).T
			
			if state.shape[0] == 1:
				# sparse_pure partial trace
				rdm_A,rdm_B = _lattice_partial_trace_sparse_pure(state,sub_sys_A,N,sps,return_rdm=return_rdm)
			else:
				if state.shape[0]!=state.shape[1] or enforce_pure:
					# vectorize sparse_pure partial trace 
					state = state.tocsr()
					try:
						state_gen = (_lattice_partial_trace_sparse_pure(state.getrow(i),sub_sys_A,N,sps,return_rdm=return_rdm) for i in xrange(state.shape[0]))
					except NameError:
						state_gen = (_lattice_partial_trace_sparse_pure(state.getrow(i),sub_sys_A,N,sps,return_rdm=return_rdm) for i in range(state.shape[0]))

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
				# calculate full H-space representation of state
				state=self.get_vec(state,sparse=False)
				rdm_A,rdm_B = _lattice_partial_trace_pure(state.T,sub_sys_A,N,sps,return_rdm=return_rdm)

			elif state.ndim==2: 
				if state.shape[0]!=state.shape[1] or enforce_pure:
					# calculate full H-space representation of state
					state=self.get_vec(state,sparse=False)
					rdm_A,rdm_B = _lattice_partial_trace_pure(state.T,sub_sys_A,N,sps,return_rdm=return_rdm)

				else: 
					proj = self.get_proj(_dtypes[state.dtype.char])
					proj_state = proj*state*proj.H

					shape0 = proj_state.shape
					proj_state = proj_state.reshape((1,)+shape0)					

					rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,N,sps,return_rdm=return_rdm)

			elif state.ndim==3: #3D DM 
				proj = self.get_proj(_dtypes[state.dtype.char])
				state = state.transpose((2,0,1))
				
				Ns_full = proj.shape[0]
				n_states = state.shape[0]
				
				gen = (proj*s*proj.H for s in state[:])

				proj_state = _np.zeros((n_states,Ns_full,Ns_full),dtype=_dtypes[state.dtype.char])
				
				for i,s in enumerate(gen):
					proj_state[i,...] += s[...]

				rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,N,sps,return_rdm=return_rdm)
			else:
				raise ValueError("state must have ndim < 4")

		if return_rdm == "A":
			return rdm_A
		elif return_rdm == "B":
			return rdm_B
		else:
			return rdm_A,rdm_B

	def _ent_entropy(self,state,sub_sys_A=None,density=True,subsys_ordering=True,return_rdm=None,enforce_pure=False,return_rdm_EVs=False,sparse=False,alpha=1.0,sparse_diag=True,maxiter=None):
		"""Calculates entanglement entropy of subsystem A and the corresponding reduced density matrix

		.. math::
			S_\\mathrm{ent}(\\alpha) = \\frac{1}{N}\\frac{1}{1-\\alpha}\\log \\mathrm{tr}_{A} \\left( \\mathrm{tr}_{A^c} \\vert\\psi\\rangle\\langle\\psi\\vert \\right)^\\alpha 

		where the normalization :math:`N` can be switched on and off using the optional argument `density`.
			
		**Note:** The logarithm used is the natural logarithm (base e).

		Notes
		-----
		Algorithm is based on both partial tracing and sigular value decomposition (SVD), optimised for speed.

		Parameters
		-----------
		state : obj
			State of the quantum system. Can be either one of:

				* numpy.ndarray [shape (Ns,)]: pure state (default).
				* numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
				* dict('V_states',V_states) [shape (Ns,Nvecs)]: collection of `Nvecs` states stored in the columns of `V_states`.
		sub_sys_A : tuple/list, optional
			Defines the sites contained in subsystem A [by python convention the first site of the chain is labelled j=0].
			Default is `tuple(range(N//2))` with `N` the number of lattice sites.
		density : bool, optional
			Toggles whether to return entanglement entropy normalized by the number of sites in the subsystem.
		return_rdm : str, optional
			Toggles returning the reduced DM. Can be tierh one of:

				* "A": returns reduced DM of subsystem A.
				* "B": returns reduced DM of subsystem B.
				* "both": returns reduced DM of both A and B subsystems.
		enforce_pure : bool, optional
			Whether or not to assume `state` is a collection of pure states or a mixed density matrix, if
			it is a square array. Default is `False`.
		subsys_ordering : bool, optional
			Whether or not to reorder the sites in `sub_sys_A` in ascending order. Default is `True`.
		sparse : bool, optional
			Whether or not to return a sparse DM. Default is `False`.
		return_rdm_EVs : bool, optional 
			Whether or not to return the eigenvalues of rthe educed DM. If `return_rdm` is specified,
			the eigenvalues of the corresponding DM are returned. If `return_rdm` is NOT specified, 
			the spectrum of `rdm_A` is returned by default. Default is `False`.
		alpha : float, optional
			Renyi :math:`\\alpha` parameter for the entanglement entropy. Default is :math:`\\alpha=1`.

			
		sparse_diag : bool, optional
			When `sparse=True`, this flag enforces the use of
			`scipy.sparse.linalg.eigsh() <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html>`_
			to calculate the eigenvaues of the reduced DM.
		maxiter : int, optional
			Specifies the number of iterations for Lanczos diagonalisation. Look up documentation for 
			`scipy.sparse.linalg.eigsh() <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html>`_.

		Returns
		--------
		dict
			Dictionary with following keys, depending on input parameters:
				* "Sent_A": entanglement entropy of subsystem A (default).
				* "Sent_B": entanglement entropy of subsystem B.
				* "p_A": singular values of reduced DM of subsystem A (default).
				* "p_B": singular values of reduced DM of subsystem B.
				* "rdm_A": reduced DM of subsystem A.
				* "rdm_B": reduced DM of subsystem B.

		Examples
		--------

		>>> ent_entropy(state,sub_sys_A=[0,3,4,7],return_rdm="A",enforce_pure=False,return_rdm_EVs=False,
		>>>				sparse=False,alpha=1.0,sparse_diag=True,subsys_ordering=True)

		"""
		if sub_sys_A is None:
			sub_sys_A = list(range(self.N//2))
		else:
			sub_sys_A = list(sub_sys_A)
	
		if len(sub_sys_A)>=self.N:
			raise ValueError("Size of subsystem must be strictly smaller than total system size N!")

		N_A = len(sub_sys_A)
		N_B = self.N - N_A

		if any(not _np.issubdtype(type(s),_np.integer) for s in sub_sys_A):
			raise ValueError("sub_sys_A must iterable of integers with values in {0,...,N-1}!")

		if any(s < 0 or s > self.N for s in sub_sys_A):
			raise ValueError("sub_sys_A must iterable of integers with values in {0,...,N-1}")

		doubles = tuple(s for s in set(sub_sys_A) if sub_sys_A.count(s) > 1)
		if len(doubles) > 0:
			raise ValueError("sub_sys_A contains repeated values: {}".format(doubles))

		if return_rdm not in set(["A","B","both",None]):
			raise ValueError("return_rdm must be: 'A','B','both' or None")

		if subsys_ordering:
			sub_sys_A = sorted(sub_sys_A)

		sps = self.sps
		N = self.N

		if not hasattr(state,"shape"):
			state = _np.asanyarray(state)
			state = state.squeeze() # avoids artificial higher-dim reps of ndarray


		if state.shape[0] != self.Ns:
			raise ValueError("state shape {0} not compatible with Ns={1}".format(state.shape,self._Ns))

		

		pure=True # set pure state parameter to True
		if _sp.issparse(state) or sparse:
			if state.ndim == 1:
				state = state.reshape((-1,1))

			sparse=True # set sparse flag to True
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
				Sent_A = - _np.nansum((p_A * _np.log(p_A)),axis=-1)
				if density: Sent_A /= N_A
			if p_B is not None:
				Sent_B = - _np.nansum((p_B * _np.log(p_B)),axis=-1)
				if density: Sent_B /= N_B
		elif alpha >= 0.0:
			if p_A is not None:
				Sent_A = _np.log(_np.nansum(_np.power(p_A,alpha),axis=-1))/(1.0-alpha)
				if density: Sent_A /= N_A
			if p_B is not None:
				Sent_B = _np.log(_np.nansum(_np.power(p_B,alpha),axis=-1))/(1.0-alpha)
				if density: Sent_B /= N_B
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



	##### private methods

	def _p_pure(self,state,sub_sys_A,return_rdm=None):
		
		# calculate full H-space representation of state
		state=self.get_vec(state,sparse=False)
		# put states in rows
		state=state.T
		# reshape state according to sub_sys_A
		v=_lattice_reshape_pure(state,sub_sys_A,self.N,self._sps)
		
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


		return lmbda**2 + _np.finfo(lmbda.dtype).eps, rdm_A, rdm_B

	def _p_pure_sparse(self,state,sub_sys_A,return_rdm=None,sparse_diag=True,maxiter=None):

		partial_trace_args = dict(sub_sys_A=sub_sys_A,sparse=True,enforce_pure=True)

		N_A=len(sub_sys_A)
		N_B=self.N-N_A

		rdm_A=None
		rdm_B=None

		if return_rdm is None:
			if N_A <= N_B:
				partial_trace_args["return_rdm"] = "A"
				rdm = self._partial_trace(state,**partial_trace_args)
			else:
				partial_trace_args["return_rdm"] = "B"
				rdm = self._partial_trace(state,**partial_trace_args)

		elif return_rdm=='A' and N_A <= N_B:
			partial_trace_args["return_rdm"] = "A"
			rdm_A = self._partial_trace(state,**partial_trace_args)
			rdm = rdm_A

		elif return_rdm=='B' and N_B <= N_A:
			partial_trace_args["return_rdm"] = "B"
			rdm_B = self._partial_trace(state,**partial_trace_args)
			rdm = rdm_B

		else:
			partial_trace_args["return_rdm"] = "both"
			rdm_A,rdm_B = self._partial_trace(state,**partial_trace_args)

			if N_A < N_B:
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
				p = p.reshape((1,-1))
			else:
				p_gen = (get_p_patchy(dm) for dm in rdm[:])
				p = _np.stack(p_gen)

		else:
			if _sp.issparse(rdm):
				p = eigvalsh(rdm.todense())[::-1] + _np.finfo(rdm.dtype).eps
				p = p.reshape((1,-1))
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
		N = self.N
		sps = self.sps

		N_A = len(sub_sys_A)
		N_B = N - N_A

		proj = self.get_proj(_dtypes[state.dtype.char])
		state = state.transpose((2,0,1))

		Ns_full = proj.shape[0]
		n_states = state.shape[0]
		
		gen = (proj*s*proj.H for s in state[:])

		proj_state = _np.zeros((n_states,Ns_full,Ns_full),dtype=_dtypes[state.dtype.char])
		
		for i,s in enumerate(gen):
			proj_state[i,...] += s[...]	

		rdm_A,p_A=None,None
		rdm_B,p_B=None,None
		
		if return_rdm=='both':
			rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,N,sps,return_rdm="both")
			
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

		elif return_rdm=='A':
			rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,N,sps,return_rdm="A")
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			
		elif return_rdm=='B':
			rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,N,sps,return_rdm="B")
			p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

		else:
			rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,N,sps,return_rdm="A")
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			
			
		return p_A, p_B, rdm_A, rdm_B

	def _get__str__(self):

		def get_state(b):
			n_space = len(str(self.sps))
			if self.N <= 64:
				bits = (int(b)//int(self.sps**(self.N-i-1))%self.sps for i in range(self.N))
				state = "|"+(" ".join(("{:"+str(n_space)+"d}").format(bit) for bit in bits))+">"
			else:
				left_bits = (int(b)//int(self.sps**(self.N-i-1))%self.sps for i in range(32))
				right_bits = (int(b)//int(self.sps**(self.N-i-1))%self.sps for i in range(self.N-32,self.N,1))

				str_list = [("{:"+str(n_space)+"d}").format(bit) for bit in left_bits]
				str_list.append("...")
				str_list.extend(("{:"+str(n_space)+"d}").format(bit) for bit in right_bits)
				state = "|"+(" ".join(str_list))+">"

			return state


		temp1 = "     {0:"+str(len(str(self.Ns)))+"d}.  "
		if self._Ns > MAXPRINT:
			half = MAXPRINT // 2
			str_list = [(temp1.format(i))+get_state(b) for i,b in zip(range(half),self._basis[:half])]
			str_list.extend([(temp1.format(i))+get_state(b) for i,b in zip(range(self._Ns-half,self._Ns,1),self._basis[-half:])])
		else:
			str_list = [(temp1.format(i))+get_state(b) for i,b in enumerate(self._basis)]

		return tuple(str_list)



