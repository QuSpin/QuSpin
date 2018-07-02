from __future__ import print_function, division

from ..basis import spin_basis_1d as _default_basis
from ..basis import isbasis as _isbasis

from ._make_hamiltonian import make_static

from . import hamiltonian_core

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

import functools
from six import iteritems,itervalues,viewkeys

__all__=["quantum_operator","isquantum_operator"]
		
# function used to create Linearquantum_operator with fixed set of parameters. 
def _quantum_operator_dot(op,pars,v):
	return op.dot(v,pars=pars,check=False)

class quantum_operator(object):
	"""Constructs parameter-dependent (hermitian and nonhermitian) operators.

		The `quantum_operator` class maps quantum operators to keys of a dictionary. When calling various methods
		of `quantum_operator`, it allows one to 'dynamically' specify the pre-factors of these operators.

		Examples
		---------

		It is often required to be able to handle a parameter-dependent Hamiltonian :math:`H(\\lambda)=H_1 + \\lambda H_2`, e.g.

		.. math::
			H_1=\sum_j J_{zz}S^z_jS^z_{j+1} + h_xS^x_j, \\qquad H_2=\\sum_j S^z_j

		The following code snippet shows how to use the `quantum_operator` class to vary the parameter :math:`\\lambda`
		without having to re-build the Hamiltonian every time.

		.. literalinclude:: ../../doc_examples/quantum_operator-example.py
			:linenos:
			:language: python
			:lines: 7-

	"""
	def __init__(self,input_dict,N=None,basis=None,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**basis_args):
		"""Intializes the `quantum_operator` object (parameter dependent quantum quantum_operators).

		Parameters
		-----------
		input_dict : dict
			The `values` of this dictionary contain quantum_operator lists, in the same format as the `static_list` 
			argument of the `hamiltonian` class.

			The `keys` of this dictionary correspond to the parameter values, e.g. :math:`J_{zz},h_x`, and are 
			used to specify the coupling strength during calls of the `quantum_operator` class methods.

			>>> # use "Jzz" and "hx" keys to specify the zz and x coupling strengths, respectively
			>>> input_dict = { "Jzz": [["zz",Jzz_bonds]], "hx" : [["x" ,hx_site ]] } 

		N : int, optional
			Number of lattice sites for the `hamiltonian` object.
		dtype : 'type'
			Data type (e.g. numpy.float64) to construct the quantum_operator with.
		shape : tuple, optional
			Shape to create the `hamiltonian` object with. Default is `shape = None`.
		copy: bool, optional
			If set to `True`, this option creates a copy of the input array. 
		check_symm : bool, optional 
			Enable/Disable symmetry check on `static_list` and `dynamic_list`.
		check_herm : bool, optional
			Enable/Disable hermiticity check on `static_list` and `dynamic_list`.
		check_pcon : bool, optional
			Enable/Disable particle conservation check on `static_list` and `dynamic_list`.
		kw_args : dict
			Optional additional arguments to pass to the `basis` class, if not already using a `basis` object
			to create the quantum_operator.		
			
		"""
		self._is_dense = False
		self._ndim = 2
		self._basis = basis



		if not (dtype in hamiltonian_core.supported_dtypes):
			raise TypeError('hamiltonian does not support type: '+str(dtype))
		else:
			self._dtype=dtype
		
		opstr_dict = {}
		other_dict = {}
		self._quantum_operator = {}
		if isinstance(input_dict,dict):
			for key,op in input_dict.items():
				if type(key) is not str:
					raise ValueError("keys to input_dict must be strings.")
					
				if type(op) not in [list,tuple]:
					raise ValueError("input_dict must contain values which are lists/tuples.")
				opstr_list = []
				other_list = []
				for ele in op:
					if hamiltonian_core._check_static(ele):
						opstr_list.append(ele)
					else:
						other_list.append(ele)

				if opstr_list:
					opstr_dict[key] = opstr_list
				if other_list:
					other_dict[key] = other_list
		elif isinstance(input_dict,quantum_operator):
			other_dict = {key:[value] for key,value in input_dict._quantum_operator_dict.items()} 
		else:
			raise ValueError("input_dict must be dictionary or another quantum_operator quantum_operators")
			


		if opstr_dict:
			# check if user input basis

			if basis is not None:
				if len(basis_args) > 0:
					wrong_keys = set(basis_args.keys())
					temp = ", ".join(["{}" for key in wrong_keys])
					raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))

			# if not
			if basis is None: 
				if N is None: # if L is missing 
					raise Exception('if opstrs in use, argument N needed for basis class')

				if type(N) is not int: # if L is not int
					raise TypeError('argument N must be integer')

				basis=_default_basis(N,**basis_args)

			elif not _isbasis(basis):
				raise TypeError('expecting instance of basis class for argument: basis')


			static_opstr_list = []
			for key,opstr_list in iteritems(opstr_dict):
				static_opstr_list.extend(opstr_list)

			if check_herm:
				basis.check_hermitian(static_opstr_list, [])

			if check_symm:
				basis.check_symm(static_opstr_list,[])

			if check_pcon:
				basis.check_pcon(static_opstr_list,[])

			self._shape=(basis.Ns,basis.Ns)

			for key,opstr_list in iteritems(opstr_dict):
				self._quantum_operator[key]=make_static(basis,opstr_list,dtype)

		if other_dict:
			if not hasattr(self,"_shape"):
				found = False
				if shape is None: # if no shape argument found, search to see if the inputs have shapes.
					for key,O_list in iteritems(other_dict):
						for O in O_list:
							try: # take the first shape found
								shape = O.shape
								found = True
								break
							except AttributeError: 
								continue
				else:
					found = True

				if not found:
					raise ValueError('no dictionary entries have shape attribute.')
				if shape[0] != shape[1]:
					raise ValueError('quantum_operator must be square matrix')

				self._shape=shape



			for key,O_list in iteritems(other_dict):
				for i,O in enumerate(O_list):
					if _sp.issparse(O):
						self._mat_checks(O)
						if i == 0:
							self._quantum_operator[key] = O
						else:
							try:
								self._quantum_operator[key] += O
							except NotImplementedError:
								self._quantum_operator[key] = self._quantum_operator[key] + O

					elif O.__class__ is _np.ndarray:
						self._mat_checks(O)
						self._is_dense=True
						if i == 0:
							self._quantum_operator[key] = O
						else:
							try:
								self._quantum_operator[key] += O
							except NotImplementedError:
								self._quantum_operator[key] = self._quantum_operator[key] + O

					elif O.__class__ is _np.matrix:
						self._mat_checks(O)
						self._is_dense=True
						if i == 0:
							self._quantum_operator[key] = O
						else:
							try:
								self._quantum_operator[key] += O
							except NotImplementedError:
								self._quantum_operator[key] = self._quantum_operator[key] + O

					else:
						O = _np.asanyarray(O)
						self._mat_checks(O)
						if i == 0:
							self._quantum_operator[key] = O
						else:
							try:
								self._quantum_operator[key] += O
							except NotImplementedError:
								self._quantum_operator[key] = self._quantum_operator[key] + O

					

		else:
			if not hasattr(self,"_shape"):
				if shape is None:
					# check if user input basis
					basis=basis_args.get('basis')	

					# if not
					if basis is None: 
						if N is None: # if N is missing 
							raise Exception("argument N or shape needed to create empty hamiltonian")

						if type(N) is not int: # if L is not int
							raise TypeError('argument N must be integer')

						basis=_default_basis(N,**basis_args)

					elif not _isbasis(basis):
						raise TypeError('expecting instance of basis class for argument: basis')

					shape = (basis.Ns,basis.Ns)

				else:
					basis=basis_args.get('basis')	
					if not basis is None: 
						raise ValueError("empty hamiltonian only accepts basis or shape, not both")

			
				if len(shape) != 2:
					raise ValueError('expecting ndim = 2')
				if shape[0] != shape[1]:
					raise ValueError('hamiltonian must be square matrix')

				self._shape=shape

		if basis is not None:
			self._basis = basis

		self._Ns = self._shape[0]


	@property
	def basis(self):
		""":obj:`basis`: basis used to build the `hamiltonian` object. Defaults to `None` if quantum_operator has 
		no basis (i.e. was created externally and passed as a precalculated array).

		"""
		if self._basis is not None:
			return self._basis
		else:
			raise AttributeError("object has no attribute 'basis'")

	@property
	def ndim(self):
		"""int: number of dimensions, always equal to 2. """
		return self._ndim
	
	@property
	def Ns(self):
		"""int: number of states in the (symmetry-reduced) Hilbert space spanned by `basis`."""
		return self._Ns

	@property
	def get_shape(self):
		"""tuple: shape of the `quantum_operator` object, always equal to `(Ns,Ns)`."""
		return self._shape

	@property
	def is_dense(self):
		"""bool: `True` if the quantum_operator contains a dense matrix as a componnent of either 
		the static or dynamic lists.

		"""
		return self._is_dense

	@property
	def dtype(self):
		"""type: data type of `quantum_operator` object."""
		return _np.dtype(self._dtype).name

	@property
	def T(self):
		""":obj:`quantum_operator`: transposes the operator matrix: :math:`H_{ij}\\mapsto H_{ji}`."""
		return self.transpose()

	@property
	def H(self):
		""":obj:`quantum_operator`: transposes and conjugates the operator matrix: :math:`H_{ij}\\mapsto H_{ji}^*`."""
		return self.getH()




	### state manipulation/observable routines

	def matvec(self,x):
		"""Matrix-vector multiplication.

		Performs the operation y=A*x where A is an MxN linear operator and x is a column vector or 1-d array.

		Notes
		-----
		This matvec wraps the user-specified matvec routine or overridden _matvec method to ensure that y has the correct shape and type.
	
		Parameters
		----------
		x : {matrix, ndarray}
			An array with shape (N,) or (N,1).

		Returns
		-------
		y : {matrix, ndarray}
			A matrix or ndarray with shape (M,) or (M,1) depending on the type and shape of the x argument.

		"""

		return self.dot(x)

	def rmatvec(self,x):
		"""Adjoint matrix-vector multiplication.

		Performs the operation y = A^H * x where A is an MxN linear operator and x is a column vector or 1-d array.

		Notes
		-----
		This rmatvec wraps the user-specified rmatvec routine or overridden _rmatvec method to ensure that y has the correct shape and type.

		Parameters
		----------
		x : {matrix, ndarray}
			An array with shape (M,) or (M,1).
		
		Returns
		-------
		y : {matrix, ndarray}
			A matrix or ndarray with shape (N,) or (N,1) depending on the type and shape of the x argument.

		"""
		return self.H.dot(x)

	def matmat(self,X):
		"""Matrix-matrix multiplication.

		Performs the operation y=A*X where A is an MxN linear operator and X dense N*K matrix or ndarray.

		Notes
		-----
		This matmat wraps any user-specified matmat routine or overridden _matmat method to ensure that y has the correct type.

		Parameters
		----------
		X : {matrix, ndarray}
			An array with shape (N,K).

		Returns
		-------
		Y : {matrix, ndarray}
			A matrix or ndarray with shape (M,K) depending on the type of the X argument.

		"""
		return self.dot(X)

	def dot(self,V,pars={},check=True):
		"""Matrix-vector multiplication of `quantum_operator` quantum_operator for parameters `pars`, with state `V`.

		.. math::
			H(t=\\lambda)|V\\rangle

		Notes
		-----
		It is faster to multiply the individual (static, dynamic) parts of the Hamiltonian first, then add all those 
		vectors together.

		Parameters
		-----------
		V : numpy.ndarray
			Vector (quantums tate) to multiply the `quantum_operator` quantum_operator with.
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		check : bool, optional
			Whether or not to do checks for shape compatibility.
			

		Returns
		--------
		numpy.ndarray
			Vector corresponding to the `quantum_operator` quantum_operator applied on the state `V`.

		Examples
		---------
		>>> B = H.dot(A,pars=pars,check=True)

		corresponds to :math:`B = HA`. 
	
		"""

		
		if self.Ns <= 0:
			return _np.asarray([])

		pars = self._check_scalar_pars(pars)


		if not check:
			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)
			for key,J in pars.items():
				V_dot += J*self._quantum_operator[key].dot(V)
			return V_dot

		if V.ndim > 2:
			raise ValueError("Expecting V.ndim < 3.")




		if V.__class__ is _np.ndarray:
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)
			for key,J in pars.items():
				V_dot += J*self._quantum_operator[key].dot(V)


		elif _sp.issparse(V):
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)	
			for key,J in pars.items():
				V_dot += J*self._quantum_operator[key].dot(V)



		elif V.__class__ is _np.matrix:
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)
			for key,J in pars.items():
				V_dot += J*self._quantum_operator[key].dot(V)

		else:
			V = _np.asanyarray(V)
			if V.ndim not in [1,2]:
				raise ValueError("Expecting 1 or 2 dimensional array")

			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)
			for key,J in pars.items():
				V_dot += J*self._quantum_operator[key].dot(V)


		return V_dot

	def rdot(self,V,pars={},check=False):
		"""Vector-matrix multiplication of `quantum_operator` quantum_operator for parameters `pars`, with state `V`.

		.. math::
			\\langle V]H(t=\\lambda)

		
		Parameters
		-----------
		V : numpy.ndarray
			Vector (quantums tate) to multiply the `quantum_operator` quantum_operator with.
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		check : bool, optional
			Whether or not to do checks for shape compatibility.
			

		Returns
		--------
		numpy.ndarray
			Vector corresponding to the `quantum_operator` quantum_operator applied on the state `V`.

		Examples
		---------
		>>> B = H.dot(A,pars=pars,check=True)

		corresponds to :math:`B = AH`. 
	
		"""
		try:
			V = V.transpose()
		except AttributeError:
			V = _np.asanyarray(V)
			V = V.transpose()
		return (self.transpose().dot(V,pars=pars,check=check)).transpose()

	def matrix_ele(self,Vl,Vr,pars={},diagonal=False,check=True):
		"""Calculates matrix element of `quantum_operator` quantum_operator for parameters `pars` in states `Vl` and `Vr`.

		.. math::
			\\langle V_l|H(\\lambda)|V_r\\rangle

		Notes
		-----
		Taking the conjugate or transpose of the state `Vl` is done automatically.  

		Parameters
		-----------
		Vl : numpy.ndarray
			Vector(s)/state(s) to multiple with on left side.
		Vl : numpy.ndarray
			Vector(s)/state(s) to multiple with on right side.
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		diagonal : bool, optional
			When set to `True`, returs only diagonal part of expectation value. Default is `diagonal = False`.
		check : bool,

		Returns
		--------
		float
			Matrix element of `quantum_operator` quantum_operator between the states `Vl` and `Vr`.

		Examples
		---------
		>>> H_lr = H.expt_value(Vl,Vr,pars=pars,diagonal=False,check=True)

		corresponds to :math:`H_{lr} = \\langle V_l|H(\\lambda=0)|V_r\\rangle`. 

		"""
		if self.Ns <= 0:
			return _np.array([])

		pars = self._check_scalar_pars(pars)

		Vr=self.dot(Vr,pars=pars,check=check)

		if not check:
			if diagonal:
				return _np.einsum("ij,ij->j",Vl.conj(),Vr)
			else:
				return Vl.T.conj().dot(Vr)
 

		if Vl.__class__ is _np.ndarray:
			if Vl.ndim == 1:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V1.shape,self._shape))

				return Vl.conj().dot(Vr)
			elif Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V1.shape,self._shape))

				if diagonal:
					return _np.einsum("ij,ij->j",Vl.conj(),Vr)
				else:
					return Vl.T.conj().dot(Vr)
			else:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V1.shape,self._shape))

		elif Vl.__class__ is _np.matrix:
			if Vl.ndim == 1:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V1.shape,self._shape))

				return Vl.conj().dot(Vr)
			elif Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V1.shape,self._shape))

				if diagonal:
					return _np.einsum("ij,ij->j",Vl.conj(),Vr)
				else:
					return Vl.H.dot(Vr)
			else:
				raise ValueError('Expecting Vl to have ndim < 3')

		elif _sm.issparse(Vl):
			if Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError('dimension mismatch')
				if diagonal:
					return Vl.H.dot(Vr).diagonal()
				else:
					return Vl.H.dot(Vr)
			else:
				raise ValueError('Expecting Vl to have ndim < 3')

		else:
			Vl = _np.asanyarray(Vl)
			if Vl.ndim == 1:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V1.shape,self._shape))
				if diagonal:
					return _np.einsum("ij,ij->j",Vl.conj(),Vr)
				else:
					return Vl.conj().dot(Vr)
			elif Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V1.shape,self._shape))

				return Vl.T.conj().dot(Vr)
			else:
				raise ValueError('Expecting Vl to have ndim < 3')

	### Diagonalisation routines

	def eigsh(self,pars={},**eigsh_args):
		"""Computes SOME eigenvalues and eigenvectors of hermitian `quantum_operator` quantum_operator using SPARSE hermitian methods.

		This function method solves for eigenvalues and eigenvectors, but can only solve for a few of them accurately.
		It calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html>`_, which is a wrapper for ARPACK.

		Notes
		-----
		Assumes the quantum_operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		eigsh_args : 
			For all additional arguments see documentation of `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html>`_.
			
		Returns
		--------
		tuple
			Tuple containing the `(eigenvalues, eigenvectors)` of the `quantum_operator` quantum_operator.

		Examples
		---------
		>>> eigenvalues,eigenvectors = H.eigsh(pars=pars,**eigsh_args)

		"""
		if self.Ns == 0:
			return _np.array([]),_np.array([[]])

		return _sla.eigsh(self.tocsr(pars),**eigsh_args)

	def eigh(self,pars={},**eigh_args):
		"""Computes COMPLETE eigensystem of hermitian `quantum_operator` quantum_operator using DENSE hermitian methods.

		This function method solves for all eigenvalues and eigenvectors. It calls 
		`numpy.linalg.eigh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigh.html>`_, 
		and uses wrapped LAPACK functions which are contained in the module py_lapack.

		Notes
		-----
		Assumes the quantum_operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		eigh_args : 
			For all additional arguments see documentation of `numpy.linalg.eigh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigh.html>`_.
			
		Returns
		--------
		tuple
			Tuple containing the `(eigenvalues, eigenvectors)` of the `quantum_operator` quantum_operator.

		Examples
		---------
		>>> eigenvalues,eigenvectors = H.eigh(pars=pars,**eigh_args)

		"""
		eigh_args["overwrite_a"] = True
		
		if self.Ns <= 0:
			return _np.asarray([]),_np.asarray([[]])

		# fill dense array with hamiltonian
		H_dense = self.todense(pars=pars)		
		# calculate eigh
		E,H_dense = _la.eigh(H_dense,**eigh_args)
		return E,H_dense

	def eigvalsh(self,pars={},**eigvalsh_args):
		"""Computes ALL eigenvalues of hermitian `quantum_operator` quantum_operator using DENSE hermitian methods.

		This function method solves for all eigenvalues. It calls 
		`numpy.linalg.eigvalsh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh>`_, 
		and uses wrapped LAPACK functions which are contained in the module py_lapack.

		Notes
		-----
		Assumes the quantum_operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		eigvalsh_args : 
			For all additional arguments see documentation of `numpy.linalg.eigvalsh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh>`_.
			
		Returns
		--------
		numpy.ndarray
			Eigenvalues of the `quantum_operator` quantum_operator.

		Examples
		---------
		>>> eigenvalues = H.eigvalsh(pars=pars,**eigvalsh_args)

		"""

		if self.Ns <= 0:
			return _np.asarray([])

		H_dense = self.todense(pars=pars)
		E = _np.linalg.eigvalsh(H_dense,**eigvalsh_args)
		#eigvalsh_args["overwrite_a"] = True
		#E = _la.eigvalsh(H_dense,**eigvalsh_args)
		return E


	### routines to change object type	

	def tocsr(self,pars={}):
		"""Returns copy of a `quantum_operator` object for parameters `pars` as a `scipy.sparse.csr_matrix`.

		Casts the `quantum_operator` object as a
		`scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_
		object.

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity. 

		Returns
		--------
		:obj:`scipy.sparse.csr_matrix`

		Examples
		---------
		>>> H_csr=H.tocsr(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)

		H = _sp.csr_matrix(self.get_shape,dtype=self._dtype)

		for key,J in pars.items():
			try:
				H += J*_sp.csr_matrix(self._quantum_operator[key])
			except:
				H = H + J*_sp.csr_matrix(self._quantum_operator[key])

		return H

	def tocsc(self,pars={}):
		"""Returns copy of a `quantum_operator` object for parameters `pars` as a `scipy.sparse.csc_matrix`.

		Casts the `quantum_operator` object as a
		`scipy.sparse.csc_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html>`_
		object.

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		--------
		:obj:`scipy.sparse.csc_matrix`

		Examples
		---------
		>>> H_csc=H.tocsc(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)

		H = _sp.csc_matrix(self.get_shape,dtype=self._dtype)

		for key,J in pars.items():
			try:
				H += J*_sp.csc_matrix(self._quantum_operator[key])
			except:
				H = H + J*_sp.csc_matrix(self._quantum_operator[key])

		return H

	def todense(self,pars={},out=None):
		"""Returns copy of a `quantum_operator` object for parameters `pars` as a dense array.

		This function can overflow memory if not used carefully!

		Notes
		-----
		If the array dimension is too large, scipy may choose to cast the `quantum_operator` quantum_operator as a
		`numpy.matrix` instead of a `numpy.ndarray`. In such a case, one can use the `quantum_operator.toarray()`
		method.

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		out : numpy.ndarray
			Array to fill in with the output.
		
		Returns
		--------
		obj
			Depending of size of array, can be either one of

			* `numpy.ndarray`.
			* `numpy.matrix`.

		Examples
		---------
		>>> H_dense=H.todense(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)
			out = _np.asmatrix(out)

		for key,J in pars.items():
			out += J * self._quantum_operator[key]
		
		return out

	def toarray(self,pars={},out=None):
		"""Returns copy of a `quantum_operator` object for parameters `pars` as a dense array.

		This function can overflow memory if not used carefully!


		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		out : numpy.ndarray
			Array to fill in with the output.
		
		Returns
		--------
		numpy.ndarray
			Dense array.

		Examples
		---------
		>>> H_dense=H.toarray(pars=pars)

		"""

		pars = self._check_scalar_pars(pars)

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)

		for key,J in pars.items():
			out += J * self._quantum_operator[key]
		
		return out

	def aslinearoperator(self,pars={}):
		"""Returns copy of a `quantum_operator` object for parameters `pars` as a `scipy.sparse.linalg.Linearquantum_operator`.

		Casts the `quantum_operator` object as a
		`scipy.sparse.linalg.Linearquantum_operator <https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.sparse.linalg.Linearquantum_operator.html>`_
		object.

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		--------
		:obj:`scipy.sparse.linalg.Linearquantum_operator`

		Examples
		---------
		>>> H_aslinop=H.aslinearquantum_operator(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)
		matvec = functools.partial(_quantum_operator_dot,self,pars)
		rmatvec = functools.partial(_quantum_operator_dot,self.H,pars)
		return _sla.LinearOperator(self.get_shape,matvec,rmatvec=rmatvec,matmat=matvec,dtype=self._dtype)		

	def tohamiltonian(self,pars={}):
		"""Returns copy of a `quantum_operator` object for parameters `pars` as a `hamiltonian` object.

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		--------
		:obj:`hamiltonian`

		Examples
		---------
		>>> H_aslinop=H.tohamiltonian(pars=pars)

		"""
		pars = self._check_hamiltonian_pars(pars)

		static=[]
		dynamic=[]

		for key,J in pars.items():
			if type(J) is tuple and len(J) == 2:
				dynamic.append([self._quantum_operator[key],J[0],J[1]])
			else:
				if J == 1.0:
					static.append(self._quantum_operator[key])
				else:
					static.append(J*self._quantum_operator[key])

		return hamiltonian_core.hamiltonian(static,dynamic,dtype=self._dtype)


	### algebra operations

	def transpose(self,copy=False):
		"""Transposes `quantum_operator` quantum_operator.

		Notes
		-----
		This function does NOT conjugate the quantum_operator.

		Returns
		--------
		:obj:`quantum_operator`
			:math:`H_{ij}\\mapsto H_{ji}`

		Examples
		---------

		>>> H_tran = H.transpose()

		"""
		new_dict = {key:op.transpose() for key,op in iteritems(self._quantum_operator)}
		return quantum_operator(new_dict,basis=self._basis,dtype=self._dtype,copy=copy)

	def conjugate(self):
		"""Conjugates `quantum_operator` quantum_operator.

		Notes
		-----
		This function does NOT transpose the quantum_operator.

		Returns
		--------
		:obj:`quantum_operator`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Examples
		---------

		>>> H_conj = H.conj()

		"""
		new_dict = {key:op.conjugate() for key,op in iteritems(self._quantum_operator)}
		return quantum_operator(new_dict,basis=self._basis,dtype=self._dtype,copy=False)

	def conj(self):
		"""Conjugates `quantum_operator` quantum_operator.

		Notes
		-----
		This function does NOT transpose the quantum_operator.

		Returns
		--------
		:obj:`quantum_operator`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Examples
		---------

		>>> H_conj = H.conj()

		"""
		return self.conjugate()

	def getH(self,copy=False):
		"""Calculates hermitian conjugate of `quantum_operator` quantum_operator.

		Parameters
		-----------
		copy : bool, optional
			Whether to return a deep copy of the original object. Default is `copy = False`.

		Returns
		--------
		:obj:`quantum_operator`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Examples
		---------

		>>> H_herm = H.getH()

		"""
		return self.conj().transpose(copy=copy)


	### lin-alg operations

	def diagonal(self,pars={}):
		""" Returns diagonal of `quantum_operator` quantum_operator for parameters `pars`.

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		--------
		numpy.ndarray
			array containing the diagonal part of the operator :math:`diag_j = H_{jj}(\\lambda)`.

		Examples
		---------

		>>> H_diagonal = H.diagonal(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)
		diag = _np.zeros(self.Ns,dtype=self._dtype)
		for key,value in iteritems(self._quantum_operator_dict):
			diag += pars[key] * value.diagonal()
		return diag

	def trace(self,pars={}):
		""" Calculates trace of `quantum_operator` quantum_operator for parameters `pars`.

		Parameters
		-----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		--------
		float
			Trace of quantum_operator :math:`\\sum_{j=1}^{Ns} H_{jj}(\\lambda)`.

		Examples
		---------

		>>> H_tr = H.trace(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)
		tr = 0.0
		for key,value in iteritems(self._quantum_operator_dict):
			try:
				tr += pars[key] * value.trace()
			except AttributeError:
				tr += pars[key] * value.diagonal().sum()
		return tr

	def astype(self,dtype):
		""" Changes data type of `quantum_operator` object.

		Parameters
		-----------
		dtype : 'type'
			The data type (e.g. numpy.float64) to cast the Hamiltonian with.

		Returns
		`quantum_operator`
			quantum_operator with altered data type.

		Examples
		---------
		>>> H_cpx=H.astype(np.complex128)

		"""
		if dtype not in hamiltonian_core.supported_dtypes:
			raise ValueError("quantum_operator can only be cast to floating point types")

		if dtype == self._dtype:
			return quantum_operator(self._quantum_operator_dict,basis=self._basis,dtype=dtype,copy=False)
		else:
			return quantum_operator(self._quantum_operator_dict,basis=self._basis,dtype=dtype,copy=True)

	def copy(self,deep=False):
		"""Returns a deep copy of `quantum_operator` object."""
		return quantum_operator(self._quantum_operator_dict,basis=self._basis,dtype=self._dtype,copy=deep)


	def __call__(self,**pars):
		pars = self._check_scalar_pars(pars)
		if self.is_dense:
			return self.todense(pars)
		else:
			return self.tocsr(pars)

	def __neg__(self):
		return self.__imul__(-1)

	def __iadd__(self,other):
		self._is_dense = self._is_dense or other._is_dense
		if isinstance(other,quantum_operator):
			for key,value in iteritems(other._quantum_operator_dict):
				if key in self._quantum_operator_dict:
					self._quantum_operator_dict[key] = self._quantum_operator_dict[key] + value
				else:
					self._quantum_operator_dict[key] = value
			return self
		elif other == 0:
			return self
		else:
			return NotImplemented

	def __add__(self,other):
		new_type = _np.result_type(self._dtype, other.dtype)
		new = self.astype(new_type)
		new += other
		return new



	def __isub__(self,other):
		self._is_dense = self._is_dense or other._is_dense
		if isinstance(other,quantum_operator):
			for key,values in iteritems(other._quantum_operator_dict):
				if key in self._quantum_operator_dict:
					self._quantum_operator_dict[key] = self._quantum_operator_dict[key] - value
				else:
					self._quantum_operator_dict[key] = -value
			return self
		elif other == 0:
			return self
		else:
			return NotImplemented

	def __sub__(self,other):
		new_type = _np.result_type(self._dtype, other.dtype)
		new = self.astype(new_type)
		new -= other
		return new		

	def __imul__(self,other):
		if not _np.isscalar(other):
			return NotImplemented
		else:
			for op in itervalues(self._quantum_operator_dict):
				op *= other

			return self

	def __mul__(self,other):
		new_type = _np.result_type(self._dtype, other.dtype)
		new = self.astype(new_type)
		new *= other
		return new

	def __idiv__(self,other):
		if not _np.isscalar(other):
			return NotImplemented
		else:
			for op in itervalues(self._quantum_operator_dict):
				op /= other

			return self

	def __div__(self,other):
		new_type = _np.result_type(self._dtype, other.dtype)
		new = self.astype(new_type)
		new /= other
		return new




	def _check_hamiltonian_pars(self,pars):

		if not isinstance(pars,dict):
			raise ValueError("expecing dictionary for parameters.")

		extra = set(pars.keys()) - set(self._quantum_operator.keys())
		if extra:
			raise ValueError("unexpected couplings: {}".format(extra))

		missing = set(self._quantum_operator.keys()) - set(pars.keys())
		for key in missing:
			pars[key] = _np.array(1,dtype=_np.int32)


		for key,J in pars.items():
			if type(J) is tuple:
				if len(J) != 2:
					raise ValueError("expecting parameters to be either scalar or tuple of function and arguements of function.")
			else:
				J = _np.array(J)				
				if J.ndim > 0:
					raise ValueError("expecting parameters to be either scalar or tuple of function and arguements of function.")


		return pars

	def _check_scalar_pars(self,pars):

		if not isinstance(pars,dict):
			raise ValueError("expecing dictionary for parameters.")

		extra = set(pars.keys()) - set(self._quantum_operator.keys())
		if extra:
			raise ValueError("unexpected couplings: {}".format(extra))


		missing = set(self._quantum_operator.keys()) - set(pars.keys())
		for key in missing:
			pars[key] = _np.array(1,dtype=_np.int32)

		for J in pars.values():
			J = _np.array(J)				
			if J.ndim > 0:
				raise ValueError("expecting parameters to be either scalar or tuple of function and arguements of function.")

		return pars

	# checks
	def _mat_checks(self,other,casting="same_kind"):
		try:
			if other.shape != self._shape: # only accepts square matricies 
				raise ValueError('shapes do not match')
			if not _np.can_cast(other.dtype,self._dtype,casting=casting):
				raise ValueError('cannot cast types')
		except AttributeError:
			if other._shape != self._shape: # only accepts square matricies 
				raise ValueError('shapes do not match')
			if not _np.can_cast(other.dtype,self._dtype,casting=casting):
				raise ValueError('cannot cast types')	


def isquantum_operator(obj):
	"""Checks if instance is object of `quantum_operator` class.

	Parameters
	-----------
	obj : 
		Arbitraty python object.

	Returns
	--------
	bool
		Can be either of the following:

		* `True`: `obj` is an instance of `quantum_operator` class.
		* `False`: `obj` is NOT an instance of `quantum_operator` class.

	"""
	return isinstance(obj,quantum_operator)

	
