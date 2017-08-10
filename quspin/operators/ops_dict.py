from __future__ import print_function, division

from ..basis import spin_basis_1d as _default_basis
from ..basis import isbasis as _isbasis

from .make_hamiltonian import make_static as _make_static

import hamiltonian

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

import functools

from copy import deepcopy as _deepcopy # recursively copies all data into new object
from copy import copy as _shallowcopy # copies only at top level references the data of old objects


__all__=["ops_dict","isops_dict"]
		
# function used to create LinearOperator with fixed set of parameters. 
def _ops_dict_dot(op,pars,v):
	return op.dot(v,pars=pars,check=False)

class ops_dict(object):
	"""Constructs parameter-dependent quantum operators.

		The `ops_dict` class maps operators to keys of a dictionary. When calling various methods
		of the operators, it allows one to 'dynamically' specify the pre-factors of these operators.

		It is often required to be able to handle a parameter-dependent Hamiltonian :math:`H(\\lambda)`, e.g.

		.. math::
			H(J_{zz}, h_x) = \sum_j J_{zz}S^z_jS^z_{j+1} + h_xS^x_j

		Example
		-------

	"""
	def __init__(self,input_dict,N=None,basis=None,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**kwargs):
		"""Intializes the `ops_dict` object (parameter dependent quantum operators).

		Parameters
		----------
		input_dict : dict
			The `values` of this dictionary contain operator lists, in the same format as the `static_list` 
			argument of the `hamiltonian` class.

			The `keys` of this dictionary correspond to the parameter values, e.g. :math:`J_{zz},h_x`, and are 
			used to specify the coupling strength during calls of the `ops_dict` class methods.

			>>> # use "Jzz" and "hx" keys to specify the zz and x coupling strengths, respectively
			>>> input_dict = { "Jzz": [["zz",Jzz_bonds]], "hx" : [["x" ,hx_site ]] } 

		N : int, optional
			Number of lattice sites for the `hamiltonian` object.
		dtype : 'type'
			Data type (e.g. numpy.float64) to construct the operator with.
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
			to create the operator.		
			
		"""
		self._is_dense = False
		self._ndim = 2
		self._basis = None



		if not (dtype in hamiltonian.supported_dtypes):
			raise TypeError('hamiltonian does not support type: '+str(dtype))
		else:
			self._dtype=dtype
		
		opstr_dict = {}
		other_dict = {}
		self._ops_dict = {}
		if isinstance(input_dict,dict):
			for key,op in input_dict.items():
				if type(key) is not str:
					raise ValueError("keys to input_dict must be strings.")
					
				if type(op) not in [list,tuple]:
					raise ValueError("input_dict must contain values which are lists/tuples.")
				opstr_list = []
				other_list = []
				for ele in op:
					if hamiltonian._check_static(ele):
						opstr_list.append(ele)
					else:
						other_list.append(ele)

				if opstr_list:
					opstr_dict[key] = opstr_list
				if other_list:
					other_dict[key] = other_list
		elif isinstance(input_dict,ops_dict):
			other_dict = {key:[value] for key,value in input_dict._operator_dict.items()} 
		else:
			raise ValueError("input_dict must be dictionary or another ops_dict operators")
			


		if opstr_dict:
			# check if user input basis

			if basis is not None:
				if len(kwargs) > 0:
					wrong_keys = set(kwargs.keys())
					temp = ", ".join(["{}" for key in wrong_keys])
					raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))

			# if not
			if basis is None: 
				if N is None: # if L is missing 
					raise Exception('if opstrs in use, argument N needed for basis class')

				if type(N) is not int: # if L is not int
					raise TypeError('argument N must be integer')

				basis=_default_basis(N,**kwargs)

			elif not _isbasis(basis):
				raise TypeError('expecting instance of basis class for argument: basis')


			static_opstr_list = []
			for key,opstr_list in opstr_dict.items():
				static_opstr_list.extend(opstr_list)

			if check_herm:
				basis.check_hermitian(static_opstr_list, [])

			if check_symm:
				basis.check_symm(static_opstr_list,[])

			if check_pcon:
				basis.check_pcon(static_opstr_list,[])

			self._basis=basis
			self._shape=(basis.Ns,basis.Ns)

			for key,opstr_list in opstr_dict.items():
				self._ops_dict[key]=_make_static(basis,opstr_list,dtype)

		if other_dict:
			if not hasattr(self,"_shape"):
				found = False
				if shape is None: # if no shape argument found, search to see if the inputs have shapes.
					for key,O_list in other_dict.items():
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
					raise ValueError('operator must be square matrix')

				self._shape=shape



			for key,O_list in other_dict.items():
				for i,O in enumerate(O_list):
					if _sp.issparse(O):
						self._mat_checks(O)
						if i == 0:
							self._ops_dict[key] = O
						else:
							try:
								self._ops_dict[key] += O
							except NotImplementedError:
								self._ops_dict[key] = self._ops_dict[key] + O

					elif O.__class__ is _np.ndarray:
						self._mat_checks(O)
						self._is_dense=True
						if i == 0:
							self._ops_dict[key] = O
						else:
							try:
								self._ops_dict[key] += O
							except NotImplementedError:
								self._ops_dict[key] = self._ops_dict[key] + O

					elif O.__class__ is _np.matrix:
						self._mat_checks(O)
						self._is_dense=True
						if i == 0:
							self._ops_dict[key] = O
						else:
							try:
								self._ops_dict[key] += O
							except NotImplementedError:
								self._ops_dict[key] = self._ops_dict[key] + O

					else:
						O = _np.asanyarray(O)
						self._mat_checks(O)
						if i == 0:
							self._ops_dict[key] = O
						else:
							try:
								self._ops_dict[key] += O
							except NotImplementedError:
								self._ops_dict[key] = self._ops_dict[key] + O

					

		else:
			if not hasattr(self,"_shape"):
				if shape is None:
					# check if user input basis
					basis=kwargs.get('basis')	

					# if not
					if basis is None: 
						if N is None: # if N is missing 
							raise Exception("argument N or shape needed to create empty hamiltonian")

						if type(N) is not int: # if L is not int
							raise TypeError('argument N must be integer')

						basis=_default_basis(N,**kwargs)

					elif not _isbasis(basis):
						raise TypeError('expecting instance of basis class for argument: basis')

					shape = (basis.Ns,basis.Ns)

				else:
					basis=kwargs.get('basis')	
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
		""":obj:`basis`: basis used to build the `hamiltonian` object. Defaults to `None` if operator has 
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
		"""tuple: shape of the `ops_dict` object, always equal to `(Ns,Ns)`."""
		return self._shape

	@property
	def is_dense(self):
		"""bool: `True` if the operator contains a dense matrix as a componnent of either 
		the static or dynamic lists.

		"""
		return self._is_dense

	@property
	def dtype(self):
		"""type: data type of `ops_dict` object."""
		return _np.dtype(self._dtype).name

	@property
	def T(self):
		""":obj:`ops_dict`: Transposes the matrix defining the operator: :math:`H_{ij}\\mapsto H_{ji}`."""
		return self.transpose()

	@property
	def H(self):
		""":obj:`ops_dict`: Transposes and conjugates the matrix defining the operator: :math:`H_{ij}\\mapsto H_{ji}^*`."""
		return self.getH()




	### state manipulation/observable routines

	def matvec(self,V):
		return self.dot(V)

	def rmatvec(self,V):
		return self.H.dot(V)

	def matmat(self,V):
		return self.dot(V)

	def dot(self,V,pars={},check=True):
		"""Matrix-vector multiplication of `ops_dict` operator for parameters `pars`, with state `V`.

		.. math::
			H(t=\\lambda)|V\\rangle

		Note
		----
		It is faster to multiply the individual (static, dynamic) parts of the Hamiltonian first, then add all those 
		vectors together.

		Parameters
		----------
		V : numpy.ndarray
			Vector (quantums tate) to multiply the `ops_dict` operator with.
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		check : bool, optional
			Whether or not to do checks for shape compatibility.
			

		Returns
		-------
		numpy.ndarray
			Vector corresponding to the `ops_dict` operator applied on the state `V`.

		Example
		-------
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
				V_dot += J*self._ops_dict[key].dot(V)
			return V_dot

		if V.ndim > 2:
			raise ValueError("Expecting V.ndim < 3.")




		if V.__class__ is _np.ndarray:
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)
			for key,J in pars.items():
				V_dot += J*self._ops_dict[key].dot(V)


		elif _sp.issparse(V):
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)	
			for key,J in pars.items():
				V_dot += J*self._ops_dict[key].dot(V)



		elif V.__class__ is _np.matrix:
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)
			for key,J in pars.items():
				V_dot += J*self._ops_dict[key].dot(V)

		else:
			V = _np.asanyarray(V)
			if V.ndim not in [1,2]:
				raise ValueError("Expecting 1 or 2 dimensional array")

			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			result_dtype = _np.result_type(V,self._dtype)
			V_dot = _np.zeros(V.shape,dtype=result_dtype)
			for key,J in pars.items():
				V_dot += J*self._ops_dict[key].dot(V)


		return V_dot

	def rdot(self,V,pars={},check=False):
		"""Vector-matrix multiplication of `ops_dict` operator for parameters `pars`, with state `V`.

		.. math::
			\\lamgle V]H(t=\\lambda)

		
		Parameters
		----------
		V : numpy.ndarray
			Vector (quantums tate) to multiply the `ops_dict` operator with.
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		check : bool, optional
			Whether or not to do checks for shape compatibility.
			

		Returns
		-------
		numpy.ndarray
			Vector corresponding to the `ops_dict` operator applied on the state `V`.

		Example
		-------
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
		"""Calculates matrix element of `ops_dict` operator for parameters `pars` in states `Vl` and `Vr`.

		.. math::
			\\langle V_l|H(\\lambda)|V_r\\rangle

		Note
		----
		Taking the conjugate or transpose of the state `Vl` is done automatically.  

		Parameters
		----------
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
		-------
		float
			Matrix element of `ops_dict` operator between the states `Vl` and `Vr`.

		Example
		-------
		>>> H_lr = H.expt_value(Vl,Vr,pars=pars,diagonal=False,check=True)

		corresponds to :math:`H_\\{lr} = \\langle V_l|H(\\lambda=0)|V_r\\rangle`. 

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
		"""Computes SOME eigenvalues of hermitian `ops_dict` operator using SPARSE hermitian methods.

		This function method solves for eigenvalues and eigenvectors, but can only solve for a few of them accurately.
		It calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html/>`_, which is a wrapper for ARPACK.

		Note
		----
		Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		eigsh_args : 
			For all additional arguments see documentation of `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html/>`_.
			
		Returns
		-------
		tuple
			Tuple containing the `(eigenvalues, eigenvectors)` of the `ops_dict` operator.

		Example
		-------
		>>> eigenvalues,eigenvectors = H.eigsh(pars=pars,**eigsh_args)

		"""
		if self.Ns == 0:
			return _np.array([]),_np.array([[]])

		return _sla.eigsh(self.tocsr(pars),**eigsh_args)

	def eigh(self,pars={},**eigh_args):
		"""Computes COMPLETE eigensystem of hermitian `ops_dict` operator using DENSE hermitian methods.

		This function method solves for all eigenvalues and eigenvectors. It calls 
		`numpy.linalg.eigh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigh.html/>`_, 
		and uses wrapped LAPACK functions which are contained in the module py_lapack.

		Note
		----
		Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		eigh_args : 
			For all additional arguments see documentation of `numpy.linalg.eigh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigh.html/>`_.
			
		Returns
		-------
		tuple
			Tuple containing the `(eigenvalues, eigenvectors)` of the `ops_dict` operator.

		Example
		-------
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
		"""Computes ALL eigenvalues of hermitian `ops_dict` operator using DENSE hermitian methods.

		This function method solves for all eigenvalues. It calls 
		`numpy.linalg.eigvalsh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh/>`_, 
		and uses wrapped LAPACK functions which are contained in the module py_lapack.

		Note
		----
		Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		eigvalsh_args : 
			For all additional arguments see documentation of `numpy.linalg.eigvalsh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh/>`_.
			
		Returns
		-------
		numpy.ndarray
			Eigenvalues of the `ops_dict` operator.

		Example
		-------
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
		"""Returns copy of a `ops_dict` object for parameters `pars` as a `scipy.sparse.csr_matrix`.

		Casts the `ops_dict` object as a
		`scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html/>`_
		object.

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity. 

		Returns
		-------
		:obj:`scipy.sparse.csr_matrix`

		Example
		-------
		>>> H_csr=H.tocsr(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)

		H = _sp.csr_matrix(self.get_shape,dtype=self._dtype)

		for key,J in pars.items():
			try:
				H += J*_sp.csr_matrix(self._ops_dict[key])
			except:
				H = H + J*_sp.csr_matrix(self._ops_dict[key])

		return H

	def tocsc(self,pars={}):
		"""Returns copy of a `ops_dict` object for parameters `pars` as a `scipy.sparse.csc_matrix`.

		Casts the `ops_dict` object as a
		`scipy.sparse.csc_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html/>`_
		object.

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		-------
		:obj:`scipy.sparse.csc_matrix`

		Example
		-------
		>>> H_csc=H.tocsc(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)

		H = _sp.csc_matrix(self.get_shape,dtype=self._dtype)

		for key,J in pars.items():
			try:
				H += J*_sp.csc_matrix(self._ops_dict[key])
			except:
				H = H + J*_sp.csc_matrix(self._ops_dict[key])

		return H

	def todense(self,pars={},out=None):
		"""Returns copy of a `ops_dict` object for parameters `pars` as a dense array.

		This function can overflow memory if not used carefully!

		Note
		----
		If the array dimension is too large, scipy may choose to cast the `ops_dict` operator as a
		`numpy.matrix` instead of a `numpy.ndarray`. In such a case, one can use the `ops_dict.toarray()`
		method.

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		out : numpy.ndarray
			Array to fill in with the output.
		
		Returns
		-------
		obj
			Depending of size of array, can be either one of

			* `numpy.ndarray`.
			* `numpy.matrix`.

		Example
		-------
		>>> H_dense=H.todense(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)
			out = _np.asmatrix(out)

		for key,J in pars.items():
			out += J * self._ops_dict[key]
		
		return out

	def toarray(self,pars={},out=None):
		"""Returns copy of a `ops_dict` object for parameters `pars` as a dense array.

		This function can overflow memory if not used carefully!


		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.
		out : numpy.ndarray
			Array to fill in with the output.
		
		Returns
		-------
		numpy.ndarray
			Dense array.

		Example
		-------
		>>> H_dense=H.toarray(pars=pars)

		"""

		pars = self._check_scalar_pars(pars)

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)

		for key,J in pars.items():
			out += J * self._ops_dict[key]
		
		return out

	def aslinearoperator(self,pars={}):
		"""Returns copy of a `ops_dict` object for parameters `pars` as a `scipy.sparse.linalg.LinearOperator`.

		Casts the `ops_dict` object as a
		`scipy.sparse.linalg.LinearOperator <https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.sparse.linalg.LinearOperator.html/>`_
		object.

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		-------
		:obj:`scipy.sparse.linalg.LinearOperator`

		Example
		-------
		>>> H_aslinop=H.aslinearoperator(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)
		matvec = functools.partial(_ops_dict_dot,self,pars)
		rmatvec = functools.partial(_ops_dict_dot,self.H,pars)
		return _sla.LinearOperator(self.get_shape,matvec,rmatvec=rmatvec,matmat=matvec,dtype=self._dtype)		

	def tohamiltonian(self,pars={}):
		"""Returns copy of a `ops_dict` object for parameters `pars` as a `hamiltonian` object.

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		-------
		:obj:`hamiltonian`

		Example
		-------
		>>> H_aslinop=H.tohamiltonian(pars=pars)

		"""
		pars = self._check_hamiltonian_pars(pars)

		static=[]
		dynamic=[]

		for key,J in pars.items():
			if type(J) is tuple and len(J) == 2:
				dynamic.append([self._ops_dict[key],J[0],J[1]])
			else:
				if J == 1.0:
					static.append(self._ops_dict[key])
				else:
					static.append(J*self._ops_dict[key])

		return hamiltonian.hamiltonian(static,dynamic,dtype=self._dtype)


	### algebra operations

	def transpose(self,copy=False):
		"""Transposes `ops_dict` operator.

		Note
		----
		This function does NOT conjugate the operator.

		Returns
		-------
		:obj:`ops_dict`
			:math:`H_{ij}\\mapsto H_{ji}`

		Example
		-------

		>>> H_tran = H.transpose()

		"""
		new = _shallowcopy(self)
		for key,op in self._ops_dict.items():
			new._ops_dict[key] = op.transpose()
		return new

	def conjugate(self):
		"""Conjugates `ops_dict` operator.

		Note
		----
		This function does NOT transpose the operator.

		Returns
		-------
		:obj:`ops_dict`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Example
		-------

		>>> H_conj = H.conj()

		"""
		new = _shallowcopy(self)
		for key,op in self._ops_dict.items():
			new._ops_dict[key] = op.conj()
		return new

	def conj(self):
		"""Conjugates `ops_dict` operator.

		Note
		----
		This function does NOT transpose the operator.

		Returns
		-------
		:obj:`ops_dict`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Example
		-------

		>>> H_conj = H.conj()

		"""
		return self.conjugate()

	def getH(self,copy=False):
		"""Calculates hermitian conjugate of `ops_dict` operator.

		Parameters
		----------
		copy : bool, optional
			Whether to return a deep copy of the original object. Default is `copy = False`.

		Returns
		-------
		:obj:`ops_dict`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Example
		-------

		>>> H_herm = H.getH()

		"""
		return self.conj().transpose(copy=copy)


	### lin-alg operations

	def trace(self,pars={}):
		""" Calculates trace of `ops_dict` operator for parameters `pars`.

		Parameters
		----------
		pars : dict, optional
			Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
			are assumed to be set to inity.

		Returns
		-------
		float
			Trace of operator :math:`\\sum_{j=1}^{Ns} H_{jj}(\\lambda)`.

		Example
		-------

		>>> H_tr = H.tr(pars=pars)

		"""
		pars = self._check_scalar_pars(pars)
		tr = 0.0
		for key,value in self._operator_dict.items():
			try:
				tr += pars[key] * value.trace()
			except AttributeError:
				tr += pars[key] * value.diagonal().sum()
		return tr

	

	def astype(self,dtype):
		""" Changes data type of `ops_dict` object.

		Parameters
		----------
		dtype : 'type'
			The data type (e.g. numpy.float64) to cast the Hamiltonian with.

		Returns
		`ops_dict`
			Operator with altered data type.

		Example
		-------
		>>> H_cpx=H.astype(np.complex128)

		"""
		if dtype not in hamiltonian.supported_dtypes:
			raise ValueError("operator can only be cast to floating point types")
		new = _shallowcopy(self)
		new._dtype = dtype
		for key in self._ops_dict.keys():
			new._ops_dict[key] = self._ops_dict[key].astype(dtype)

		return new

	def copy(self,dtype=None):
		"""Returns a deep copy of `ops_dict` object."""
		return _deepcopy(self)



	"""
	def SO_LinearOperator(self,pars={}):
		pars = self._check_scalar_pars(pars)
		i_pars = {}
		i_pars_c = {}
		for key,J in pars.items():
			i_pars[key] = -1j*J
			i_pars_c[key] = 1j*J

		new = self.astype(_np.complex128)
		matvec = functools.partial(_ops_dict_dot,new,i_pars)
		rmatvec = functools.partial(_ops_dict_dot,new.H,i_pars_c)
		return _sla.LinearOperator(self.get_shape,matvec,rmatvec=rmatvec,matmat=matvec,dtype=_np.complex128)		
	"""



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
		if isinstance(other,ops_dict):
			for key,value in other._operator_dict.items():
				if key in self._operator_dict:
					self._operator_dict[key] = self._operator_dict[key] + value
				else:
					self._operator_dict[key] = value
		elif other == 0:
			return _shallowcopy(self)
		else:
			return NotImplemented

	def __add__(self,other):
		new_type = _np.result_type(self._dtype, other.dtype)
		new = self.astype(new_type)
		new += other
		return new



	def __isub__(self,other):
		self._is_dense = self._is_dense or other._is_dense
		if isinstance(other,ops_dict):
			for key,values in other._operator_dict.items():
				if key in self._operator_dict:
					self._operator_dict[key] = self._operator_dict[key] - value
				else:
					self._operator_dict[key] = -value

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
			for op in self._operator_dict.values():
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
			for op in self._operator_dict.values():
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

		extra = set(pars.keys()) - set(self._ops_dict.keys())
		if extra:
			raise ValueError("unexpected couplings: {}".format(extra))

		missing = set(self._ops_dict.keys()) - set(pars.keys())
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

		extra = set(pars.keys()) - set(self._ops_dict.keys())
		if extra:
			raise ValueError("unexpected couplings: {}".format(extra))


		missing = set(self._ops_dict.keys()) - set(pars.keys())
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


def isops_dict(obj):
	"""Checks if instance is object of `ops_dict` class.

	Parameters
	----------
	obj : 
		Arbitraty python object.

	Returns
	-------
	bool
		Can be either of the following:

		* `True`: `obj` is an instance of `ops_dict` class.
		* `False`: `obj` is NOT an instance of `ops_dict` class.

	"""
	return isinstance(obj,ops_dict)

	
