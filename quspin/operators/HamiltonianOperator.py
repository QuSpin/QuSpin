from __future__ import print_function, division

import hamiltonian

from .hamiltonian import ishamiltonian
from .hamiltonian import _check_static
from .hamiltonian import supported_dtypes

from ..basis import spin_basis_1d as _default_basis
from ..basis import isbasis as _isbasis

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.sparse as _sp
import numpy as _np

from copy import deepcopy as _deepcopy # copies only at top level references the data of old objects


__all__=["HamiltonianOperator","isHamiltonianOperator"]

class HamiltonianOperator(object):
	"""Applies quantum operator directly on state, without constructing operator.

	The `HamiltonianOperator` class uses the `basis.Op` function to calculate the matrix vector product on the 
	fly, greatly reducing the amount of memory needed for a calculation at the cost of speed.

	This object is useful for doing large scale Lanczos calculations using the `eigsh` method.

	Notes
	-----
	The class does NOT yet support time-dependent operators. 

	Examples
	--------

	"""
	def __init__(self,operator_list,system_arg,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**basis_args):
		"""Intializes the `HamiltonianOperator` object.
		
		Parameters
		----------

		operator_list : list
			Contains list of objects to calculate the static part of a `HamiltonianOperator` operator.
			The format goes like:

			>>> static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]
		system_arg : obj
			Can be either one of
				* int: number of sites to create the default spin basis with.
				* :obj:`basis`: basis object to construct quantum operator with.
		dtype : 'type'
			Data type (e.g. numpy.float64) to construct the operator with.
		check_symm : bool, optional 
			Enable/Disable symmetry check on `static_list` and `dynamic_list`.
		check_herm : bool, optional
			Enable/Disable hermiticity check on `static_list` and `dynamic_list`.
		check_pcon : bool, optional
			Enable/Disable particle conservation check on `static_list` and `dynamic_list`.
		basis_args : dict
			Optional additional arguments to pass to the `basis` class, if not already using a `basis` object
			to create the operator.

		"""

		if type(operator_list) in [list,tuple]:
			for ele in operator_list:
				if not _check_static(ele):
					raise ValueError("HamiltonianOperator only supports operator string representations.")
		else: 
			raise TypeError('expecting list/tuple of lists/tuples containing opstr and list of indx')

		self._operator_list = tuple(operator_list)

		if not (dtype in supported_dtypes):
			raise TypeError('hamiltonian does not support type: '+str(dtype))
		else:
			self._dtype=dtype

		if type(system_arg) is int:
			self._basis = _default_basis(system_arg,**basis_args)
		elif _isbasis(system_arg):
			self._basis = system_arg
		else:
			raise ValueError("expecting integer or basis object for 'system_arg'")


		if check_herm:
			self.basis.check_hermitian(operator_list, [])

		if check_symm:
			self.basis.check_symm(operator_list,[])

		if check_pcon:
			self.basis.check_pcon(operator_list,[])

		self._unique_me = self.basis.unique_me
		

		self._transposed = False
		self._conjugated = False
		self._scale = dtype(1.0)
		self._dtype = dtype
		self._ndim = 2
		self._shape = (self._basis.Ns,self._basis.Ns)
		self._LinearOperator = _sp.linalg.LinearOperator(self._shape,self.matvec,matmat=self.matvec,rmatvec=self.rmatvec,dtype=self._dtype)



	@property
	def basis(self):
		""":obj:`basis`: basis used to build the `hamiltonian` object. Defaults to `None` if operator has 
		no basis (i.e. was created externally and passed as a precalculated array).

		"""
		return self._basis

	@property
	def ndim(self):
		"""int: number of dimensions, always equal to 2. """
		return self._ndim

	@property
	def operator_list(self):
		"""list: operator list used to create this object."""
		return self._operator_list

	@property
	def get_shape(self):
		"""tuple: shape of the `HamiltonianOperator` object, always equal to `(Ns,Ns)`."""
		return self._shape

	@property
	def Ns(self):
		"""int: number of states in the (symmetry-reduced) Hilbert space spanned by `basis`."""
		return self._shape[0]

	@property
	def dtype(self):
		"""type: data type of `HamiltonianOperator` object."""
		return _np.dtype(self._dtype).name

	@property
	def T(self):
		""":obj:`HamiltonianOperator`: Transposes the matrix defining the operator: :math:`H_{ij}\\mapsto H_{ji}`."""
		return self.transpose(copy = False)

	@property
	def H(self):
		""":obj:`HamiltonianOperator`: Transposes and conjugates the matrix defining the operator: :math:`H_{ij}\\mapsto H_{ji}^*`."""
		return self.getH(copy = False)

	@property
	def LinearOperator(self):
		""":obj:`scipy.sparse.linalg.LinearOperator`: Casts `HamiltonianOperator` object as `sparse.linalg.LinearOperator <https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.sparse.linalg.LinearOperator.html/>`_."""
		return self.get_LinearOperator()


	### state manipulation/observable routines

	def dot(self,other):
		"""Matrix-vector multiplication of `HamiltonianOperator` operator, with state `V`.

		.. math::
			H|V\\rangle

		Parameters
		----------
		other : numpy.ndarray
			Vector (quantums tate) to multiply the `HamiltonianOperator` operator with.	

		Returns
		-------
		numpy.ndarray
			Vector corresponding to the `hamiltonian` operator applied on the state `V`.

		Examples
		--------
		>>> B = H.dot(A,check=True)

		corresponds to :math:`B = HA`. 
	
		"""
		return self.__mul__(other)

	def rdot(self,other):
		"""Vector-matrix multiplication of `HamiltonianOperator` operator, with state `V`.

		.. math::
			\\langle V|H

		Parameters
		----------
		other : numpy.ndarray
			Vector (quantums tate) to multiply the `HamiltonianOperator` operator with.	

		Returns
		-------
		numpy.ndarray
			Vector corresponding to the `hamiltonian` operator applied on the state `V`.

		Examples
		--------
		>>> B = H.dot(A,check=True)

		corresponds to :math:`B = AH`. 
	
		"""
		return self.__rmul__(other)

	def rmatvec(self,other):
		return self.H.matvec(other)

	def matvec(self,other):
		result_dtype = _np.result_type(self._dtype,other.dtype)
		new_other = _np.zeros_like(other,dtype=result_dtype)
		for opstr, bonds in self.operator_list:
			for bond in bonds:
				J = bond[0]*self._scale
				indx = _np.asarray(bond[1:])
				if not self._transposed:
					ME, row, col = self.basis.Op(opstr, indx, J, self._dtype)
				else:
					ME, col, row = self.basis.Op(opstr, indx, J, self._dtype)

				if self._conjugated:
					ME = ME.conj()

				if self._unique_me:
					new_other[row] += (other[col] * ME)
				else:
					while len(row) > 0:
						row_unique,args = _np.unique(row,return_index=True)
						col_unique = col[args]

						new_other[row_unique] += (other[col_unique] * ME[args])
						row = _np.delete(row,args)
						col = _np.delete(col,args)
						ME = _np.delete(ME,args)


		return new_other

	def matmat(self,other):
		result_dtype = _np.result_type(self._dtype,other.dtype)
		new_other = _np.zeros_like(other,dtype=result_dtype)
		for opstr, bonds in self.operator_list:
			for bond in bonds:
				J = bond[0]*self._scale
				indx = _np.asarray(bond[1:])
				if not self._transposed:
					ME, row, col = self.basis.Op(opstr, indx, J, self._dtype)
				else:
					ME, col, row = self.basis.Op(opstr, indx, J, self._dtype)

				if self._conjugated:
					ME = ME.conj()


				# if there are only one matirx element per row then the indexing should work
				if self._unique_me:
					new_other[row] += (other[col] * ME)
				else:
				# if there are multiply matrix elements per row as there are for some
				# symmetries availible then do the indexing for unique elements then
				# delete them from the list and then repeat until all elements have been 
				# taken care of. This is less memory efficient but works well for when
				# there are a few number of matrix elements per row. 
					while len(row) > 0:
						row_unique,args = _np.unique(row,return_index=True)
						col_unique = col[args]

						new_other[row_unique] += (other[col_unique] * ME[args])
						row = _np.delete(row,args)
						col = _np.delete(col,args)
						ME = _np.delete(ME,args)
	
		if isinstance(other,_np.matrix):		
			return _np.asmatrix(new_other)
		else:
			return new_other

	### Diagonalisation routines

	def eigsh(self,**eigsh_args):
		"""Computes SOME eigenvalues of hermitian `HamiltonianOperator` operator using SPARSE hermitian methods.

		This function method solves for eigenvalues and eigenvectors, but can only solve for a few of them accurately.
		It calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html/>`_, which is a wrapper for ARPACK.

		Notes
		-----
		Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		----------
		eigsh_args : 
			For all additional arguments see documentation of `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html/>`_.
			
		Returns
		-------
		tuple
			Tuple containing the `(eigenvalues, eigenvectors)` of the `HamiltonianOperator` operator.

		Examples
		--------
		>>> eigenvalues,eigenvectors = H.eigsh(**eigsh_args)

		"""
		return _sla.eigsh(self.LinearOperator,**eigsh_args)

	### algebra operations

	def transpose(self,copy=False):
		"""Transposes `HamiltonianOperator` operator.

		Notes
		-----
		This function does NOT conjugate the operator.

		Returns
		-------
		:obj:`HamiltonianOperator`
			:math:`H_{ij}\\mapsto H_{ji}`

		Examples
		--------

		>>> H_tran = H.transpose()

		"""
		if copy:
			return self.copy().transpose()
		else:
			self._transposed = not self._transposed
			return self

	def conj(self):
		"""Conjugates `HamiltonianOperator` operator.

		Notes
		-----
		This function does NOT transpose the operator.

		Returns
		-------
		:obj:`HamiltonianOperator`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Examples
		--------

		>>> H_conj = H.conj()

		"""
		self._conjugated = not self._conjugated
		return self

	def getH(self,copy=False):
		"""Calculates hermitian conjugate of `HamiltonianOperator` operator.

		Parameters
		----------
		copy : bool, optional
			Whether to return a deep copy of the original object. Default is `copy = False`.

		Returns
		-------
		:obj:`HamiltonianOperator`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Examples
		--------

		>>> H_herm = H.getH()

		"""
		if copy:
			return self.copy().get_H()
		else:
			return self.conj().transpose()

	
	### special methods

	def get_LinearOperator(self):
		"""Casts `HamiltonianOperator` object as `scipy.sparse.linalg.LinearOperator <https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.sparse.linalg.LinearOperator.html/>`_."""
		return self._LinearOperator


	def copy(self):
		"""Returns a deep copy of `HamiltonianOperator` object."""
		return _deepcopy(self)



	def __repr__(self):
		return "<{0}x{1} quspin HamiltonianOperator of type '{2}'>".format(*(self._shape[0],self._shape[1],self._dtype))




	def __neg__(self):
		return self.__mul__(-1)

	def __add__(self,other):
		if other.__class__ in [_np.ndarray,_np.matrix]:
			dense = True
		elif _sp.issparse(other):
			dense = False
		elif ishamiltonian(other):
			return self._add_hamiltonian(other)
		elif _np.isscalar(other):
			return self._mul_scalar(other)
		else:
			dense = True
			other = np.asanyarray(other)

		if self._shape != other.shape:
			raise ValueError("dimension mismatch with shapes {0} and {1}".format(self._shape,other.shape))

		if dense:
			return self._add_dense(other)
		else:
			return self._add_sparse(other)

	def __iadd__(self,other):
		return NotImplemented

	def __radd__(self,other):
		return self.__add__(other)

	def __sub__(self,other):
		if other.__class__ in [_np.ndarray,_np.matrix]:
			dense = True
		elif _sp.issparse(other):
			dense = False
		elif ishamiltonian(other):
			return self._sub_hamiltonian(other)
		elif _np.isscalar(other):
			return self._mul_scalar(other)
		else:
			dense = False
			other = np.asanyarray(other)

		if self._shape != other.shape:
			raise ValueError("dimension mismatch with shapes {0} and {1}".format(self._shape,other.shape))

		if dense:
			return self._sub_dense(other)
		else:
			return self._sub_sparse(other)

	def __isub__(self,other):
		return NotImplemented

	def __rsub__(self,other):
		return -(self.__sub__(other))

	def __imul__(self,other):
		if _np.isscalar(other):
			return self._mul_scalar(other)
		else:
			return NotImplemented

	def __mul__(self,other):
		if other.__class__ in [_np.ndarray,_np.matrix]:
			dense = True
		elif _sp.issparse(other):
			dense = False
		elif ishamiltonian(other):
			return self._mul_hamiltonian(other)
		elif _np.isscalar(other):
			return self._mul_scalar(other)
		else:
			dense = True
			other = np.asanyarray(other)

		if self._shape[1] != other.shape[0]:
			raise ValueError("dimension mismatch with shapes {0} and {1}".format(self._shape,other.shape))

		if dense:
			if other.ndim == 1:
				return self.matvec(other)
			elif other.ndim == 2:
				return self.matmat(other)
			else:
				raise ValueError
		else:
			return self._mul_sparse(other)

	def __rmul__(self,other):
		if other.__class__ in [_np.ndarray,_np.matrix]:
			dense = True
		elif _sp.issparse(other):
			dense = False
		elif ishamiltonian(other):
			return self._rmul_hamiltonian(other)
		elif _np.isscalar(other):
			return self._mul_scalar(other)
		else:
			dense = True
			other = np.asanyarray(other)

		if dense:
			if other.ndim == 1:
				return self.T.matvec(other)
			elif other.ndim == 2:
				if self._shape[0] != other.shape[1]:
					raise ValueError("dimension mismatch with shapes {0} and {1}".format(self._shape,other.shape))
				return (self.T.matmat(other.T)).T
			else:
				raise ValueError
		else:
			if self._shape[0] != other.shape[1]:
				raise ValueError("dimension mismatch with shapes {0} and {1}".format(self._shape,other.shape))
			return (self.T._mul_sparse(other.T)).T

	



	def _mul_scalar(self,other):
		self._dtype = _np.result_type(self._dtype,other)
		self._scale *= other

	def _mul_hamiltonian(self,other):
		return NotImplemented

	def _rmul_hamiltonian(self,other):
		return NotImplemented

	def _add_hamiltonian(self,other):
		return NotImplemented

	def _sub_hamiltonian(self,other):
		return NotImplemented

	def _mul_sparse(self,other):
		result_dtype = _np.result_type(self._dtype,other.dtype)
		new_other = other.__class__(other.shape,dtype=result_dtype)
		for opstr, bonds in self.operator_list:
			for bond in bonds:
				J = bond[0]
				indx = _np.asarray(bond[1:])
				if not self._transposed:
					ME, row, col = self.basis.Op(opstr, indx, J, self._dtype)
				else:
					ME, col, row = self.basis.Op(opstr, indx, J, self._dtype)

				if self._conjugated:
					ME = ME.conj()

				new_other += _sp.csr_matrix((ME,(row,col)),shape=self._shape).dot(other)

		return new_other

	def _add_sparse(self,other):
		return NotImplemented

	def _sub_sparse(self,other):
		return NotImplemented

	def _add_dense(self,other):
		return NotImplemented

	def _sub_dense(self,other):
		return NotImplemented

	def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
		# """Method for compatibility with NumPy's ufuncs and dot
		# functions.
		# """

		if (func == np.dot) or (func == np.multiply):
			if pos == 0:
				return self.__mul__(inputs[1])
			if pos == 1:
				return self.__rmul__(inputs[0])
			else:
				return NotImplemented


def isHamiltonianOperator(obj):
	"""Checks if instance is object of `HamiltonianOperator` class.

	Parameters
	----------
	obj : 
		Arbitraty python object.

	Returns
	-------
	bool
		Can be either of the following:

		* `True`: `obj` is an instance of `HamiltonianOperator` class.
		* `False`: `obj` is NOT an instance of `HamiltonianOperator` class.

	"""
	return isinstance(obj,HamiltonianOperator)



	