from __future__ import print_function, division

from .hamiltonian_core import ishamiltonian
from .hamiltonian_core import _check_static
from .hamiltonian_core import supported_dtypes
from .hamiltonian_core import hamiltonian

from ._make_hamiltonian import _consolidate_static

from ..basis import spin_basis_1d as _default_basis
from ..basis import isbasis as _isbasis

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.sparse as _sp
import numpy as _np

from scipy.sparse.linalg import LinearOperator

from six import iteritems
from six.moves import zip

__all__=["quantum_LinearOperator","isquantum_LinearOperator"]

class quantum_LinearOperator(LinearOperator):
	"""Applies a quantum operator directly onto a state, without constructing the operator matrix.

	The `quantum_LinearOperator` class uses the `basis.Op()` function to calculate the matrix vector product on the 
	fly, greatly reducing the amount of memory needed for a calculation at the cost of speed.

	This object is useful for doing large scale Lanczos calculations using the `eigsh` method.

	Notes
	-----
	The class does NOT yet support time-dependent operators. 

	Examples
	---------

	The following example shows how to construct and use `quantum_LinearOperator` objects.

	.. literalinclude:: ../../doc_examples/quantum_linearOperator-example.py
		:linenos:
		:language: python
		:lines: 7-

	"""
	def __init__(self,static_list,N=None,basis=None,diagonal=None,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**basis_args):
		"""Intializes the `quantum_LinearOperator` object.
		
		Parameters
		-----------

		static_list : list
			Contains list of objects to calculate the static part of a `quantum_LinearOperator` operator. Same as
			the `static` argument of the `quantum_operator` class. The format goes like:

			>>> static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]

		N : int, optional 
			number of sites to create the default spin basis with.
		basis : :obj:`basis`, optional
			basis object to construct quantum operator with.
		diagonal : array_like
			array containing diagonal matrix elements precalculated by other means. 
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

		if type(static_list) in [list,tuple]:
			for ele in static_list:
				if not _check_static(ele):
					raise ValueError("quantum_LinearOperator only supports operator string representations.")
		else: 
			raise TypeError('expecting list/tuple of lists/tuples containing opstr and list of indx')

		if dtype not in supported_dtypes:
			raise TypeError('hamiltonian does not support type: '+str(dtype))
		else:
			self._dtype=dtype


		if N==[]:
			raise ValueError("second argument of `quantum_LinearOperator()` canNOT be an empty list.")
		elif type(N) is int and basis is None:
			self._basis = _default_basis(N,**basis_args)
		elif N is None and _isbasis(basis):
			self._basis = basis
		else:
			raise ValueError("expecting integer for N or basis object for basis.")

		self._unique_me = self.basis._unique_me
		self._transposed = False
		self._conjugated = False
		self._scale = _np.array(1.0,dtype=dtype)
		self._dtype = dtype
		self._ndim = 2
		self._shape = (self._basis.Ns,self._basis.Ns)


		if check_herm:
			self.basis.check_hermitian(static_list, [])

		if check_symm:
			self.basis.check_symm(static_list,[])

		if check_pcon:
			self.basis.check_pcon(static_list,[])

		if diagonal is not None:
			self.set_diagonal(diagonal)
		else:
			self._diagonal = None



		static_list = _consolidate_static(static_list)
		self._static_list = []
		for opstr,indx,J in static_list:
			ME,row,col = self.basis.Op(opstr,indx,J,self._dtype)
			if (row==col).all():
				if self._diagonal is None:
					self._diagonal = _np.zeros((self.Ns,),dtype=ME.real.dtype)

				if self._unique_me:
					if row.shape[0] == self.Ns:
						self._diagonal += ME.real
					else:
						self._diagonal[row] += ME[row].real
				else:
					while len(row) > 0:
						# if there are multiply matrix elements per row as there are for some
						# symmetries availible then do the indexing for unique elements then
						# delete them from the list and then repeat until all elements have been 
						# taken care of. This is less memory efficient but works well for when
						# there are a few number of matrix elements per row. 
						row_unique,args = _np.unique(row,return_index=True)

						self._diagonal[row_unique] += ME[args].real
						row = _np.delete(row,args)
						ME = _np.delete(ME,args)					
			else:
				self._static_list.append((opstr,indx,J))
				





	@property
	def shape(self):
		"""tuple: shape of linear operator."""
		return self._shape

	@property
	def basis(self):
		""":obj:`basis`: basis used to build the `hamiltonian` object. 

		Defaults to `None` if operator has  no basis (i.e. was created externally and passed as a precalculated array).
		"""
		return self._basis

	@property
	def ndim(self):
		"""int: number of dimensions, always equal to 2."""
		return self._ndim

	@property
	def static_list(self):
		"""list: operator list used to create this object."""
		return self._static_list

	@property
	def get_shape(self):
		"""tuple: shape of the `quantum_LinearOperator` object, always equal to `(Ns,Ns)`."""
		return self._shape

	@property
	def Ns(self):
		"""int: number of states in the (symmetry-reduced) Hilbert space spanned by `basis`."""
		return self._shape[0]

	@property
	def dtype(self):
		"""type: data type of `quantum_LinearOperator` object."""
		return _np.dtype(self._dtype)

	@property
	def T(self):
		""":obj:`quantum_LinearOperator`: transposes the operator matrix: :math:`H_{ij}\\mapsto H_{ji}`."""
		return self.transpose(copy = False)

	@property
	def H(self):
		""":obj:`quantum_LinearOperator`: transposes and conjugates the operator matrix: :math:`H_{ij}\\mapsto H_{ji}^*`."""
		return self.getH(copy = False)

	@property
	def diagonal(self):
		"""numpy.ndarray: static diagonal part of the linear operator. """
		return self._diagonal

	def set_diagonal(self,diagonal):
		"""Sets the diagonal part of the quantum_LinearOperator.

		Parameters
		-----------
		diagonal: array_like
			array_like object containing the new diagonal.

		"""
		if diagonal.__class__ != _np.ndarray:
			diagonal = _np.asanyarray(diagonal)
		if diagonal.ndim != 1:
			raise ValueError("diagonal must be 1-d array.")
		if diagonal.shape[0] != self.Ns:
			raise ValueError("length of diagonal must be equal to dimension of matrix")

		self._diagonal = diagonal

	### state manipulation/observable routines

	# def dot(self,other):
	# 	"""Matrix-vector multiplication of `quantum_LinearOperator` operator, with state `V`.

	# 	.. math::
	# 		H|V\\rangle

	# 	Parameters
	# 	-----------
	# 	other : numpy.ndarray
	# 		Vector (quantums tate) to multiply the `quantum_LinearOperator` operator with.	

	# 	Returns
	# 	--------
	# 	numpy.ndarray
	# 		Vector corresponding to the `hamiltonian` operator applied on the state `V`.

	# 	Examples
	# 	---------
	# 	>>> B = H.dot(A,check=True)

	# 	corresponds to :math:`B = HA`. 
	
	# 	"""
	# 	return self.__mul__(other)

	# def rdot(self,other):
	# 	"""Vector-matrix multiplication of `quantum_LinearOperator` operator, with state `V`.

	# 	.. math::
	# 		\\langle V|H

	# 	Parameters
	# 	-----------
	# 	other : numpy.ndarray
	# 		Vector (quantums tate) to multiply the `quantum_LinearOperator` operator with.	

	# 	Returns
	# 	--------
	# 	numpy.ndarray
	# 		Vector corresponding to the `hamiltonian` operator applied on the state `V`.

	# 	Examples
	# 	---------
	# 	>>> B = H.dot(A,check=True)

	# 	corresponds to :math:`B = AH`. 
	
	# 	"""
	# 	return self.__rmul__(other)

	def _matvec(self,other):
		result_dtype = _np.result_type(self._dtype,other.dtype)
		new_other = _np.zeros_like(other,dtype=result_dtype)
		if self.diagonal is not None:
			_np.multiply(other.T,self.diagonal,out=new_other.T)

		for opstr,indx,J in self.static_list:
			self.basis.inplace_Op(other,opstr, indx, J, self._dtype,
								self._conjugated,self._transposed,v_out=new_other)
		return new_other

	def _rmatvec(self,other):
		return self.H._matvec(other)

	def _matmat(self,other):
		return self._matvec(other)

	### Diagonalisation routines

	def eigsh(self,**eigsh_args):
		"""Computes SOME eigenvalues and eigenvectors of hermitian `quantum_LinearOperator` operator using SPARSE hermitian methods.

		This function method solves for eigenvalues and eigenvectors, but can only solve for a few of them accurately.
		It calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html>`_, which is a wrapper for ARPACK.

		Notes
		-----
		Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		-----------
		eigsh_args : 
			For all additional arguments see documentation of `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html>`_.
			
		Returns
		--------
		tuple
			Tuple containing the `(eigenvalues, eigenvectors)` of the `quantum_LinearOperator` operator.

		Examples
		---------
		>>> eigenvalues,eigenvectors = H.eigsh(**eigsh_args)

		"""
		return _sla.eigsh(self,**eigsh_args)

	### algebra operations

	def transpose(self,copy=False):
		"""Transposes `quantum_LinearOperator` operator.

		Notes
		-----
		This function does NOT conjugate the operator.

		Returns
		--------
		:obj:`quantum_LinearOperator`
			:math:`H_{ij}\\mapsto H_{ji}`

		Examples
		---------

		>>> H_tran = H.transpose()

		"""
		if copy:
			return self.copy().transpose()
		else:
			self._transposed = not self._transposed
			return self

	def conjugate(self):
		"""Conjugates `quantum_LinearOperator` operator.

		Notes
		-----
		This function does NOT transpose the operator.

		Returns
		--------
		:obj:`quantum_LinearOperator`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Examples
		---------

		>>> H_conj = H.conj()

		"""
		self._conjugated = not self._conjugated
		return self

	def conj(self):
		"""Conjugates `quantum_LinearOperator` operator.

		Notes
		-----
		This function does NOT transpose the operator.

		Returns
		--------
		:obj:`quantum_LinearOperator`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Examples
		---------

		>>> H_conj = H.conj()

		"""
		return self.conjugate()

	def getH(self,copy=False):
		"""Calculates hermitian conjugate of `quantum_LinearOperator` operator.

		Parameters
		-----------
		copy : bool, optional
			Whether to return a deep copy of the original object. Default is `copy = False`.

		Returns
		--------
		:obj:`quantum_LinearOperator`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Examples
		---------

		>>> H_herm = H.getH()

		"""
		if copy:
			return self.copy().get_H()
		else:
			return self.conj().transpose()

	### special methods

	def copy(self):
		"""Returns a deep copy of `quantum_LinearOperator` object."""
		return quantum_LinearOperator(self._static_list,basis=self._basis,
							diagonal=self._diagonal,dtype=self._dtype,
							check_symm=False,check_herm=False,check_pcon=False)

	def __repr__(self):
		return "<{0}x{1} quspin quantum_LinearOperator of type '{2}'>".format(*(self._shape[0],self._shape[1],self._dtype))

	def __neg__(self):
		return self.__mul__(-1)

	def __add__(self,other):
		if other.__class__ in [_np.ndarray,_np.matrix]:
			dense = True
		elif _sp.issparse(other):
			dense = False
		elif ishamiltonian(other):
			return self._add_hamiltonian(other)
		elif isinstance(other,LinearOperator):
			return LinearOperator.__add__(self,other)
		elif _np.isscalar(other):
			return self._mul_scalar(other)
		else:
			dense = True
			other = _np.asanyarray(other)

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
		elif isinstance(other,LinearOperator):
			return LinearOperator.__sub__(self,other)
		elif _np.isscalar(other):
			return self._mul_scalar(other)
		else:
			dense = False
			other = _np.asanyarray(other)

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
		elif isinstance(other,LinearOperator):
			return LinearOperator.__mul__(self,other)
		elif _np.asarray(other).ndim == 0:
			return self._mul_scalar(other)
		else:
			dense = True
			other = _np.asanyarray(other)

		if self.get_shape[1] != other.shape[0]:
			raise ValueError("dimension mismatch with shapes {} and {}".format(self._shape,other.shape))

		if dense:
			if other.ndim == 1:
				return self._matvec(other)
			elif other.ndim == 2:
				return self._matmat(other)
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
		elif isinstance(other,LinearOperator):
			return LinearOperator.__rmul__(self,other)
		elif _np.isscalar(other):
			return self._mul_scalar(other)
		else:
			dense = True
			other = _np.asanyarray(other)

		if dense:
			if other.ndim == 1:
				return self.T._matvec(other)
			elif other.ndim == 2:
				if self._shape[0] != other.shape[1]:
					raise ValueError("dimension mismatch with shapes {0} and {1}".format(self._shape,other.shape))
				return (self.T._matmat(other.T)).T
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
		result_dtype = _np.result_type(self._dtype,other.dtype)
		static = self.__mul__(other._static_list)
		dynamic = [[self.__mul__(Hd),func] for func,Hd in iteritems(other.dynamic)]
		return hamiltonian([static],dynamic,
							basis=self._basis,dtype=result_dtype,copy=False)

	def _mul_sparse(self,other):
		result_dtype = _np.result_type(self._dtype,other.dtype)
		if self.diagonal is not None:
			new_other = _sp.dia_matrix((_np.asarray([self._diagonal]),_np.array([0])),shape=self._shape).dot(other)
			if new_other.dtype != result_dtype:
				new_other = new_other.astype(result_dtype)
		else:
			new_other = _sp.csr_matrix(other.shape,dtype=result_dtype)


		for opstr,indx,J in self.static_list:
			if not self._transposed:
				ME, row, col = self.basis.Op(opstr, indx, J, self._dtype)
			else:
				ME, col, row = self.basis.Op(opstr, indx, J, self._dtype)

			if self._conjugated:
				ME = ME.conj()

			new_other = new_other + _sp.csr_matrix((ME,(row,col)),shape=self._shape).dot(other)

		return new_other

	def _rmul_hamiltonian(self,other):
		result_dtype = _np.result_type(self._dtype,other.dtype)
		static = self.__rmul__(other._static_list)
		dynamic = [[self.__rmul__(Hd),func] for func,Hd in iteritems(other.dynamic)]
		return hamiltonian([static],dynamic,
							basis=self._basis,dtype=result_dtype,copy=False)

	def _add_hamiltonian(self,other):
		return NotImplemented

	def _add_sparse(self,other):
		return NotImplemented

	def _add_dense(self,other):
		return NotImplemented

	def _sub_sparse(self,other):
		return NotImplemented

	def _sub_hamiltonian(self,other):
		return NotImplemented

	def _sub_dense(self,other):
		return NotImplemented

	def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
		# """Method for compatibility with NumPy's ufuncs and dot
		# functions.
		# """

		if (func == _np.dot) or (func == _np.multiply):
			if pos == 0:
				return self.__mul__(inputs[1])
			if pos == 1:
				return self.__rmul__(inputs[0])
			else:
				return NotImplemented


def isquantum_LinearOperator(obj):
	"""Checks if instance is object of `quantum_LinearOperator` class.

	Parameters
	-----------
	obj : 
		Arbitraty python object.

	Returns
	--------
	bool
		Can be either of the following:

		* `True`: `obj` is an instance of `quantum_LinearOperator` class.
		* `False`: `obj` is NOT an instance of `quantum_LinearOperator` class.

	"""
	return isinstance(obj,quantum_LinearOperator)

	