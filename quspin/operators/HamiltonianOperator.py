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
	def __init__(self,operator_list,system_arg,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**basis_args):
		"""
		This class uses the basis.Op function to calculate the matrix vector product on the fly greating reducing the amount of memory
		needed for a calculation at the cost of speed. This object is useful for doing large scale Lanczos calculations using eigsh. 

		--- arguments ---

		* static_list: (compulsory) list of operator strings to be used for the HamiltonianOperator. The format goes like:

			```python
			operator_list=[[opstr_1,[indx_11,...,indx_1m]],...]
			```
		
		* system_arg: int/basis_object (compulsory) number of sites to create basis object/basis object.

		* check_symm: bool (optional) flag whether or not to check the operator strings if they obey the given symmetries.

		* check_herm: bool (optional) flag whether or not to check if the operator strings create hermitian matrix. 

		* check_pcon: bool (optional) flag whether or not to check if the oeprator string whether or not they conserve magnetization/particles. 

		* dtype: dtype (optional) data type to case the matrices with. 

		* kw_args: extra options to pass to the basis class.

		--- hamiltonian attributes ---: '_. ' below stands for 'object. '
		* _.basis: the basis associated with this HamiltonianOperator

		* _.ndim: number of dimensions, always 2.
		
		* _.Ns: number of states in the hilbert space.

		* _.shape: returns tuple which has the shape of the hamiltonian (Ns,Ns)

		* _.dtype: returns the data type of the hamiltonian

		* _.operator_list: return the list of operators given to this

		* _.T: return the transpose of this operator

		* _.H: return the hermitian conjugate of this operator

		* _.basis: return the basis used by this operator

		* _.LinearOperator: returns a linear operator of this object


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
		return self._basis

	@property
	def ndim(self):
		return self._ndim

	@property
	def operator_list(self):
		return self._operator_list

	@property
	def get_shape(self):
		return self._shape

	@property
	def Ns(self):
		return self._shape[0]

	@property
	def dtype(self):
		return _np.dtype(self._dtype).name

	@property
	def basis(self):
		return self._basis

	@property
	def T(self):
		return self.transpose(copy = False)

	@property
	def H(self):
		return self.getH(copy = False)

	@property
	def LinearOperator(self):
		return self.get_LinearOperator()

	def copy(self):
		return _deepcopy(self)


	def transpose(self,copy = False):
		if copy:
			return self.copy().transpose()
		else:
			self._transposed = not self._transposed
			return self

	def conj(self):
		self._conjugated = not self._conjugated
		return self

	def getH(self,copy = False):
		if copy:
			return self.copy().get_H()
		else:
			return self.conj().transpose()

	def __repr__(self):
		return "<{0}x{1} qspin HamiltonianOperator of type '{2}'>".format(*(self._shape[0],self._shape[1],self._dtype))

	def get_LinearOperator(self):
		return self._LinearOperator

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

	def dot(self,other):
		return self.__mul__(other)

	def rdot(self,other):
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

	def eigsh(self,**eigsh_args):
		return _sla.eigsh(self.LinearOperator,**eigsh_args)

	def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
		"""Method for compatibility with NumPy's ufuncs and dot
		functions.
		"""

		if (func == np.dot) or (func == np.multiply):
			if pos == 0:
				return self.__mul__(inputs[1])
			if pos == 1:
				return self.__rmul__(inputs[0])
			else:
				return NotImplemented


def isHamiltonianOperator(obj):
	return isinstance(obj,HamiltonianOperator)

	