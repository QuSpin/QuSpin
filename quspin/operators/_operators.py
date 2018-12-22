from numba import njit
import scipy.sparse as _sp
import numpy as _np
import oputils as _oputils
import cProfile,memory_profiler

class ShapeError(Exception):
	pass

class operator(object):
	def __init__(self,matrix_obj):
		self._matrix_obj = matrix_obj

	@property
	def shape(self):
		return self._matrix_obj.shape

	@property
	def dtype(self):
		return self._matrix_obj.dtype

	@property
	def T(self):
		return self.transpose()

	@property
	def H(self):
		return self.transpose().conjugate()

	def conjugate(copy=False):
		return self.__class__(self,_matrix_obj.conjugate(copy=copy))

	def conj(copy=False):
		return self.conjugate(copy=copy)

	def dot(self,other,alpha=1.0,out=None,overwrite_out=True):
		alpha = _np.asarray(alpha)
		if isinstance(other,operator) or _sp.isspmatrix(other):
			if out is not None:
				raise ValueError("out can't be used for sparse matrices or operators")

			return alpha * (self._matrix_obj * other)
		else:
			other = _np.asanyarray(other)

			if other.ndim > 2:
				ShapeError("dimension mismatch with {} with input shape {}".format(self.shape,vec_in.shape))

			if other.shape[0] != self.shape[1]:
				raise ShapeError("dimension mismatch with {} with input shape {}".format(self.shape,other.shape))

			if out is None:
				out = _np.zeros(self.shape[:1]+other.shape[1:],
					dtype=_np.result_type(self.dtype,other.dtype,alpha))

			if out.shape[0] != self.shape[0]:
				raise ShapeError("dimension mismatch with {} with output shape {}".format(self.shape,out.shape))

			if out.ndim > 1 and out.shape[1] != other.shape[1]:
				raise ShapeError("'out' shape does not match the correct output.")

			if not _np.can_cast(out.dtype,_np.result_type(self.dtype,other.dtype,alpha)):
				raise TypeError("'out' dtype does not match the correct output.")

			return self._mul_dense(other,alpha,out,overwrite_out)

	def _mul_dense(self,other,alpha,out):
		raise NotImplementedError

	def __mul__(self,other):
		if isinstance(other,operator):
			new_matrix_obj = self._matrix_obj * other._matrix_obj
		else:
			new_matrix_obj = self._matrix_obj * other

		if _sp.isspmatrix_csr(new_matrix_obj):
			return csr_operator(new_matrix_obj)
		elif _sp.isspmatrix_csc(new_matrix_obj):
			return csc_operator(new_matrix_obj)
		else:
			return dense_operator(new_matrix_obj) 

	def __rmul__(self,other):
		if isinstance(other,operator):
			new_matrix_obj =  other._matrix_obj * self._matrix_obj
		else:
			new_matrix_obj = other * self._matrix_obj

		if _sp.isspmatrix_csr(new_matrix_obj):
			return csr_operator(new_matrix_obj)
		elif _sp.isspmatrix_csc(new_matrix_obj):
			return csc_operator(new_matrix_obj)
		else:
			return dense_operator(new_matrix_obj) 

	def __add__(self,other):
		if isinstance(other,operator):
			new_matrix_obj = self._matrix_obj + other._matrix_obj
		else:
			new_matrix_obj = self._matrix_obj + other

		if _sp.isspmatrix_csr(new_matrix_obj):
			return csr_operator(new_matrix_obj)
		elif _sp.isspmatrix_csc(new_matrix_obj):
			return csc_operator(new_matrix_obj)
		else:
			return dense_operator(new_matrix_obj) 

	def __sub__(self,other):
		if isinstance(other,operator):
			new_matrix_obj = self._matrix_obj - other._matrix_obj
		else:
			new_matrix_obj = self._matrix_obj - other

		if _sp.isspmatrix_csr(new_matrix_obj):
			return csr_operator(new_matrix_obj)
		elif _sp.isspmatrix_csc(new_matrix_obj):
			return csc_operator(new_matrix_obj)
		else:
			return dense_operator(new_matrix_obj)

	def __imul__(self,other):
		return self * other

	def __iadd__(self,other):
		return self + other

	def __isub__(self,other):
		return self - other



class csr_operator(operator):
	def __init__(self,matrix_obj):
		operator.__init__(self,_sp.csr_matrix(matrix_obj))

	def transpose(self,copy=False):
		return csc_operator(self._matrix_obj.transpose(copy=copy))

	def _mul_dense(self,other,alpha,out,overwrite_out):
		return _oputils.csr_mv(overwrite_out,self._matrix_obj,other,alpha,out)

class csc_operator(operator):
	def __init__(self,matrix_obj):
		operator.__init__(self,_sp.csc_matrix(matrix_obj))

	def transpose(self,copy=False):
		return csr_operator(self._matrix_obj.transpose(copy=copy))

	def _mul_dense(self,other,alpha,out,overwrite_out):
		return _oputils.csc_mv(overwrite_out,self._matrix_obj,other,alpha,out)

class dense_operator(operator):
	def __init__(self,matrix_obj):
		operator.__init__(self,_np.asmatrix(matrix_obj))

	def transpose(self,copy=False):
		if copy:
			return dense_operator(self._matrix_obj.transpose().copy())
		else:
			return dense_operator(self._matrix_obj.transpose())

	def _mul_dense(self,other,alpha,out,overwrite_out):
		return _oputils.dense_mv(overwrite_out,self._matrix_obj,other,alpha,out)

@memory_profiler.profile
def test(K,N,M,alpha=1.0,atol=1.1e-10):
	A = _sp.random(K,N,format="csr",density=_np.log(N)/N)
	B = _np.random.uniform(-1,1,size=(N,M))
	D = _np.random.uniform(-1,1,size=(K,M))
	E = _np.random.uniform(-1,1,size=N)
	F = _np.random.uniform(-1,1,size=K)

	A_op = csr_operator(A)

	C = alpha * A.dot(B)
	A_op.dot(B,alpha=-alpha,out=C,overwrite_out=False)
	_np.testing.assert_allclose(C,0,rtol=0,atol=atol)

	C = alpha * A.T.dot(D)
	A_op.T.dot(D,alpha=-alpha,out=C,overwrite_out=False)
	# _np.testing.assert_allclose(C,0,rtol=0,atol=atol)


	A_d_op = dense_operator(A.toarray())
	C = alpha * A.dot(B)
	A_d_op.dot(B,alpha=-alpha,out=C,overwrite_out=False)
	# _np.testing.assert_allclose(C,0,rtol=0,atol=atol)

	C = alpha * A.T.dot(D)
	A_d_op.T.dot(D,alpha=-alpha,out=C,overwrite_out=False)
	# _np.testing.assert_allclose(C,0,rtol=0,atol=atol)



test(1000,1000,1000,1j)
