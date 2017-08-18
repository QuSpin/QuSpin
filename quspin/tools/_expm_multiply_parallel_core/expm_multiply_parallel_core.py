from scipy.sparse.linalg import LinearOperator,onenormest,aslinearoperator
from .expm_multiply_parallel_wrapper import _wrapper_expm_multiply,_wrapper_csr_trace
import scipy.sparse as _sp
import numpy as _np

class expm_multiply_parallel(object):
	"""

	"""
	def __init__(self,A,a=1.0):
		"""

		"""
		self._a = a
		self._A = _sp.csr_matrix(A,copy=False)

		if A.shape[0] != A.shape[1]:
			raise ValueError("A must be a square matrix.")

		self._tol = _np.finfo(A.dtype).eps

		self._mu = _wrapper_csr_trace(self._A.indptr,self._A.indices,self._A.data)/self._A.shape[0]
		self._A -= self._mu * _sp.identity(self._A.shape[0],dtype=self._A.dtype,format="csr")
		self._A_1_norm = _np.max(_np.abs(A).sum(axis=0))

		if _np.abs(self._a)*self._A_1_norm == 0:
			self._m_star, self._s = 0, 1
		else:
			ell = 2
			self._norm_info = LazyOperatorNormInfo(self._A, self._A_1_norm, self._a, ell=ell)
			self._m_star, self._s = _fragment_3_1(self._norm_info, 1, self._tol, ell=ell)


	def dot(self,v,work=None,overwrite_v=False):
		"""

		"""
		if not overwrite_v:
			v = _np.ascontiguousarray(v).copy()
		else:
			v = _np.asanyarray(v)

		if v.ndim != 1:
			raise ValueError

		if v.shape[0] != self._A.shape[1]:
			raise ValueError("dimension mismatch {}, {}".format(self._A.shape,v.shape))

		if work is None:
			work = _np.zeros((2*self._A.shape[0],),dtype=v.dtype)
		else:
			work = _np.ascontiguousarray(work)
			if work.shape != (2*self._A.shape[0],):
				raise ValueError("work array must be an array of shape (2*v.shape[0],) with same dtype as v.")
			if work.dtype != v.dtype:
				raise ValueError("work array must be the same dtype as i_nput vector v.")

		_wrapper_expm_multiply(self._A.indptr,self._A.indices,self._A.data,
					self._m_star,self._s,self._a,self._tol,self._mu,v,work)

		return v




# This table helps to compute bounds.
# They seem to have been difficult to calculate, involving symbolic
# manipulation of equations, followed by numerical root finding.
_theta = {
		# The first 30 values are from table A.3 of Computing Matrix Functions.
		1: 2.29e-16,
		2: 2.58e-8,
		3: 1.39e-5,
		4: 3.40e-4,
		5: 2.40e-3,
		6: 9.07e-3,
		7: 2.38e-2,
		8: 5.00e-2,
		9: 8.96e-2,
		10: 1.44e-1,
		# 11
		11: 2.14e-1,
		12: 3.00e-1,
		13: 4.00e-1,
		14: 5.14e-1,
		15: 6.41e-1,
		16: 7.81e-1,
		17: 9.31e-1,
		18: 1.09,
		19: 1.26,
		20: 1.44,
		# 21
		21: 1.62,
		22: 1.82,
		23: 2.01,
		24: 2.22,
		25: 2.43,
		26: 2.64,
		27: 2.86,
		28: 3.08,
		29: 3.31,
		30: 3.54,
		# The rest are from table 3.1 of
		# Computing the Action of the Matrix Exponential.
		35: 4.7,
		40: 6.0,
		45: 7.2,
		50: 8.5,
		55: 9.9,
		}


class ScaledMatrixPowerOperator(LinearOperator):

	def __init__(self, A, p, a):
		if A.ndim != 2 or A.shape[0] != A.shape[1]:
			raise ValueError('expected A to be like a square matrix')
		if p < 0:
			raise ValueError('expected p to be a non-negative integer')
		self._A = A
		self._p = p
		self._a = a
		self.ndim = A.ndim
		self.shape = A.shape
		self.dtype = A.dtype


	def _matvec(self, x):
		for i in range(self._p):
			x = self._A.dot(x)
		x *= self._a**self._p
		return x

	def _rmatvec(self, x):
		for i in range(self._p):
			x = x.dot(self._A)
		x *= self._a**self._p
		return x

	def _matmat(self, X):
		for i in range(self._p):
			X =  self._A.dot(X)
		X *= self._a**self._p
		return X

	@property
	def T(self):
		return ScaledMatrixPowerOperator(self._A.T, self._p, self._a)


def _onenormest_matrix_power(A, p, a,
		t=2, itmax=5, compute_v=False, compute_w=False):
	"""
	Efficiently estimate the 1-norm of A^p.

	Parameters
	----------
	A : ndarray
		Matrix whose 1-norm of a power is to be computed.
	p : int
		Non-negative integer power.
	t : int, optional
		A positive parameter controlling the tradeoff between
		accuracy versus time and memory usage.
		Larger values take longer and use more memory
		but give more accurate output.
	itmax : int, optional
		Use at most this many iterations.
	compute_v : bool, optional
		Request a norm-maximizing linear operator i_nput vector if True.
	compute_w : bool, optional
		Request a norm-maximizing linear operator output vector if True.

	Returns
	-------
	est : float
		An underestimate of the 1-norm of the sparse matrix.
	v : ndarray, optional
		The vector such that ||Av||_1 == est*||v||_1.
		It can be thought of as an i_nput to the linear operator
		that gives an output with particularly large norm.
	w : ndarray, optional
		The vector Av which has relatively large 1-norm.
		It can be thought of as an output of the linear operator
		that is relatively large in norm compared to the i_nput.

	"""
	return onenormest((a*aslinearoperator(A))**p)


class LazyOperatorNormInfo:
	"""
	Information about an operator is lazily computed.

	The information includes the exact 1-norm of the operator,
	in addition to estimates of 1-norms of powers of the operator.
	This uses the notation of Computing the Action (2011).
	This class is specialized enough to probably not be of general interest
	outside of this module.

	"""
	def __init__(self, A, A_1_norm, a, ell=2):
		"""
		Provide the operator and some norm-related information.

		Parameters
		----------
		A : linear operator
			The operator of interest.
		A_1_norm : float
			The exact 1-norm of A.
		ell : int, optional
			A technical parameter controlling norm estimation quality.

		"""
		self._A = A
		self._a = a
		self._A_1_norm = A_1_norm
		self._ell = ell
		self._d = {}

	def onenorm(self):
		"""
		Compute the exact 1-norm.
		"""
		return _np.abs(self._a) * self._A_1_norm

	def d(self, p):
		"""
		Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.
		"""
		if p not in self._d:
			est = _onenormest_matrix_power(self._A, p, self._a,  self._ell)
			self._d[p] = est ** (1.0 / p)
		return self._d[p]

	def alpha(self, p):
		"""
		Lazily compute max(d(p), d(p+1)).
		"""
		return max(self.d(p), self.d(p+1))


def _compute_cost_div_m(m, p, norm_info):
	"""
	A helper function for computing bounds.

	This is equation (3.10).
	It measures cost in terms of the number of required matrix products.

	Parameters
	----------
	m : int
		A valid key of _theta.
	p : int
		A matrix power.
	norm_info : LazyOperatorNormInfo
		Information about 1-norms of related operators.

	Returns
	-------
	cost_div_m : int
		Required number of matrix products divided by m.

	"""
	return int(_np.ceil(norm_info.alpha(p) / _theta[m]))


def _compute_p_max(m_max):
	"""
	Compute the largest positive integer p such that p*(p-1) <= m_max + 1.

	Do this in a slightly dumb way, but safe and not too slow.

	Parameters
	----------
	m_max : int
		A count related to bounds.

	"""
	sqrt_m_max = _np.sqrt(m_max)
	p_low = int(_np.floor(sqrt_m_max))
	p_high = int(_np.ceil(sqrt_m_max + 1))
	return max(p for p in range(p_low, p_high+1) if p*(p-1) <= m_max + 1)


def _fragment_3_1(norm_info, n0, tol, m_max=55, ell=2):
	"""
	A helper function for the _expm_multiply_* functions.

	Parameters
	----------
	norm_info : LazyOperatorNormInfo
		Information about norms of certain linear operators of interest.
	n0 : int
		Number of columns in the _expm_multiply_* B matrix.
	tol : float
		Expected to be
		:math:`2^{-24}` for single precision or
		:math:`2^{-53}` for double precision.
	m_max : int
		A value related to a bound.
	ell : int
		The number of columns used in the 1-norm approximation.
		This is usually taken to be small, maybe between 1 and 5.

	Returns
	-------
	best_m : int
		Related to bounds for error control.
	best_s : int
		Amount of scaling.

	Notes
	-----
	This is code fragment (3.1) in Al-Mohy and Higham (2011).
	The discussion of default values for m_max and ell
	is given between the definitions of equation (3.11)
	and the definition of equation (3.12).

	"""
	if ell < 1:
		raise ValueError('expected ell to be a positive integer')
	best_m = None
	best_s = None
	if _condition_3_13(norm_info.onenorm(), n0, m_max, ell):
		for m, theta in _theta.items():
			s = int(_np.ceil(norm_info.onenorm() / theta))
			if best_m is None or m * s < best_m * best_s:
				best_m = m
				best_s = s
	else:
		# Equation (3.11).
		for p in range(2, _compute_p_max(m_max) + 1):
			for m in range(p*(p-1)-1, m_max+1):
				if m in _theta:
					s = _compute_cost_div_m(m, p, norm_info)
					if best_m is None or m * s < best_m * best_s:
						best_m = m
						best_s = s
		best_s = max(best_s, 1)
	return best_m, best_s


def _condition_3_13(A_1_norm, n0, m_max, ell):
	"""
	A helper function for the _expm_multiply_* functions.

	Parameters
	----------
	A_1_norm : float
		The precomputed 1-norm of A.
	n0 : int
		Number of columns in the _expm_multiply_* B matrix.
	m_max : int
		A value related to a bound.
	ell : int
		The number of columns used in the 1-norm approximation.
		This is usually taken to be small, maybe between 1 and 5.

	Returns
	-------
	value : bool
		Indicates whether or not the condition has been met.

	Notes
	-----
	This is condition (3.13) in Al-Mohy and Higham (2011).

	"""

	# This is the rhs of equation (3.12).
	p_max = _compute_p_max(m_max)
	a = 2 * ell * p_max * (p_max + 3)

	# Evaluate the condition (3.13).
	b = _theta[m_max] / float(n0 * m_max)
	return A_1_norm <= a * b
