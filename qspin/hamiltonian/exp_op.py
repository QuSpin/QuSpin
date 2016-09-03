from ..hamiltonian import hamiltonian, ishamiltonian
from scipy.sparse.linalg import expm_multiply as _expm_multiply
import numpy as _np
import scipy.sparse as _sp


def isexp_op(obj):
	return isinstance(obj,exp_op)

class exp_op(object):
	def __init__(self,O,a=1.0,time=0.0,start=None, stop=None, num=None, endpoint=None, iterate=False):

		if not _np.isscalar(a):
			raise TypeError('expecting scalar argument for a')


		self._a=a
		self._time=time

		self._start=start
		self._stop=stop
		self._num=num
		self._endpoint=endpoint
		self._iterate=iterate

		if ishamiltonian(O):
			self._O = O
		else:
			self._O = hamiltonian([O], [])
	
		self._ndim=2.0

	@property
	def ndim(self):
	    return self._ndim
	

	@property
	def H(self,copy=False):
		return self.get_H(copy=copy)

	@property
	def O(self):
		return self._O

	@property
	def get_shape(self):
		return self.O.get_shape

	def transpose(self,copy=False):
		self._O=self._O.transpose()
		return self

	def conj(self):
		self._O=self._O.conj
		return self

	def get_H(self,copy=False):
		self._O = self._O.H
		self._a = self._a.conjugate()
		return self


	def expm(self):

		return self._O.expm(a=self._a,time=self._time)
		

	def get_mat(self,time=0.0):

		if self.O.is_dense:
			return _np.linalg.expm(self._a * self.O.todense(time))
		else:
			return _sp.linalg.expm(self._a * self.O.tocsr(time).tocsc())

	def dot(self, other, time=0.0):

		is_sp = False
		is_ham = False

		if ishamiltonian(other):
			shape = other._shape
			is_sp = False
			is_ham = True
		elif _sp.issparse(other):
			shape = other.shape
			is_sp = True
			is_ham = False
		elif other.__class__ in [_np.matrix, _np.ndarray]:
			shape = other.shape
		else:
			other = _np.asanyarray(other)
			shape = other.shape

		if other.ndim not in [1, 2]:
			raise ValueError("Expecting a 1 or 2 dimensional array for 'other'")

		if shape[0] != self.get_shape[1]:
			raise ValueError("Dimension mismatch between expO: {0} and other: {1}".format(self._O.get_shape, other.shape))

		M = self._a * self.O(time)
		if self._iterate:
			if [self._start, self._stop] == [None, None]:
				raise ValueError("iterate option only availible for time discretization.")

			grid, step = _np.linspace(self._start, self._stop, num=self._num, endpoint=self._endpoint, retstep=True)

			if is_ham:
				return _hamiltonian_iter_dot(M, other, step, grid)
			else:
				return _iter_dot(M, other, step, grid)

		else:
			if [self._start, self._stop] == [None, None]:

				if [self._num, self._endpoint] != [None, None]:
					raise ValueError('impropor linspace arguements!')

				if is_ham:
					return _hamiltonian_dot(M, other)		
				else:
					return _expm_multiply(M, other)				
			else:

				if is_sp:
					grid, step = _np.linspace(self._start, self._stop, num=self._num, endpoint=self._endpoint, retstep=True)
					mats = _iter_dot(M, other, step, grid)
					return _np.array([mat for mat in mats])
				elif is_ham:
					grid, step = _np.linspace(self._start, self._stop, num=self._num, endpoint=self._endpoint, retstep=True)
					mats = _hamiltonian_iter_dot(M, other, step, grid)
					return _np.array([mat for mat in mats])				
				else:
					return _expm_multiply(M, other, start=self._start, stop=self._stop, num=self._num, endpoint=self._endpoint)

	def rdot(self, other, time=0.0):

		is_sp = False
		is_ham = False

		if ishamiltonian(other):
			shape = other._shape
			is_sp = False
			is_ham = True
		elif _sp.issparse(other):
			shape = other.shape
			is_sp = True
			is_ham = False
		elif other.__class__ in [_np.matrix, _np.ndarray]:
			shape = other.shape
		else:
			other = _np.asanyarray(other)
			shape = other.shape

		if other.ndim not in [1, 2]:
			raise ValueError("Expecting a 1 or 2 dimensional array for 'other'")

		if shape[1] != self.get_shape[0]:
			raise ValueError("Dimension mismatch between expO: {0} and other: {1}".format(self._O.get_shape, other.shape))

		M = (self._a * self.O(time)).T.conj()
		if self._iterate:
			if [self._start, self._stop] == [None, None]:
				raise ValueError("iterate option only availible for time discretization.")

			grid, step = _np.linspace(self._start, self._stop, num=self._num, endpoint=self._endpoint, retstep=True)

			if is_ham:
				return _hamiltonian_iter_rdot(M, other.T.conj(), step, grid)
			else:
				return _iter_rdot(M, other.T.conj(), step, grid)
		else:
			if [self._start, self._stop] == [None, None]:

				if [self._num, self._endpoint] != [None, None]:
					raise ValueError('impropor linspace arguements!')

				if is_ham:
					return _hamiltonian_rdot(M, other.T.conj()).T.conj()
				else:
					return _expm_multiply(M, other.T.conj()).T.conj()
			else:
				if is_sp:
					grid, step = _np.linspace(self._start, self._stop, num=self._num, endpoint=self._endpoint, retstep=True)
					mats = _iter_rdot(M, other.T.conj(), step, grid)
					return _np.array([mat for mat in mats])
				elif is_ham:
					grid, step = _np.linspace(self._start, self._stop, num=self._num, endpoint=self._endpoint, retstep=True)
					mats = _hamiltonian_iter_rdot(M, other.T.conj(), step, grid)
					return _np.array([mat for mat in mats])				
				else:
					return _expm_multiply(M, other.T.conj(), start=self._start, stop=self._stop, num=self._num, endpoint=self._endpoint).T.conj()

	def sandwich(self, other, time=0.0):

		is_ham = False
		if ishamiltonian(other):
			shape = other._shape
			is_ham = True
		elif _sp.issparse(other):
			shape = other.shape
		elif other.__class__ in [_np.matrix, _np.ndarray]:
			shape = other.shape
		else:
			other = _np.asanyarray(other)
			shape = other.shape

		if other.ndim != 2:
			raise ValueError("Expecting a 2 dimensional array for 'other'")

		if shape[0] != shape[1]:
			raise ValueError("Expecting square array for 'other'")

		if shape[0] != self.get_shape[0]:
			raise ValueError("Dimension mismatch between expO: {0} and other: {1}".format(self.get_shape, other.shape))

		M = self._a*self.O(time)
		if self._iterate:
			if [self._start, self._stop] == [None, None]:
				raise ValueError("iterate option only availible for time discretization.")

			grid, step = _np.linspace(self._start, self._stop, num=self._num, endpoint=self._endpoint, retstep=True)

			if is_ham:
				mat_iter = _hamiltonian_iter_sandwich(M, other, step, grid)
			else:
				mat_iter = _iter_sandwich(M, other, step, grid)

			return mat_iter
		else:
			if [self._start, self._stop] == [None, None]:

				if [self._num, self._endpoint] != [None, None]:
					raise ValueError('impropor linspace arguements!')

				other = self.dot(other,time=time)
				other = self.H.rdot(other,time=time)
				return other

			else:
				grid, step = _np.linspace(self._start, self._stop, num=self._num, endpoint=self._endpoint, retstep=True)

				if is_ham:
					mat_iter = _hamiltonian_iter_sandwich(M, other, step, grid)
				else:
					mat_iter = _iter_sandwich(M, other, step, grid)

				others = _np.asarray([mat for mat in mat_iter])

				return others


def _iter_dot(M, other, step, grid):
	if grid[0] != 0:
		M *= grid[0]
		other = _expm_multiply(M, other)
		M /= grid[0]

	yield other.copy()

	M *= step
	for t in grid[1:]:
		other = _expm_multiply(M, other)
		yield other.copy()


def _iter_rdot(M, other, step, grid):
	if grid[0] != 0:
		M *= grid[0]
		other = _expm_multiply(M, other)
		M /= grid[0]

	yield other.T.conj().copy()

	M *= step
	for t in grid[1:]:
		other = _expm_multiply(M, other)
		yield other.T.conj().copy()


def _iter_sandwich(M, other, step, grid):
	if grid[0] != 0:
		M *= grid[0]
		other = _expm_multiply(M, other)
		r_other = _expm_multiply(M, other.T.conj()).T.conj()
		M /= grid[0]

		yield r_other.copy()
	else:
		yield other.copy()

	for t in grid[1:]:
		M *= step
		other = _expm_multiply(M, other)
		M /= step
		if t != 0:
			M *= t
			r_other = _expm_multiply(M, other.T.conj()).T.conj()
			M /= t

		yield r_other.copy()


def _hamiltonian_dot(M, other):
	exp_static = [_expm_multiply(M, other.static)]
	
	exp_dynamic = []
	for Hd,f,f_args in other.dynamic:
		Hd = _expm_multiply(M, Hd)
		exp_dynamic.append([Hd,f,f_args])

	return hamiltonian(exp_static,exp_dynamic)


def _hamiltonian_iter_dot(M, other, grid, step):
	if grid[0] != 0:
		M *= grid[0]
		other = _hamiltonian_dot(M, other)
		M /= grid[0]

	yield other

	M *= step
	for t in grid[1:]:
		other =  _hamiltonian_dot(M, other)
		yield other


def _hamiltonian_rdot(M, other):
	exp_static = [_expm_multiply(M, other.static)]
	
	exp_dynamic = []
	for Hd,f,f_args in other.dynamic:
		Hd = _expm_multiply(M, Hd)
		exp_dynamic.append([Hd,f,f_args])

	return hamiltonian(exp_static,exp_dynamic)


def _hamiltonian_iter_rdot(M, other, grid, step):
	if grid[0] != 0:
		M *= grid[0]
		other = _hamiltonian_rdot(M, other)
		M /= grid[0]

	yield other.conj().sandwichpose(copy=True)

	M *= step
	for t in grid[1:]:
		other =  _hamiltonian_rdot(M, other)
		yield other.conj().sandwichpose(copy=True)


def _hamiltonian_iter_sandwich(M, other, step, grid):
	if grid[0] != 0:
		M *= grid[0]
		other = _hamiltonian_dot(M, other)
		r_other = _hamiltonian_rdot(M, other.T.conj()).T.conj()
		M /= grid[0]

		yield r_other
	else:
		yield other.copy()

	for t in grid[1:]:
		M *= step
		other = _hamiltonian_dot(M, other)
		M /= step
		if t != 0:
			M *= t
			r_other = _hamiltonian_rdot(M, other.T.conj()).T.conj()
			M /= t

		yield r_other





