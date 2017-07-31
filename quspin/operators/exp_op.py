from __future__ import print_function, division


import hamiltonian
import ops_dict

# need linear algebra packages
import scipy
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

from scipy.sparse.linalg import expm_multiply as _expm_multiply

from copy import deepcopy as _deepcopy # recursively copies all data into new object
from copy import copy as _shallowcopy

from six import iteritems

__all__ = ["exp_op","isexp_op"]

class exp_op(object):
	def __init__(self, O, a = 1.0, time = 0.0, start = None, stop = None, num = None, endpoint = None, iterate = False):
		"""
		This class constructs an object which acts on various objects with the matrix exponential of the matrix/hamiltonian ```O```. 
		It does not calculate the actual matrix exponential but instead computes the action of the matrix exponential through 
		the taylor series. This is slower but for sparse arrays this is more memory efficient. All of the functions make use of the 
		expm_multiply function in Scipy's sparse library. This class also allows the option to specify a grid of points on a line in 
		the complex plane via the optional arguments. if this is specified then an array `grid` is created via the numpy function 
		linspace, then every time a math function is called the exponential is evaluated with for `a*grid[i]*O`.

		--- arguments ---

		* H: matrix/hamiltonian (compulsory), The operator which to be exponentiated.

		* a: scalar (optional), prefactor to go in front of the operator in the exponential: exp(a*O)

		* start: scalar (optional), specify the starting point for a grid of points to evaluate the matrix exponential at.

		* stop: (optional), specify the end point of the grid of points. 

		* num: (optional), specify the number of grid points between start and stop (Default if 50)

		* endpoint: (optional), if True this will make sure stop is included in the set of grid points (Note this changes the grid step size).

		* iterate: (optional), if True when mathematical methods are called they will return iterators which will iterate over the grid 
		 points as opposed to producing a list of all the evaluated points. This is more memory efficient but at the sacrifice of speed.

		--- exp_op attributes ---: '_. ' below stands for 'object. '

		* _.ndim: returns the number of dimensions, always 2

		* _.a: returns the prefactor a

		* _.H: returns the hermitian conjugate of this operator

		* _.T: returns the transpose of this operator

		* _.O: returns the operator which is being exponentiated

		* _.get_shape: returns the tuple which contains the shape of the operator

		* _.iterate: returns a bool telling whether or not this function will iterate over the grid of values or return a list

		* _.grid: returns the array containing the grid points the exponential will be evaluated at

		* _.step: returns the step size between the grid points


		"""
		if _np.array(a).ndim > 0:
			raise TypeError('expecting scalar argument for a')

		self._a = a
		self._time = time

		self._start = start
		self._stop = stop
		self._num = num
		self._endpoint = endpoint
		self._iterate = iterate
		if self._iterate:
			if self._start is None and self._stop is None:
				raise ValueError("'iterate' can only be True with time discretization. must specify 'start' and 'stop' points.")

			if num is not None:
				if type(num) is not int:
					raise ValueError("expecting integer for 'num'.")
			else:
				num = 50
				self._num = num

			if endpoint is not None:
				if type(endpoint) is not bool:
					raise ValueError("expecting bool for 'endpoint'.")
			else: 
				endpoint = True
				self._endpoint = endpoint

			self._grid, self._step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)
		else:
			if self._start is None and self._stop is None:
				if self._num != None:
					raise ValueError("unexpected argument 'num'.")

				if self._endpoint != None:
					raise ValueError("unexpected argument 'endpoint'.")

				self._grid = None
				self._step = None
			else:

				if not (_np.isscalar(start) and _np.isscalar(stop)):
					raise ValueError("expecting scalar values for 'start' and 'stop'")

				if not (_np.isreal(start) and _np.isreal(stop)):
					raise ValueError("expecting real values for 'start' and 'stop'")

				if num is not None:
					if type(num) is not int:
						raise ValueError("expecting integer for 'num'.")
				else:
					num = 50
					self._num = num

				if endpoint is not None:
					if type(endpoint) is not bool:
						raise ValueError("expecting bool for 'endpoint'.")
				else: 
					endpoint = True
					self._endpoint = endpoint

				self._grid, self._step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)


		if hamiltonian.ishamiltonian(O):
			self._O = O
		elif ops_dict.isops_dict(O):
			self._O = O.tohamiltonian()
		else:
			if _sp.issparse(O) or O.__class__ in [_np.ndarray,_np.matrix]:
				self._O = hamiltonian.hamiltonian([O], [],dtype=O.dtype)
			else:
				O = _np.asanyarray(O)
				self._O = hamiltonian.hamiltonian([O],[],dtype=O.dtype)
	
		self._ndim = 2

	@property
	def ndim(self):
		return self._ndim
	
	@property
	def H(self):
		return self.getH(copy = False)

	@property
	def T(self):
		return self.transpose(copy = False)
	
	@property
	def H(self):
		return self.getH(copy = False)

	@property
	def O(self):
		return self._O

	@property
	def a(self):
		return self._a

	@property
	def get_shape(self):
		return self.O.get_shape

	@property
	def Ns(self):
		return self.O.Ns

	@property
	def grid(self):
		return self._grid

	@property
	def step(self):
		return self._step

	def transpose(self,copy = False):
		if copy:
			return self.copy().transpose(copy = False)
		else:
			self._O=self._O.transpose()
			return self

	def conj(self):
		self._O=self._O.conj()
		self._a = self._a.conjugate()
		return self

	def getH(self,copy = False):
		if copy:
			return self.copy().getH(copy = False)
		else:
			self._O = self._O.getH(copy = False)
			self._a = self._a.conjugate()
			return self

	def copy(self):
		return _deepcopy(self)

	def set_a(self,new_a):
		if not _np.isscalar(new_a):
			raise ValueError("'a' must be set to scalar value.")
		self._a = _np.complex128(new_a)

	def set_grid(self, start, stop, num = None, endpoint = None):

		if not (_np.isscalar(start) and _np.isscalar(stop)):
			raise ValueError("expecting scalar values for 'start' and 'stop'")

		if not (_np.isreal(start) and _np.isreal(stop)):
			raise ValueError("expecting real values for 'start' and 'stop'")

		if type(num) is not None:
			if type(num) is not int:
				raise ValueError("expecting integer for 'num'.")

		if type(endpoint) is not None:
			if type(endpoint) is not bool:
				raise ValueError("expecting bool for 'endpoint'.")

		self._start=start
		self._stop=stop
		self._num=num
		self._endpoint=endpoint
		self._grid, self._step = _np.linspace(start, stop, num = num, endpoint = endpoint, retstep = True)

	def unset_grid(self):
		self._iterate=False
		self._start=None
		self._stop=None
		self._num=None
		self._endpoint=None
		self._grid, self._step = None, None

	def set_iterate(self,Value):
		if type(Value) is not bool:
			raise ValueError("iterate option must be true or false.")

		if Value:
			if (self._grid, self._step) is (None, None):
				raise ValueError("grid must be set in order to set iterate to be True.")

		self._iterate = Value
		

	def get_mat(self,time=0.0,dense=False):

		if self.O.is_dense or dense:
			return _la.expm(self._a * self.O.todense(time))
		else:
			return _la.expm(self._a * self.O.tocsc(time))


	def dot(self, other, time=0.0, shift=None):

		is_sp = False
		is_ham = False

		if hamiltonian.ishamiltonian(other):
			shape = other._shape
			is_ham = True
		elif _sp.issparse(other):
			shape = other.shape
			is_sp = True
		elif other.__class__ in [_np.matrix, _np.ndarray]:
			shape = other.shape
		else:
			other = _np.asanyarray(other)
			shape = other.shape

		if other.ndim not in [1, 2]:
			raise ValueError("Expecting a 1 or 2 dimensional array for 'other'")

		if shape[0] != self.get_shape[1]:
			raise ValueError("Dimension mismatch between expO: {0} and other: {1}".format(self._O.get_shape, other.shape))

		if shift is not None:
			M = self._a * (self.O(time) + shift*_sp.identity(self.Ns,dtype=self.O.dtype))
		else:
			M = self._a * self.O(time)

		if self._iterate:
			if is_ham:
				return _hamiltonian_iter_dot(M, other, self._step, self._grid)
			else:
				return _iter_dot(M, other, self.step, self._grid)

		else:
			if self._grid is None and self._step is None:
				if is_ham:
					return _hamiltonian_dot(M, other)
				else:
					return _expm_multiply(M, other)
			else:
				if is_sp:
					mats = _iter_dot(M, other, self._step, self._grid)
					return _np.array([mat for mat in mats])
				elif is_ham:
					mats = _hamiltonian_iter_dot(M, other, self._step, self._grid)
					return _np.array([mat for mat in mats])
				else:
					ver = [int(v) for v in scipy.__version__.split(".")]

					if _np.iscomplexobj(_np.float32(1.0).astype(M.dtype)) and ver[1] < 19:
						mats = _iter_dot(M, other, self._step, self._grid)
						return _np.array([mat for mat in mats]).T
					else:
						return _expm_multiply(M, other, start=self._start, stop=self._stop, num=self._num, endpoint=self._endpoint).T

	def rdot(self, other, time=0.0,shift=None):

		is_sp = False
		is_ham = False

		if hamiltonian.ishamiltonian(other):
			shape = other._shape
			is_ham = True
		elif _sp.issparse(other):
			shape = other.shape
			is_sp = True
		elif other.__class__ in [_np.matrix, _np.ndarray]:
			shape = other.shape
		else:
			other = _np.asanyarray(other)
			shape = other.shape

		if other.ndim not in [1, 2]:
			raise ValueError("Expecting a 1 or 2 dimensional array for 'other'")

		if shape[1] != self.get_shape[0]:
			raise ValueError("Dimension mismatch between expO: {0} and other: {1}".format(self._O.get_shape, other.shape))

		if shift is not None:
			M = (self._a * (self.O(time) + shift*_sp.identity(self.Ns,dtype=self.O.dtype))).T
		else:
			M = (self._a * self.O(time)).T

		if self._iterate:
			if is_ham:
				return _hamiltonian_iter_rdot(M, other.T, self._step, self._grid)
			else:
				return _iter_rdot(M, other.T, self._step, self._grid)
		else:
			if self._grid is None and self._step is None:

				if is_ham:
					return _hamiltonian_rdot(M, other.T).T
				else:
					return _expm_multiply(M, other.T).T
			else:
				if is_sp:
					mats = _iter_rdot(M, other.T, self._step, self._grid)
					return _np.array([mat for mat in mats])
				elif is_ham:
					mats = _hamiltonian_iter_rdot(M, other.T, self._step, self._grid)
					return _np.array([mat for mat in mats])
				else:
					ver = [int(v) for v in scipy.__version__.split(".")]
					if _np.iscomplexobj(_np.float32(1.0).astype(M.dtype)) and ver[1] < 19:
						mats = _iter_rdot(M, other.T, self._step, self._grid)
						return _np.array([mat for mat in mats])
					else:
						if other.ndim > 1:
							return _expm_multiply(M, other.T, start=self._start, stop=self._stop, num=self._num, endpoint=self._endpoint).transpose(0,2,1)
						else:
							return _expm_multiply(M, other.T, start=self._start, stop=self._stop, num=self._num, endpoint=self._endpoint)


	def sandwich(self, other, time=0.0,shift=None):

		is_ham = False
		if hamiltonian.ishamiltonian(other):
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

		if shift is not None:
			M = self._a.conjugate() * (self.O.H(time) + shift*_sp.identity(self.Ns,dtype=self.O.dtype))
		else:
			M = self._a.conjugate() * self.O.H(time)
			
		if self._iterate:

			if is_ham:
				mat_iter = _hamiltonian_iter_sandwich(M, other, self._step, self._grid)
			else:
				mat_iter = _iter_sandwich(M, other, self._step, self._grid)

			return mat_iter
		else:
			if self._grid is None and self._step is None:

				other = self.dot(other,time=time)
				other = self.H.rdot(other,time=time)
				return other

			else:
				if is_ham:
					mat_iter = _hamiltonian_iter_sandwich(M, other, self._step, self._grid)
					return _np.asarray([mat for mat in mat_iter])
				else:
					mat_iter = _iter_sandwich(M, other, self._step, self._grid)
					return _np.asarray([mat for mat in mat_iter]).transpose((1,2,0))


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

	yield other.T.copy()

	M *= step
	for t in grid[1:]:
		other = _expm_multiply(M, other)
		yield other.T.copy()


def _iter_sandwich(M, other, step, grid):
	if grid[0] != 0:
		M *= grid[0]
		other = _expm_multiply(M, other)
		other = _expm_multiply(M, other.T.conj()).T.conj()
		M /= grid[0]

	yield other.copy()

	M *= step
	for t in grid[1:]:
		other = _expm_multiply(M, other)
		other = _expm_multiply(M, other.T.conj()).T.conj()

		yield other.copy()


def _hamiltonian_dot(M, other):
	new = _shallowcopy(other)
	new._static = _expm_multiply(M, other.static)
	new._dynamic = {func:_expm_multiply(M, Hd) for func,Hd in iteritems(other._dynamic)}

	return new


def _hamiltonian_iter_dot(M, other, grid, step):
	if grid[0] != 0:
		M *= grid[0]
		other = _hamiltonian_dot(M, other)
		M /= grid[0]

	yield other

	M *= step
	for t in grid[1:]:
		other = _hamiltonian_dot(M, other)
		yield other


def _hamiltonian_rdot(M, other):
	new = _shallowcopy(other)
	new._static = _expm_multiply(M, other.static)
	new._dynamic = {func:_expm_multiply(M, Hd) for func,Hd in iteritems(other._dynamic)}

	return new


def _hamiltonian_iter_rdot(M, other, grid, step):
	if grid[0] != 0:
		M *= grid[0]
		other = _hamiltonian_rdot(M, other)
		M /= grid[0]

	yield other.transpose(copy=True)

	M *= step
	for t in grid[1:]:
		other = _hamiltonian_rdot(M, other)
		yield other.transpose(copy=True)


def _hamiltonian_iter_sandwich(M, other, step, grid):
	if grid[0] != 0:
		M *= grid[0]
		other = _hamiltonian_dot(M, other)
		other = _hamiltonian_dot(M, other.T.conj()).T.conj()
		M /= grid[0]

	yield other.copy()

	M *= step
	for t in grid[1:]:
		other = _hamiltonian_dot(M, other)
		other = _hamiltonian_dot(M, other.T.conj()).T.conj()
		yield other.copy()


def isexp_op(obj):
	return isinstance(obj,exp_op)


	