from __future__ import print_function, division

from ..basis import spin_basis_1d as _default_basis
from ..basis import isbasis as _isbasis

import exp_op

from .make_hamiltonian import make_static as _make_static
from .make_hamiltonian import make_dynamic as _make_dynamic
from .make_hamiltonian import test_function as _test_function
from ._functions import function

# need linear algebra packages
import scipy
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

from operator import mul
import functools
from six import iteritems,itervalues,viewkeys


from copy import deepcopy as _deepcopy # recursively copies all data into new object
from copy import copy as _shallowcopy # copies only at top level references the data of old objects
import warnings


__all__ = ["commutator","anti_commutator","hamiltonian","ishamiltonian"]

def commutator(H1,H2):
	""" Calculates the commutator of two Hamiltonians :math:`H_1` and :math:`H_2`.

	.. math::
		[H_1,H_2] = H_1 H_2 - H_2 H_1

	Parameters
	----------
	H1 : obj
		`numpy.ndarray` or `hamiltonian` class object to define the Hamiltonian operator as a matrix.
	H2 : obj
		`numpy.ndarray` or `hamiltonian` class object to define the Hamiltonian operator as a matrix. 

	Return
	------
	obj
		Commutator: :math:`[H_1,H_2] = H_1 H_2 - H_2 H_1`
	"""
	if ishamiltonian(H1) or ishamiltonian(H2):
		return H1*H2 - H2*H1
	else:
		return H1.dot(H2) - H2.dot(H1)

def anti_commutator(H1,H2):
	""" Calculates the anti-commutator of two Hamiltonians :math:`H_1` and :math:`H_2`.

	.. math::
		\\{H_1,H_2\\}_+ = H_1 H_2 + H_2 H_1

	Parameters
	----------
	H1 : obj
		`numpy.ndarray` or `hamiltonian` class object to define the Hamiltonian operator as a matrix.
	H2 : obj
		`numpy.ndarray` or `hamiltonian` class object to define the Hamiltonian operator as a matrix. 
	
	Return
	------
	obj
		Anticommutator: :math:`\\{H_1,H_2\\}_+ = H_1 H_2 + H_2 H_1`

	"""
	if ishamiltonian(H1) or ishamiltonian(H2):
		return H1*H2 + H2*H1
	else:
		return H1.dot(H2) + H2.dot(H1)


class HamiltonianEfficiencyWarning(Warning):
	pass


#global names:
supported_dtypes=tuple([_np.float32, _np.float64, _np.complex64, _np.complex128])

def _check_static(sub_list):
	"""Checks format of static list. """
	if (type(sub_list) in [list,tuple]) and (len(sub_list) == 2):
		if type(sub_list[0]) is not str: raise TypeError('expecting string type for opstr')
		if type(sub_list[1]) in [list,tuple]:
			for sub_sub_list in sub_list[1]:
				if (type(sub_sub_list) in [list,tuple]) and (len(sub_sub_list) > 0):
					for element in sub_sub_list:
						if not _np.isscalar(element): raise TypeError('expecting scalar elements of indx')
				else: raise TypeError('expecting list for indx') 
		else: raise TypeError('expecting a list of one or more indx')
		return True
	else: 
		return False
	

def _check_dynamic(sub_list):
	"""Checks format of dynamic list. """
	if (type(sub_list) in [list,tuple]) and (len(sub_list) == 4):
		if type(sub_list[0]) is not str: raise TypeError('expecting string type for opstr')
		if type(sub_list[1]) in [list,tuple]:
			for sub_sub_list in sub_list[1]:
				if (type(sub_sub_list) in [list,tuple]) and (len(sub_sub_list) > 0):
					for element in sub_sub_list:
						if not _np.isscalar(element): raise TypeError('expecting scalar elements of indx')
				else: raise TypeError('expecting list for indx') 
		else: raise TypeError('expecting a list of one or more indx')
		if not hasattr(sub_list[2],"__call__"): raise TypeError('expecting callable object for driving function')
		if type(sub_list[3]) not in [list,tuple]: raise TypeError('expecting list for function arguments')
		return True
	elif (type(sub_list) in [list,tuple]) and (len(sub_list) == 3): 
		if not hasattr(sub_list[1],"__call__"): raise TypeError('expecting callable object for driving function')
		if type(sub_list[2]) not in [list,tuple]: raise TypeError('expecting list for function arguments')
		return False
	else:
		raise TypeError('expecting list with object, driving function, and function arguments')


def _check_almost_zero(matrix):
	""" Check if matrix is almost zero. """
	atol = 100*_np.finfo(matrix.dtype).eps

	if _sp.issparse(matrix):
		return _np.allclose(matrix.data,0,atol=atol)
	else:
		return _np.allclose(matrix,0,atol=atol)



def _hamiltonian_dot(hamiltonian,time,v):
	"""Used to create linear operator of a hamiltonian."""
	return hamiltonian.dot(v,time=time,check=False)

class hamiltonian(object):
	""" Construct quantum operators.

	The hamiltonian class wraps most of the functionalty of the package. This object allows the user to construct 
	lattice Hamiltonians and operators, solve the time-dependent Schroedinger equation, do full/Lanczos 
	diagonalization, etc.

	The user can create both static and time-dependent, hermitian and non-hermitian operators for a particle
	type (e.g. boson, spin, fermion) specified by the basis constructor.

	Note
	----
	Once can instantiate the class either by parsing a set of symmetries, or an instance of `basis`. Note that
	instantiation with a `basis` will automatically ignore all symmetry inputs. 

	Example
	-------

	Here is an example how to construct the periodically driven XXZ Hamiltonian using a `basis` object
	
	.. math::
		H(t) = \\sum_{j=0}^{L-1} \\left( JS^z_{j+1}S^z_j + hS^z_j + g\cos(\\Omega t)S^x_j \\right)

	in the zero-momentum sector (`kblock=0`) of positive parity (`pblock=1`). We use periodic boundary conditions.
 
	>>> from quspin.operators import hamiltonian # Hamiltonians and operators
	>>> from quspin.basis import spin_basis_1d # Hilbert space spin basis
	>>> import numpy as np # generic math functions
	>>> #
	>>> ##### define model parameters #####
	>>> L=6 # system size
	>>> J=1.0 # spin interaction
	>>> g=0.809 # transverse field
	>>> h=0.9045 # parallel field
	>>> ##### define periodic drive #####
	>>> Omega=4.5 # drive frequency
	>>> def drive(t,Omega):
	>>> 	return np.cos(Omega*t)
	>>> drive_args=[Omega]
	>>> #
	>>> ##### construct basis in the 0-total momentum and +1-parity sector
	>>> basis=spin_basis_1d(L=L,a=1,kblock=0,pblock=1)
	>>> # define PBC site-coupling lists for operators
	>>> x_field=[[g,i] for i in range(L)]
	>>> z_field=[[h,i] for i in range(L)]
	>>> J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
	>>> # static and dynamic lists
	>>> static=[["zz",J_nn],["z",z_field]]
	>>> dynamic=[["x",x_field,drive,drive_args]]
	>>> ###### construct Hamiltonian
	>>> H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
	>>> print(H.toarray())

	"""
	def __init__(self,static_list,dynamic_list,N=None,dtype=_np.complex128,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,**kwargs):
		"""Intializes the `hamtilonian` object (any quantum operator).

		Parameters
		----------
		static_list : 
			List of objects to calculate the static part of a `hamiltonian` operator. The format goes like:

			>>> static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]
			
		dynamic_list : 
			List of objects to calculate the dynamic (time-dependent) part of a `hamiltonian` operator.
			The format goes like:

			>>> dynamic_list=[[opstr_1,[indx_11,...,indx_1n],fun_1,fun_1_args],[matrix_2,fun_2,fun_2_args],...]
			
			* `fun`: function object which multiplies the matrix or operator given in the same list.
			* `func_args`: tuple of the extra arguments which go into the function to evaluate it like: 

				>>> f_val = func(t,*func_args)

			If the operator is time-INdependent, one must pass an empty list: `dynamic_list = []`.
		N : int, optional
			Number of lattice sites for the `hamiltonian` object.
		dtype : numpy.datatype, optional
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
			to create the operator (see Example).
		 
		"""

		self._is_dense = False
		self._ndim = 2
		self._basis = None



		if not (dtype in supported_dtypes):
			raise TypeError('hamiltonian does not support type: '+str(dtype))
		else:
			self._dtype=dtype
		


		if type(static_list) in [list,tuple]:
			static_opstr_list=[]
			static_other_list=[]
			for ele in static_list:
				if _check_static(ele):
					static_opstr_list.append(ele)
				else:
					static_other_list.append(ele)
		else: 
			raise TypeError('expecting list/tuple of lists/tuples containing opstr and list of indx')

		if type(dynamic_list) in [list,tuple]:
			dynamic_opstr_list=[]
			dynamic_other_list=[]
			for ele in dynamic_list:
				if _check_dynamic(ele):
					dynamic_opstr_list.append(ele)
				else: 
					dynamic_other_list.append(ele)					
		else: 
			raise TypeError('expecting list/tuple of lists/tuples containing opstr and list of indx, functions, and function args')

		# need for check_symm
		self._static_opstr_list = static_opstr_list
		self._dynamic_opstr_list = dynamic_opstr_list


		# if any operator strings present must get basis.
		if static_opstr_list or dynamic_opstr_list:
			# check if user input basis
			basis=kwargs.get('basis')

			if basis is not None:
				kwargs.pop('basis')
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

			if check_herm:
				basis.check_hermitian(static_opstr_list, dynamic_opstr_list)

			if check_symm:
				basis.check_symm(static_opstr_list,dynamic_opstr_list)

			if check_pcon:
				basis.check_pcon(static_opstr_list,dynamic_opstr_list)



			self._static=_make_static(basis,static_opstr_list,dtype)
			self._dynamic=_make_dynamic(basis,dynamic_opstr_list,dtype)
			self._shape = self._static.shape
			self._basis=basis


		if static_other_list or dynamic_other_list:
			if not hasattr(self,"_shape"):
				found = False
				if shape is None: # if no shape argument found, search to see if the inputs have shapes.
					for O in static_other_list:
						try: # take the first shape found
							shape = O.shape
							found = True
							break
						except AttributeError: 
							continue

					if not found:
						for O,f,fargs in dynamic_other_list:
							try:
								shape = O.shape
								found = True
								break
							except AttributeError:
								continue
				else:
					found = True

				if not found:
					raise ValueError('missing argument shape')
				if shape[0] != shape[1]:
					raise ValueError('hamiltonian must be square matrix')

				self._shape=shape
				self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)
				self._dynamic = {}

			for O in static_other_list:
				if _sp.issparse(O):
					self._mat_checks(O)

					try:
						self._static += O.astype(self._dtype)
					except NotImplementedError:
						self._static = self._static + O.astype(self._dtype)

				elif O.__class__ is _np.ndarray:
					self._mat_checks(O)

					self._is_dense=True
					try:
						self._static += O.astype(self._dtype)
					except NotImplementedError:
						self._static = self._static + O.astype(self._dtype)

				elif O.__class__ is _np.matrix:
					self._mat_checks(O)

					self._is_dense=True
					try:
						self._static += O.astype(self._dtype)
					except NotImplementedError:
						self._static = self._static + O.astype(self._dtype)
				else:
					O = _np.asanyarray(O)
					self._mat_checks(O)

					self._is_dense=True			
					try:
						self._static += O.astype(self._dtype)
					except NotImplementedError:
						self._static = self._static + O.astype(self._dtype)

			try:
				self._static = self._static.tocsr(copy=False)
				self._static.sum_duplicates()
				self._static.eliminate_zeros()
			except: pass



			for	O,f,f_args in dynamic_other_list:
				_test_function(f,f_args)
				func = function(f,tuple(f_args))

				if _sp.issparse(O):
					self._mat_checks(O)

					O = O.astype(self._dtype)
					
				elif O.__class__ is _np.ndarray:
					self._mat_checks(O)
					self._is_dense=True

					O = _np.array(O,dtype=self._dtype,copy=copy)


				elif O.__class__ is _np.matrix:
					self._mat_checks(O)
					self._is_dense=True

					O = _np.array(O,dtype=self._dtype,copy=copy)

				else:
					O = _np.asanyarray(O)
					self._mat_checks(O)
					self._is_dense=True


				if func in self._dynamic:
					try:
						self._dynamic[func] += O
					except:
						self._dynamic[func] = self._dynamic[func] + O
				else:
					self._dynamic[func] = O


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
				self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)
				self._dynamic = {}

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
		"""tuple: shape of the `hamiltonian` object, always equal to `(Ns,Ns)`."""
		return self._shape

	@property
	def is_dense(self):
		"""bool: `True` if the operator contains a dense matrix as a componnent of either 
		the static or dynamic lists.

		""" 
		return self._is_dense

	@property
	def dtype(self):
		"""type: data type of `hamiltonian` object."""
		return _np.dtype(self._dtype).name

	@property
	def static(self):
		"""scipy.sparse.csr: static part of the operator."""
		return self._static

	@property
	def dynamic(self):
		"""tuple: contains dynamic parts of the operator as `(scipy.sparse.csr, fun, fun_args)`. See definition
		if the `dynamic_list` argument.
		
		"""
		return self._dynamic

	@property
	def T(self):
		""" Transposes the matrix defining the operator: :math:`H_{ij}\\mapsto H_{ji}`."""
		return self.transpose()

	@property
	def H(self):
		""" Transposes and conjugates the matrix defining the operator: :math:`H_{ij}\\mapsto H_{ji}^*`."""
		return self.getH()


	def check_is_dense(self):
		""" Toggles attribute `_.is_dense`."""
		is_sparse = _sp.issparse(self._static)
		for Hd in itervalues(self._dynamic):
			is_sparse *= _sp.issparse(Hd)

		self._is_dense = not is_sparse

	### state manipulation/observable routines

	def dot(self,V,time=0,check=True):
		"""Matrix-vector multiplication of `hamiltonian` operator at time `time`, with state `V`.

		.. math::
			H(t=\\texttt{time})|V\\rangle

		Note
		----
			It is faster to multiply the individual (static, dynamic) parts of the Hamiltonian first, then add all those 
			vectors together.

		Parameters
		----------
		V : numpy.ndarray
			Vector (quantums tate) to multiply the `hamiltonian` operator with.
		time : obj, optional
			Can be either one of the following:

			* float: time to evalute the time-dependent part of the operator at (if existent). 
				Default is `time = 0`.
			* list: there are two possible outcomes:

				-- if `V.shape[1] == len(time)`, the `hamiltonian` operator is evaluated at the i-th time 
					and dotted into the i-th column of `V` to get the i-th column of the output array.
				-- if `V.shape[1] == 1` or `V.shape[1] == 0`, the `_.dot` is evaluated on `V` for each time
					in `time`. The results are then stacked such that the columns contain all the vectors. 

				If either of these cases do not match, an error is thrown.
		check : bool, optional
			Whether or not to do checks for shape compatibility.
			

		Returns
		-------
		numpy.ndarray
			Vector corresponding to the `hamiltonian` operator applied on the state `V`.

		Example
		-------
		>>> B = H.dot(A,time=0,check=True)

		corresponds to :math:`B = HA`. 
	
		"""

		
		if self.Ns <= 0:
			return _np.asarray([])

		if ishamiltonian(V):
			raise ValueError("To multiply hamiltonians use '*' operator.")


		if _np.array(time).ndim > 0:
			if V.ndim > 3:
				raise ValueError("Expecting V.ndim < 4.")


			time = _np.asarray(time)
			if time.ndim > 1:
				raise ValueError("Expecting time to be one dimensional array-like.")

			if _sp.issparse(V):
				if V.shape[1] == time.shape[0]:
					V = V.tocsc()
					return _sp.vstack([self.dot(V.get_col(i),time=t,check=check) for i,t in enumerate(time)])
				else:
					raise ValueError("For non-scalar times V.shape[-1] must be equal to len(time).")
			else:
				V = _np.asarray(V)
				if V.ndim == 2 and V.shape[-1] == time.shape[0]:
					if V.shape[0] != self._shape[1]:
						raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

					V = V.T
					V_dot = _np.vstack([self.dot(v,time=t,check=check) for v,t in zip(V[:],time)]).T
					return V_dot

				elif V.ndim == 3 and V.shape[-1] == time.shape[0]:
					if V.shape[0] != self._shape[1]:
						raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

					if V.shape[0] != V.shape[1]:
						raise ValueError("Density matricies must be square!")

					V = V.transpose((2,0,1))
					V_dot = _np.dstack([self.dot(v,time=t,check=check) for v,t in zip(V[:],time)])

					return V_dot

				else:
					raise ValueError("For non-scalar times V.shape[-1] must be equal to len(time).")
		else:	
			if not check:
				V_dot = self._static.dot(V)	
				for func,Hd in iteritems(self._dynamic):
					V_dot += func(time)*(Hd.dot(V))

				return V_dot

			if V.__class__ is _np.ndarray:
				if V.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
		
				V_dot = self._static.dot(V)	
				for func,Hd in iteritems(self._dynamic):
					V_dot += func(time)*(Hd.dot(V))


			elif _sp.issparse(V):
				if V.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
		
				V_dot = self._static * V
				for func,Hd in iteritems(self._dynamic):
					V_dot += func(time)*(Hd.dot(V))
				return V_dot

			elif V.__class__ is _np.matrix:
				if V.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

				V_dot = self._static.dot(V)	
				for func,Hd in iteritems(self._dynamic):
					V_dot += func(time)*(Hd.dot(V))

			else:
				V = _np.asanyarray(V)
				if V.ndim not in [1,2]:
					raise ValueError("Expecting 1 or 2 dimensional array")

				if V.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

				V_dot = self._static.dot(V)	
				for func,Hd in iteritems(self._dynamic):
					V_dot += func(time)*(Hd.dot(V))

			return V_dot

	def expt_value(self,V,time=0,check=True,enforce_pure=False):
		""" Calculates expectation value of `hamiltonian` operator at time `time`, in state `V`.

		.. math::
			\\langle V|H(t=\\texttt{time})|V\\rangle

		Parameters
		----------
		V : numpy.ndarray
			Depending on the shape, can be a single state or a collection of pure or mixed states
			[see `enformce_pure`].
		time : obj, optional
			Can be either one of the following:

			* float: time to evalute the time-dependent part of the operator at (if existent). 
				Default is `time = 0`.
			* list: there are two possible outcomes:

				-- if `V.shape[1] == len(time)`, the `hamiltonian` operator is evaluated at the i-th time 
					and dotted into the i-th column of `V` to get the i-th column of the output array.
				-- if `V.shape[1] == 1` or `V.shape[1] == 0`, the `_.dot` is evaluated on `V` for each time
					in `time`. The results are then stacked such that the columns contain all the vectors. 

				If either of these cases do not match, an error is thrown.
		enforce_pure : bool, optional
			Flag to enforce pure expectation value of `V` is a square matrix with multiple pure states
			in the columns.
		check : bool, optional
			
		Returns
		-------
		float
			Expectation value of `hamiltonian` operator in state `V`.

		Example
		-------
		>>> H_expt = H.expt_value(V,time=0,diagonal=False,check=True)

		corresponds to :math:`H_\\{expt} = \\langle V|H(t=0)|V\\rangle`. 
			 
		"""
		if self.Ns <= 0:
			return _np.asarray([])

		if ishamiltonian(V):
			raise TypeError("Can't take expectation value of hamiltonian")

		if exp_op.isexp_op(V):
			raise TypeError("Can't take expectation value of exp_op")

		
		V_dot = self.dot(V,time=time,check=check)
		if _np.array(time).ndim > 0: # multiple time point expectation values
			if _sp.issparse(V): # multiple pure states multiple time points
				return (V.H.dot(V_dot)).diagonal()
			else:
				V = _np.asarray(V)
				if V.ndim == 2: # multiple pure states multiple time points
					return _np.einsum("ij,ij->j",V.conj(),V_dot)
				elif V.ndim == 3: # multiple mixed states multiple time points
					return _np.einsum("iij->j",V_dot)

		else:

			if _sp.issparse(V):
				if V.shape[0] != V.shape[1]: # pure states
					return _np.asscalar((V.H.dot(V_dot)).toarray())
				else: # density matrix
					return V.diagonal().sum()
			else:
				V_dot = _np.asarray(V_dot).squeeze()
				if V.ndim == 1: # pure state
					return _np.vdot(V,V_dot)
				elif (V.ndim == 2 and V.shape[0] != V.shape[1]) or enforce_pure: # multiple pure states
					return _np.einsum("ij,ij->j",V.conj(),V_dot)
				else: # density matrix
					return V_dot.trace()
			
	def matrix_ele(self,Vl,Vr,time=0,diagonal=False,check=True):
		"""Calculates matrix element of `hamiltonian` operator at time `time` in states `Vl` and `Vr`.

		.. math::
			\\langle V_l|H(t=\\texttt{time})|V_r\\rangle

		Note
		----
		Taking the conjugate or transpose of the state `Vl` is done automatically.  

		Parameters
		----------
		Vl : numpy.ndarray
			Vector(s)/state(s) to multiple with on left side.
		Vl : numpy.ndarray
			Vector(s)/state(s) to multiple with on right side.
		time : obj, optional
			Can be either one of the following:

			* float: time to evalute the time-dependent part of the operator at (if existent). 
				Default is `time = 0`.
			* list: there are two possible outcomes:

				-- if `V.shape[1] == len(time)`, the `hamiltonian` operator is evaluated at the i-th time 
					and dotted into the i-th column of `V` to get the i-th column of the output array.
				-- if `V.shape[1] == 1` or `V.shape[1] == 0`, the `_.dot` is evaluated on `V` for each time
					in `time`. The results are then stacked such that the columns contain all the vectors. 

				If either of these cases do not match, an error is thrown.
		diagonal : bool, optional
			When set to `True`, returs only diagonal part of expectation value. Default is `diagonal = False`.
		check : bool,

		Returns
		-------
		float
			Matrix element of `hamiltonian` operator between the states `Vl` and `Vr`.

		Example
		-------
		>>> H_lr = H.expt_value(Vl,Vr,time=0,diagonal=False,check=True)

		corresponds to :math:`H_\\{lr} = \\langle V_l|H(t=0)|V_r\\rangle`. 

		"""
		if self.Ns <= 0:
			return np.array([])

		Vr=self.dot(Vr,time=time,check=check)

		if not check:
			if diagonal:
				return _np.einsum("ij,ij->j",Vl.conj(),Vr)
			else:
				return Vl.T.conj().dot(Vr)
 
		if Vr.ndim > 2:
			raise ValueError('Expecting Vr to have ndim < 3')

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
				raise ValueError('Expecting Vl to have ndim < 3')

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

		elif _sp.issparse(Vl):
			if Vl.shape[0] != self._shape[1]:
				raise ValueError('dimension mismatch')
			if diagonal:
				return Vl.H.dot(Vr).diagonal()
			else:
				return Vl.H.dot(Vr)

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

		
	### transformation routines

	def project_to(self,proj):
		"""Projects/Transforms `hamiltonian` operator with projector/operator `proj`.

		Let us call the projector/transformation :math:`V`. Then, the function computes

		.. math::
			V^\\dagger H V

		Note
		----
		The `proj` argument can be a square array, in which case the function just transforms the
		`hailtonian` operator :math:`H`. Or it can be a projector which then projects :math:`H` onto
		a smaller Hilbert space.

		Projectors onto bases with symmetries other than `H.basis` can be conveniently obtain using the 
		`basis.get_proj()` method of the basis constructor class.

		Parameters
		----------
		proj : obj
			Can be either one of the following:
			
				* `hamiltonian` object
				* `exp_op` object
				* `numpy.ndarray`
				* `scipy.sparse` array

			The shape of `proj` need not be square, but has to comply with the matrix multiplication requirements
			in the definition above.

		Returns
		-------
		obj
			Projected/Transformed `hamiltonian` operator. The output object type depends on the object 
			type of `proj`.

		Example
		-------

		>>> H_new = H.project_to(V)

		correponds to :math:`V^\\dagger H V`.

		"""

		if ishamiltonian(proj):
			new = self._rmul_hamiltonian(proj.getH())
			return new._imul_hamiltonian(proj)

		elif exp_op.isexp_op(proj):
			return proj.sandwich(self)

		elif _sp.issparse(proj):
			if self._shape[1] != proj.shape[0]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(proj.shape,self._shape))
					
			new = self._rmul_sparse(proj.getH())
			new._shape = (proj.shape[1],proj.shape[1])
			return new._imul_sparse(proj)

		elif _np.isscalar(proj):
			raise NotImplementedError

		elif proj.__class__ == _np.ndarray:
			if self._shape[1] != proj.shape[0]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(proj.shape,self._shape))

			new = self._rmul_dense(proj.T.conj())
			new._shape = (proj.shape[1],proj.shape[1])
			return new._imul_dense(proj)


		elif proj.__class__ == _np.matrix:
			if self._shape[1] != proj.shape[0]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(proj.shape,self._shape))

			new = self._rmul_dense(proj.H)
			new._shape = (proj.shape[1],proj.shape[1])
			return new._imul_dense(proj)


		else:
			proj = _np.asanyarray(proj)
			if self._shape[1] != proj.shape[0]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(proj.shape,self._shape))

			new = self._rmul_dense(proj.T.conj())
			new._shape = (proj.shape[1],proj.shape[1])
			return new._imul_dense(proj)

	def rotate_by(self, other, generator=False, a=1.0, time=0.0,start=None, stop=None, num=None, endpoint=None, iterate=False):
		"""Rotates/Transforms `hamiltonian` object by an operator `other`.

		Let us denote the transformation by :math:`V`. With `generator=False`, `other` corresponds to the
		transformation :math:`V`, and this function implements

		.. math::
			V^\\dagger H V

		while for `generator=True`, `other` corresponds to a generator :math:`K`, and the function implements

		.. math::
			\\exp(a^*K^\\dagger) H \\exp(a K)

		Note
		----
		If `generator = False`, this function calls `project_to`.

		Parameters
		----------
		other : obj
			Can be either one of the following:
			
				* `hamiltonian` object
				* `exp_op` object
				* `numpy.ndarray`
				* `scipy.sparse` array
		generator : bool, optional
			If set to `True`, this flag renders `other` a generator, and implements the calculation of

			.. math::
				\\exp(a^*K^\\dagger) H \\exp(a K)

			If set to `False`, the function implements

			.. math::
			V^\\dagger H V

			Default is `generator = False`.

		All other optional arguments are the same as for the `exp_op` class.

		Returns
		-------
		obj
			Transformed `hamiltonian` operator. The output object type depends on the object type of `other`.
	
		Examples
		--------
		>>> H_new = H.rotate_by(V,generator=False)

		corresponds to :math:`V^\\dagger H V`.

		>>> H_new = H.rotate_by(K,generator=True,**exp_op_args)

		corresponds to :math:`\\exp(K^\\dagger) H \\exp(K)`.

		"""

		if generator:
			return exp_op(other,a=a,time=time,start=start,stop=stop,num=num,endpoint=endpoint,iterate=iterate).sandwich(self)
		else:
			return self.project_to(other)


	### Diagonalisation routines

	def eigsh(self,time=0.0,**eigsh_args):
		"""Computes SOME eigenvalues of hermitian `hamiltonian` operator using SPARSE hermitian methods.

		This function method solves for eigenvalues and eigenvectors, but can only solve for a few of them accurately.
		It calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html/>`_, which is a wrapper for ARPACK.

		Note
		----
		Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		----------
		time : float
			Time to evalute the `hamiltonian` operator at (if time dependent). Default is `time = 0.0`.
		eigsh_args : 
			For all additional arguments see documentation of `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/generated/scipy.sparse.linalg.eigsh.html/>`_.
			
		Returns
		-------
		tuple
			Tuple containing the `(eigenvalues, eigenvectors)` of the `hamiltonian` operator.

		Example
		-------
		>>> eigenvalues,eigenvectors = H.eigsh(time=time,**eigsh_args)

		"""
		if _np.array(time).ndim > 0:
			raise TypeError('expecting scalar argument for time')

		if self.Ns <= 0:
			return _np.asarray([]), _np.asarray([[]])

		return _sla.eigsh(self.tocsr(time=time),**eigsh_args)

	def eigh(self,time=0,**eigh_args):
		"""Computes COMPLETE eigensystem of hermitian `hamiltonian` operator using DENSE hermitian methods.

		This function method solves for all eigenvalues and eigenvectors. It calls 
		`numpy.linalg.eigh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigh.html/>`_, 
		and uses wrapped LAPACK functions which are contained in the module py_lapack.

		Note
		----
		Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		----------
		time : float
			Time to evalute the `hamiltonian` operator at (if time dependent). Default is `time = 0.0`.
		
		eigh_args : 
			For all additional arguments see documentation of `numpy.linalg.eigh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigh.html/>`_.
			
		Returns
		-------
		tuple
			Tuple containing the `(eigenvalues, eigenvectors)` of the `hamiltonian` operator.

		Example
		-------
		>>> eigenvalues,eigenvectors = H.eigh(time=time,**eigh_args)

		"""
		eigh_args["overwrite_a"] = True
		
		if _np.array(time).ndim > 0:
			raise TypeError('expecting scalar argument for time')


		if self.Ns <= 0:
			return _np.asarray([]),_np.asarray([[]])

		# fill dense array with hamiltonian
		H_dense = self.todense(time=time)		
		# calculate eigh
		E,H_dense = _la.eigh(H_dense,**eigh_args)
		return E,H_dense

	def eigvalsh(self,time=0,**eigvalsh_args):
		"""Computes ALL eigenvalues of hermitian `hamiltonian` operator using DENSE hermitian methods.

		This function method solves for all eigenvalues. It calls 
		`numpy.linalg.eigvalsh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh/>`_, 
		and uses wrapped LAPACK functions which are contained in the module py_lapack.

		Note
		----
		Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
		to reassure themselves of the hermiticity properties before use. 

		Parameters
		----------
		time : float
			Time to evalute the `hamiltonian` operator at (if time dependent). Default is `time = 0.0`.
		
		eigvalsh_args : 
			For all additional arguments see documentation of `numpy.linalg.eigvalsh <https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linalg.eigvalsh.html#numpy.linalg.eigvalsh/>`_.
			
		Returns
		-------
		numpy.ndarray
			Eigenvalues of the `hamiltonian` operator.

		Example
		-------
		>>> eigenvalues = H.eigvalsh(time=time,**eigvalsh_args)

		"""

		
		if _np.array(time).ndim > 0:
			raise TypeError('expecting scalar argument for time')

		if self.Ns <= 0:
			return _np.asarray([])

		H_dense = self.todense(time=time)
		eigvalsh_args["overwrite_a"] = True
		E = _la.eigvalsh(H_dense,**eigvalsh_args)
		return E


	### Schroedinger evolution routines

	def __LO(self,time,rho):
		rho = rho.reshape((self.Ns,self.Ns))

		rho_comm = self._static.dot(rho)
		rho_comm -= (self._static.T.dot(rho.T)).T
		for func,Hd in iteritems(self._dynamic):
			ft = func(time)
			rho_comm += ft*Hd.dot(rho)	
			rho_comm -= ft*(Hd.T.dot(rho.T)).T

		rho_comm *= -1j
		return rho_comm.reshape((-1,))

	def __multi_SO_real(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the real time Schrodinger operator -i*H(t)*|V >
			This function is designed for real hamiltonians and increases the speed of integration compared to __SO
		
		u_dot + iv_dot = -iH(u + iv)
		u_dot = Hv
		v_dot = -Hu
		"""
		V = V.reshape((2*self._Ns,-1))
		V_dot = _np.zeros_like(V)
		V_dot[:self._Ns,:] = self._static.dot(V[self._Ns:,:])
		V_dot[self._Ns:,:] = -self._static.dot(V[:self._Ns,:])
		for func,Hd in iteritems(self._dynamic):
			V_dot[:self._Ns,:] += func(time)*Hd.dot(V[self._Ns:,:])
			V_dot[self._Ns:,:] += -func(time)*Hd.dot(V[:self._Ns,:])

		return V_dot.reshape((-1,))

	def __multi_SO(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the real time Schrodinger operator -i*H(t)*|V >
		"""
		V = V.reshape((self.Ns,-1))
		V_dot = self._static.dot(V)	
		for func,Hd in iteritems(self._dynamic):
			V_dot += func(time)*(Hd.dot(V))

		return -1j*V_dot.reshape((-1,))

	def __multi_ISO(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the Imaginary time Schrodinger operator -H(t)*|V >
		"""
		V = V.reshape((self._Ns,-1))
		V_dot = -self._static.dot(V)	
		for func,Hd in iteritems(self._dynamic):
			V_dot -= func(time)*(Hd.dot(V))

		return V_dot.reshape((-1,))

	def __SO_real(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the real time Schrodinger operator -i*H(t)*|V >
			This function is designed for real hamiltonians and increases the speed of integration compared to __SO
		
		u_dot + iv_dot = -iH(u + iv)
		u_dot = Hv
		v_dot = -Hu
		"""
		V_dot = _np.zeros_like(V)
		V_dot[:self._Ns] = self._static.dot(V[self._Ns:])
		V_dot[self._Ns:] = -self._static.dot(V[:self._Ns])
		for func,Hd in iteritems(self._dynamic):
			V_dot[:self._Ns] += func(time)*Hd.dot(V[self._Ns:])
			V_dot[self._Ns:] += -func(time)*Hd.dot(V[:self._Ns])

		return V_dot

	def __SO(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the real time Schrodinger operator -i*H(t)*|V >
		"""
		V_dot = self._static.dot(V)	
		for func,Hd in iteritems(self._dynamic):
			V_dot += func(time)*(Hd.dot(V))

		return -1j*V_dot

	def __ISO(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the Imaginary time Schrodinger operator -H(t)*|V >
		"""

		V_dot = -self._static.dot(V)	
		for func,Hd in iteritems(self._dynamic):
			V_dot -= func(time)*(Hd.dot(V))

		return V_dot


	def evolve(self,v0,t0,times,eom="SE",solver_name="dop853",H_real=False,verbose=False,iterate=False,imag_time=False,**solver_args):
		"""Implements (imaginary) time evolution generated by the `hamiltonian` object.

		The functions handles evolution generated by both time-dependent and time-independent Hamiltonians. 

		Currently the following three built-in routines are supported:
			
			i) real-time Schroedinger equation: :math:`\\partial_t|v(t)\\rangle=-iH(t)|v(t)\\rangle`
			ii) imaginary-time Schroedinger equation: :math:`\\partial_t|v(t)\\rangle=-H(t)|v(t)\\rangle`
			iii) Liouvillian dynamics: 

		Note
		----
		For a user-defined custom ODE solver which can handle non-linear equations, check out the
		`measurements.evolve()` routine, which has a similar functionality but allows for a complete freedom
		over the differential equation to be solved.
		
		Parameters
		----------
		v0 : numpy.ndarray
			Initial state.
		t0 : float
			Initial time.
		times : numpy.ndarray
			Vector of times to compute the time-evolved state at.
		eom : str, optional
			Specifies the ODE type. Can be either one of

				* "SE", real and imaginary-time Schroedinger equation.
				* "LvNE", real-time Liouville equation.

			Default is "eom = SE" (Schroedinger evolution).
		iterate : bool, optional
			If set to `True`, creates a generator object for the time-evolved the state. Default is `False`.
		solver_name : str, optional
			Scipy solver integrator name. Default is `dop853`. 

			See `scipy integrator (solver) <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.integrate.ode.html/>`_ for other options.
		solver_args : dict, optional
			Dictionary with additional `scipy integrator (solver) <https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.integrate.ode.html/>`_.	
		H_real : bool, optional 
			Flag to determine if `f` is real or complex-valued. Default is `False`.
		imag_time : bool, optional
			Must be set to `True` when `f` defines imaginary-time evolution, in order to normalise the state 
			at each time in `times`. Default is `False`.
		verbose : bool, optional
			If set to `True`, prints normalisation of state at teach time in `times`.

		Returns
		-------
		obj
			Can be either one of the following:
			* numpy.ndarray containing evolved state against time.
			* generator object for time-evolved state (requires `iterate = True`).

			Note that for Liouvillian dynamics the output is a square complex `numpy.ndarray`.

		Example
		-------
		>>> v_t = H.evolve(v0,t0,times,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False,**solver_args)

		"""

		from scipy.integrate import complex_ode
		from scipy.integrate import ode

		shape0 = v0.shape

		if eom == "SE":
			n = _np.linalg.norm(v0,axis=0) # needed for imaginary time to preserve the proper norm of the state. 

			
			if v0.ndim > 2:
				raise ValueError("v0 must have ndim <= 2")

			if v0.shape[0] != self.Ns:
				raise ValueError("v0 must have {0} elements".format(self.Ns))

			if imag_time:
				v0 = v0.astype(self.dtype)
				if _np.iscomplexobj(v0):
					if v0.ndim == 1:
						solver = complex_ode(self.__ISO)
					else:
						solver = complex_ode(self.__multi_ISO)
				else:
					if v0.ndim == 1:
						solver = ode(self.__ISO)
					else:
						solver = ode(self.__multi_ISO)
			else:
				if H_real:
					v1 = v0
					v0 = _np.zeros((2*self._Ns,)+v0.shape[1:],dtype=v1.real.dtype)
					v0[:self._Ns] = v1.real
					v0[self._Ns:] = v1.imag
					if v0.ndim == 1:
						solver = ode(self.__SO_real)
					else:
						solver = ode(self.__multi_SO_real)
				else:
					v0 = v0.astype(_np.complex128)
					if v0.ndim == 1:
						solver = complex_ode(self.__SO)
					else:
						solver = complex_ode(self.__multi_SO)

		elif eom == "LvNE":
			n = 1.0
			if v0.ndim != 2:
				raise ValueError("v0 must have ndim = 2")

			if v0.shape != self._shape:
				raise ValueError("v0 must be same shape as Hamiltonian")

			if imag_time:
				raise NotImplementedError("imaginary time not implemented for Liouville-von Neumann dynamics")
			else:
				if H_real:
					raise NotImplementedError("H_real not implemented for Liouville-von Neumann dynamics")
				else:
					solver = complex_ode(self.__LO)
		else:
			raise ValueError("'{} equation' not recognized, must be 'SE' or 'LvNE'".format(equation))


		if _np.iscomplexobj(times):
			raise ValueError("times must be real number(s).")

		if solver_name in ["dop853","dopri5"]:
			if solver_args.get("nsteps") is None:
				solver_args["nsteps"] = _np.iinfo(_np.int32).max
			if solver_args.get("rtol") is None:
				solver_args["rtol"] = 1E-9
			if solver_args.get("atol") is None:
				solver_args["atol"] = 1E-9

				
		solver.set_integrator(solver_name,**solver_args)
		solver.set_initial_value(v0.ravel(), t0)

		if _np.isscalar(times):
			return self._evolve_scalar(solver,v0,t0,times,imag_time,H_real,n,shape0)
		else:
			if iterate:
				return self._evolve_iter(solver,v0,t0,times,verbose,imag_time,H_real,n,shape0)
			else:
				return self._evolve_list(solver,v0,t0,times,verbose,imag_time,H_real,n,shape0)

			
	def _evolve_scalar(self,solver,v0,t0,time,imag_time,H_real,n,shape0):
		from numpy.linalg import norm
		N_ele = v0.size//2

		if time == t0:
			if H_real:
				_np.squeeze((v0[:N_ele] + 1j*v0[N_ele:]).reshape(shape0))
			else:
				return _np.squeeze(v0.reshape(shape0))

		solver.integrate(time)
		if solver.successful():
			if imag_time: solver._y /= (norm(solver._y)/n)
			if H_real:
				return _np.squeeze((solver.y[:N_ele] + 1j*solver.y[N_ele:]).reshape(shape0))
			else:
				return _np.squeeze(solver.y.reshape(shape0))
		else:
			raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(time))	

	def _evolve_list(self,solver,v0,t0,times,verbose,imag_time,H_real,n,shape0):
		from numpy.linalg import norm

		N_ele = v0.size//2
		v = _np.empty(shape0+(len(times),),dtype=_np.complex128)
		
		for i,t in enumerate(times):
			if t == t0:
				if verbose: print("evolved to time {0}, norm of state {1}".format(t,_np.linalg.norm(solver.y)))
				if H_real:
					v[...,i] = _np.squeeze((v0[:N_ele] + 1j*v0[N_ele:]).reshape(shape0))
				else:
					v[...,i] = _np.squeeze(v0.reshape(shape0))
				continue

			solver.integrate(t)
			if solver.successful():
				if verbose: print("evolved to time {0}, norm of state {1}".format(t,_np.linalg.norm(solver.y)))
				if imag_time: solver._y /= (norm(solver._y)/n)
				if H_real:
					v[...,i] = _np.squeeze((solver.y[:N_ele] + 1j*solver.y[N_ele:]).reshape(shape0))
				else:
					v[...,i] = _np.squeeze(solver.y.reshape(shape0))
			else:
				raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(t))
				
		return _np.squeeze(v)

	def _evolve_iter(self,solver,v0,t0,times,verbose,imag_time,H_real,n,shape0):
		from numpy.linalg import norm
		N_ele = v0.size//2

		for i,t in enumerate(times):
			if t == t0:
				if verbose: print("evolved to time {0}, norm of state {1}".format(t,_np.linalg.norm(solver.y)))
				if H_real:
					yield _np.squeeze((v0[:N_ele] + 1j*v0[N_ele:]).reshape(shape0))
				else:
					yield _np.squeeze(v0.reshape(shape0))
				continue
				

			solver.integrate(t)
			if solver.successful():
				if verbose: print("evolved to time {0}, norm of state {1}".format(t,_np.linalg.norm(solver.y)))
				if imag_time: solver._y /= (norm(solver.y)/n)
				if H_real:
					yield _np.squeeze((solver.y[:N_ele] + 1j*solver.y[N_ele:]).reshape(shape0))
				else:
					yield _np.squeeze(solver.y.reshape(shape0))
			else:
				raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(t))

	
	### routines to change object type	

	def aslinearoperator(self,time=0.0):
		"""Returns copy of a `hamiltonian` object at time `time` as a `scipy.sparse.linalg.LinearOperator`.

		Casts the `hamiltonian` object as a
		`scipy.sparse.linalg.LinearOperator <https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.sparse.linalg.LinearOperator.html/>_
		object.

		Parameters
		----------
		time : float, optional
			Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
		
		Returns
		-------
		:obj:`scipy.sparse.linalg.LinearOperator`

		Example
		-------
		>>> H_aslinop=H.aslinearoperator(time=time)

		"""
		time = _np.array(time)
		if time.ndim > 0:
			raise ValueError("time must be scalar!")
		matvec = functools.partial(_hamiltonian_dot,self,time)
		rmatvec = functools.partial(_hamiltonian_dot,self.H,time)
		return _sla.LinearOperator(self.get_shape,matvec,rmatvec=rmatvec,matmat=matvec,dtype=self._dtype)		

	def tocsr(self,time=0):
		"""Returns copy of a `hamiltonian` object at time `time` as a `scipy.sparse.csr_matrix`.

		Casts the `hamiltonian` object as a
		`scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html/>_
		object.

		Parameters
		----------
		time : float, optional
			Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
		
		Returns
		-------
		:obj:`scipy.sparse.csr_matrix`

		Example
		-------
		>>> H_csr=H.tocsr(time=time)

		"""

		if self.Ns <= 0:
			return _sp.csr_matrix(_np.asarray([[]]))
		if _np.array(time).ndim > 0:
			raise TypeError('expecting scalar argument for time')


		H = _sp.csr_matrix(self._static)

		for func,Hd in iteritems(self._dynamic):
			Hd = _sp.csr_matrix(Hd)
			try:
				H += Hd * func(time)
			except:
				H = H + Hd * func(time)


		return H

	def tocsc(self,time=0):
		"""Returns copy of a `hamiltonian` object at time `time` as a `scipy.sparse.csc_matrix`.

		Casts the `hamiltonian` object as a
		`scipy.sparse.csc_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html/>_
		object.

		Parameters
		----------
		time : float, optional
			Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
		
		Returns
		-------
		:obj:`scipy.sparse.csc_matrix`

		Example
		-------
		>>> H_csc=H.tocsc(time=time)

		"""
		if self.Ns <= 0:
			return _sp.csc_matrix(_np.asarray([[]]))
		if _np.array(time).ndim > 0:
			raise TypeError('expecting scalar argument for time')

		H = _sp.csc_matrix(self._static)
		for func,Hd in iteritems(self._dynamic):
			Hd = _sp.csc_matrix(Hd)
			try:
				H += Hd * func(time)
			except:
				H = H + Hd * func(time)


		return H
	
	def todense(self,time=0,order=None, out=None):
		"""Returns copy of a `hamiltonian` object at time `time` as a dense array.

		This function can overflow memory if not used carefully!

		Note
		----
		If the array dimension is too large, scipy may choose to cast the `hamiltonian` operator as a
		`numpy.matrix` instead of a `numpy.ndarray`. In such a case, one can use the `hamiltonian.toarray()`
		method.

		Parameters
		----------
		time : float, optional
			Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
		order : str, optional
			Whether to store multi-dimensional data in C (rom-major) or Fortran (molumn-major) order in memory.
			
			Default is `order = None`, indicating the NumPy default of C-ordered.
			
			Cannot be specified in conjunction with the `out` argument.
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
		>>> H_dense=H.todense(time=time)

		"""

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)
			out = _np.asmatrix(out)

		if _sp.issparse(self._static):
			self._static.todense(order=order,out=out)
		else:
			out[:] = self._static[:]

		for func,Hd in iteritems(self._dynamic):
			out += Hd * func(time)
		
		return out

	def toarray(self,time=0,order=None, out=None):
		"""Returns copy of a `hamiltonian` object at time `time` as a dense array.

		This function can overflow memory if not used carefully!


		Parameters
		----------
		time : float, optional
			Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
		order : str, optional
			Whether to store multi-dimensional data in C (rom-major) or Fortran (molumn-major) order in memory.
			
			Default is `order = None`, indicating the NumPy default of C-ordered.
			
			Cannot be specified in conjunction with the `out` argument.
		out : numpy.ndarray
			Array to fill in with the output.
		
		Returns
		-------
		numpy.ndarray
			Dense `hamiltonian` array.

		Example
		-------
		>>> H_dense=H.toarray(time=time)

		"""

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)

		if _sp.issparse(self._static):
			self._static.toarray(order=order,out=out)
		else:
			out[:] = self._static[:]

		for func,Hd in iteritems(self._dynamic):
			out += Hd * func(time)
		
		return out

	def as_dense_format(self,copy=False):
		"""Casts `hamiltonian` operator to DENSE format.

		Parameters
		----------
		copy : bool,optional
			Whether to return a deep copy of the original object. Default is `copy = False`.

		Returns
		-------
		obj
			Either one of the following:

			* Shallow copy, if `copy = False`.
			* Deep copy, if `copy = True`.

		Example
		-------
		>>> H_dense=H.as_dense_format()

		"""
		if copy:
			new = _deepcopy(self)
		else:
			new = _shallowcopy(self)


		if _sp.issparse(new._static):
			new._static = new._static.todense()
		else:
			new._static = _np.asarray(new._static)


		for func in new._dynamic:
			if _sp.issparse(new._dynamic[func]):
				new._dynamic[func] = new._dynamic[func].toarray()
			else:
				new._dynamic[func] = _np.asarray(new._dynamic[func],copy=copy)

		return new

	def as_sparse_format(self,fmt,copy=False):
		"""Casts `hamiltonian` operator to SPARSE format.

		Parameters
		----------
		fmt : str {"csr","csc","dia","bsr"}
			Specifies for mat of sparse array.
		copy : bool,optional
			Whether to return a deep copy of the original object. Default is `copy = False`.

		Returns
		-------
		obj
			Either one of the following:

			* Shallow copy, if `copy = False`.
			* Deep copy, if `copy = True`.

		Example
		-------
		>>> H_sparse=H.as_sparse_format()

		"""
		if type(fmt) is not str:
			raise ValueError("Expecting string for 'fmt'")

		if fmt not in ["csr","csc","dia","bsr"]:
			raise ValueError("'{0}' is not a valid sparse format or does not support arithmetic.".format(fmt))

		if copy:
			new = _deepcopy(self)
		else:
			new = _shallowcopy(self)

		sparse_constuctor = getattr(_sp,fmt+"_matrix")

		new._static = sparse_constuctor(new._static)
		new._dynamic = list(new._dynamic)
		new._dynamic = {func:sparse_constructor(Hd) for func,Hd in iteritems(new._dynamic)}

		return new


	### algebra operations

	def transpose(self,copy=False):
		"""Transposes `hamiltonian` operator.

		Note
		----
		This function does NOT conjugate the operator.

		Returns
		-------
		:obj:`hamiltonian`
			:math:`H_{ij}\\mapsto H_{ji}`

		Example
		-------

		>>> H_tran = H.transpose()

		"""
		if copy:
			new = _deepcopy(self)
		else:
			new = _shallowcopy(self)

		new._static = new._static.T
		new._dynamic = {func:Hd.T for func,Hd in iteritems(new._dynamic)}

		return new

	def conj(self):
		"""Conjugates `hamiltonian` operator.

		Note
		----
		This function does NOT transpose the operator.

		Returns
		-------
		:obj:`hamiltonian`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Example
		-------

		>>> H_conj = H.conj()

		"""
		new = _shallowcopy(self)

		new._static = new._static.conj()
		new._dynamic = {func.conj():Hd.conj() for func,Hd in iteritems(new._dynamic)}
		
		'''
		new._dynamic = list(new._dynamic)
		n = len(self._dynamic)
		for i in range(n):
		 	new._dynamic[i] = list(new._dynamic[i])
		 	new._dynamic[i][0] = new._dynamic[i][0].conj()
		 	new._dynamic[i] = tuple(new._dynamic[i])

		new._dynamic = tuple(new._dynamic)
		'''
		return new

	def getH(self,copy=False):
		"""Calculates hermitian conjugate of `hamiltonian` operator.

		Parameters
		----------
		copy : bool, optional
			Whether to return a deep copy of the original object. Default is `copy = False`.

		Returns
		-------
		:obj:`hamiltonian`
			:math:`H_{ij}\\mapsto H_{ij}^*`

		Example
		-------

		>>> H_herm = H.getH()

		"""
		return self.conj().transpose(copy=copy)	

	### lin-alg operations

	def diagonal(self,time=0):
		"""Calculates diagonal of `hamiltonian` operator at time `time`.

		Parameters
		----------
		time : float, optional
			Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
		
		Returns
		-------
		numpy.ndarray
			Diagonal part of operator :math:`H(t=\\texttt{time})`.

		Example
		-------

		>>> H_diag = H.diagonal(time=0.0)

		"""
		if self.Ns <= 0:
			return 0
		if _np.array(time).ndim > 0:
			raise TypeError('expecting scalar argument for time')

		diagonal = self._static.diagonal()
		for func,Hd in iteritems(self._dynamic):
			diagonal += Hd.diagonal() * func(time)

		return diagonal

	def trace(self,time=0):
		""" Calculates trace of `hamiltonian` operator at time `time`.

		Parameters
		----------
		time : float, optional
			Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
		
		Returns
		-------
		float
			Trace of operator :math:`\\sum_{j=1}^{Ns} H_{jj}(t=\\texttt{time})`.

		Example
		-------

		>>> H_tr = H.tr(time=0.0)

		"""
		if self.Ns <= 0:
			return 0
		if _np.array(time).ndim > 0:
			raise TypeError('expecting scalar argument for time')

		trace = self._static.diagonal().sum()
		for func,Hd in iteritems(self._dynamic):
			trace += Hd.diagonal().sum() * func(time)

		return trace
 		

	def astype(self,dtype):
		""" Changes data type of `hamiltonian` object.

		Parameters
		----------
		dtype : 'type'
			The data type (e.g. numpy.float64) to cast the Hamiltonian with.

		Returns
		:obj:`hamiltonian`
			`hamiltonian` operator with altered data type.

		Example
		-------
		>>> H_cpx=H.astype(np.complex128)

		"""

		if dtype not in supported_dtypes:
			raise TypeError('hamiltonian does not support type: '+str(dtype))

		new = _shallowcopy(self)

		new._dtype = dtype
		new._static = new._static.astype(dtype)
		new._dynamic = {func:Hd.astype(dtype) for func,Hd in iteritems(new._dynamic)}

		return new

	def copy(self):
		"""Returns a deep copy of `hamiltonian` object."""
		return _deepcopy(self)


	###################
	# special methods #
	###################


	def __getitem__(self,key):
		if len(key) != 3:
			raise IndexError("invalid number of indices, hamiltonian must be indexed with three indices [time,row,col].")
		try:
			times = iter(key[0])
			iterate=True
		except TypeError:
			time = key[0]
			iterate=False

		key = tuple(key[1:])
		if iterate:
			ME = []
			if self.is_dense:
				for t in times:
					ME.append(self.todense(time=t)[key])
			else:
				for t in times:
					ME.append(self.tocsr(time=t)[key])
				
			ME = tuple(ME)
		else:
			ME = self.tocsr(time=time)[key]

		return ME
			
		


	def __str__(self):
		string = "static mat: \n{0}\n\n\ndynamic:\n".format(self._static.__str__())
		for i,(func,Hd) in enumerate(iteritems(self._dynamic)):
			h_str = Hd.__str__()
			func_str = func.__str__()
			
			string += ("{0}) func: {2}, mat: \n{1} \n".format(i,h_str,func_str))

		return string
		

	def __repr__(self):
		matrix_format={"csr":"Compressed Sparse Row",
						"csc":"Compressed Sparse Column",
						"dia":"DIAgonal",
						"bsr":"Block Sparse Row"
						}
		if self.is_dense:
			return "<{0}x{1} qspin dense hamiltonian of type '{2}'>".format(*(self._shape[0],self._shape[1],self._dtype))
		else:
			fmt = matrix_format[self._static.getformat()]
			return "<{0}x{1} qspin sprase hamiltonian of type '{2}' stored in {3} format>".format(*(self._shape[0],self._shape[1],self._dtype,fmt))


	def __neg__(self): # -self
		new = _shallowcopy(self)

		new._static = -new._static

		new._dynamic = {func:-Hd for func,Hd in iteritems(new._dynamic)}
		# new._dynamic = list(new._dynamic)
		# n = len(new._dynamic)
		# for i in range(n):
		# 	new._dynamic[i][-1] = -new._dynamic[i][-1]

		# new._dynamic = tuple(new._dynamic)
		
		return new


	def __call__(self,time):
		if self.is_dense:
			return self.toarray(time)
		else:
			return self.tocsr(time)


	##################################
	# symbolic arithmetic operations #
	# currently only have +,-,* like #
	# operators implimented.		 #
	##################################

	def __pow__(self,power):
		if type(power) is not int:
			raise TypeError("hamiltonian can only be raised to integer power.")

		return reduce(mul,(self for i in range(power)))



	def __mul__(self,other): # self * other
		if ishamiltonian(other):			
			return self._mul_hamiltonian(other)

		elif _sp.issparse(other):
			self._mat_checks(other,casting="unsafe")
			return self._mul_sparse(other)

		elif _np.isscalar(other):
			return self._mul_scalar(other)

		elif other.__class__ == _np.ndarray:
			self._mat_checks(other,casting="unsafe")
			return self._mul_dense(other)

		elif other.__class__ == _np.matrix:
			self._mat_checks(other,casting="unsafe")
			return self._mul_dense(other)

		else:
			other = _np.asanyarray(other)
			self._mat_checks(other,casting="unsafe")
			return self._mul_dense(other)






	def __rmul__(self,other): # other * self
		if ishamiltonian(other):
			self._mat_checks(other,casting="unsafe")
			return self._rmul_hamiltonian(other)

		elif _sp.issparse(other):
			self._mat_checks(other,casting="unsafe")
			return self._rmul_sparse(other)

		elif _np.isscalar(other):

			return self._mul_scalar(other)

		elif other.__class__ == _np.ndarray:
			self._mat_checks(other,casting="unsafe")
			return self._rmul_dense(other)

		elif other.__class__ == _np.matrix:
			self._mat_checks(other,casting="unsafe")
			return self._rmul_dense(other)

		else:
			other = _np.asanyarray(other)
			self._mat_checks(other,casting="unsafe")
			return self._rmul_dense(other)







	def __imul__(self,other): # self *= other
		if ishamiltonian(other):
			self._mat_checks(other)
			return self._imul_hamiltonian(other)

		
		elif _sp.issparse(other):
			self._mat_checks(other)	
			return self._imul_sparse(other)

		elif _np.isscalar(other):
			return self._imul_scalar(other)

		elif other.__class__ == _np.ndarray:
			self._mat_checks(other)	
			return self._imul_dense(other)

		elif other.__class__ == _np.matrix:
			self._mat_checks(other)	
			return self._imul_dense(other)

		else:
			other = _np.asanyarray(other)
			self._mat_checks(other)	
			return self._imul_dense(other)


	def __truediv__(self,other):
		return self.__div__(other)

	def __div__(self,other): # self / other
		if ishamiltonian(other):			
			return NotImplemented

		elif _sp.issparse(other):
			return NotImplemented

		elif _np.isscalar(other):
			return self._mul_scalar(1.0/other)

		elif other.__class__ == _np.ndarray:
			return NotImplemented

		elif other.__class__ == _np.matrix:
			return NotImplemented

		else:
			return NotImplemented





	def __rdiv__(self,other): # other / self
		return NotImplemented


	def __idiv__(self,other): # self *= other
		if ishamiltonian(other):
			return NotImplemented
		
		elif _sp.issparse(other):
			return NotImplemented

		elif _np.isscalar(other):
			return self._imul_scalar(1.0/other)

		elif other.__class__ == _np.ndarray:
			return NotImplemented

		elif other.__class__ == _np.matrix:
			return NotImplemented

		else:
			return NotImplemented




	def __add__(self,other): # self + other
		if ishamiltonian(other):
			self._mat_checks(other,casting="unsafe")
			return self._add_hamiltonian(other)

		elif _sp.issparse(other):
			self._mat_checks(other,casting="unsafe")
			return self._add_sparse(other)
			
		elif _np.isscalar(other):
			if other==0.0:
				return self.copy()
			else:
				raise NotImplementedError('hamiltonian does not support addition by nonzero scalar')

		elif other.__class__ == _np.ndarray:
			self._mat_checks(other,casting="unsafe")
			return self._add_dense(other)

		elif other.__class__ == _np.matrix:
			self._mat_checks(other,casting="unsafe")
			return self._add_dense(other)

		else:
			other = _np.asanyarray(other)
			self._mat_checks(other,casting="unsafe")
			return self._add_dense(other)





	def __radd__(self,other): # other + self
		return self.__add__(other)






	def __iadd__(self,other): # self += other
		if ishamiltonian(other):
			self._mat_checks(other)
			return self._iadd_hamiltonian(other)

		elif _sp.issparse(other):
			self._mat_checks(other)	
			return self._iadd_sparse(other)

		elif _np.isscalar(other):
			if other==0.0:
				return self.copy()
			else:
				raise NotImplementedError('hamiltonian does not support addition by nonzero scalar')

		elif other.__class__ == _np.ndarray:
			self._mat_checks(other)	
			return self._iadd_dense(other)

		else:
			other = _np.asanyarray(other)
			self._mat_checks(other)				
			return self._iadd_dense(other)






	def __sub__(self,other): # self - other
		if ishamiltonian(other):
			self._mat_checks(other,casting="unsafe")
			return self._sub_hamiltonian(other)

		elif _sp.issparse(other):
			self._mat_checks(other,casting="unsafe")
			return self._sub_sparse(other)

		elif _np.isscalar(other):
			if other==0.0:
				return self.copy()
			else:
				raise NotImplementedError('hamiltonian does not support subtraction by nonzero scalar')

		elif other.__class__ == _np.ndarray:
			self._mat_checks(other,casting="unsafe")
			return self._sub_dense(other)

		else:
			other = _np.asanyarray(other)
			self._mat_checks(other,casting="unsafe")
			return self._sub_dense(other)



	def __rsub__(self,other): # other - self
		# NOTE: because we use signed types this is possble
		return self.__sub__(other).__neg__()




	def __isub__(self,other): # self -= other
		if ishamiltonian(other):
			self._mat_checks(other)
			return self._isub_hamiltonian(other)

		elif _sp.issparse(other):
			self._mat_checks(other)			
			return self._isub_sparse(other)

		elif _np.isscalar(other):
			if other==0.0:
				return self.copy()
			else:
				raise NotImplementedError('hamiltonian does not support subtraction by nonzero scalar')

		elif other.__class__ == _np.ndarray:
			self._mat_checks(other)	
			return self._isub_dense(other)

		else:
			other = _np.asanyarray(other)
			self._mat_checks(other)	
			return self._sub_dense(other)

	##########################################################################################	
	##########################################################################################
	# below all of the arithmetic functions are implimented for various combination of types #
	##########################################################################################
	##########################################################################################


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



	def _add_hamiltonian(self,other): 
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype)

		new._is_dense = new._is_dense or other._is_dense

		try:
			new._static += other._static 
		except NotImplementedError:
			new._static = new._static + other._static 

		try:
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		for func,Hd in iteritems(other._dynamic):
			if func in new._dynamic:
				try:
					new._dynamic[func] += Hd
				except NotImplementedError:
					new._dynamic[func] = new._dynamic[func] + Hd

				try:
					new._dynamic[func].sum_duplicates()
					new._dynamic[func].eliminate_zeros()
				except: pass

				if _check_almost_zero(new._dynamic[func]):
					new._dynamic.pop(func)
			else:
				new._dynamic[func] = Hd

		new.check_is_dense()
		return new




	def _iadd_hamiltonian(self,other):
		self._is_dense = self._is_dense or other._is_dense

		try:
			self._static += other._static 
		except NotImplementedError:
			self._static = self._static + other._static 

		try:
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(self._static):
			self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)

		for func,Hd in iteritems(other._dynamic):
			if func in self._dynamic:
				try:
					self._dynamic[func] += Hd
				except NotImplementedError:
					self._dynamic[func] = self._dynamic[func] + Hd

				try:
					self._dynamic[func].sum_duplicates()
					self._dynamic[func].eliminate_zeros()
				except: pass

				if _check_almost_zero(self._dynamic[func]):
					self._dynamic.pop(func)

			else:
				self._dynamic[func] = Hd

		self.check_is_dense()
		return _shallowcopy(self)




	def _sub_hamiltonian(self,other): 
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype)

		new._is_dense = new._is_dense or other._is_dense

		try:
			new._static -= other._static 
		except NotImplementedError:
			new._static = new._static - other._static 

		try:
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass


		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)


		for func,Hd in iteritems(other._dynamic):
			if func in new._dynamic:
				try:
					new._dynamic[func] -= Hd
				except NotImplementedError:
					new._dynamic[func] = new._dynamic[func] - Hd

				try:
					new._dynamic[func].sum_duplicates()
					new._dynamic[func].eliminate_zeros()
				except: pass


				if _check_almost_zero(new._dynamic[func]):
					new._dynamic.pop(func)

			else:
				new._dynamic[func] = -Hd		

		new.check_is_dense()
		return new





	def _isub_hamiltonian(self,other): 
		self._is_dense = self._is_dense or other._is_dense

		try:
			self._static -= other._static 
		except NotImplementedError:
			self._static = self._static - other._static 

		try:
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(self._static):
			self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)



		for func,Hd in iteritems(other._dynamic):
			if func in self._dynamic:
				try:
					self._dynamic[func] -= Hd
				except NotImplementedError:
					self._dynamic[func] = new._dynamic[func] - Hd

				try:
					self._dynamic[func].sum_duplicates()
					self._dynamic[func].eliminate_zeros()
				except: pass

				if _check_almost_zero(new._dynamic[func]):
					self._dynamic.pop(func)

			else:
				self._dynamic[func] = -Hd

	
		self.check_is_dense()
		return _shallowcopy(self)


	def _mul_hamiltonian(self,other): # self * other
		if self.dynamic and other.dynamic:
			new = self.copy()
			return new.__imul__(other)
		elif self.dynamic:
			return self.__mul__(other.static)
		elif other.dynamic:
			return other.__rmul__(self.static)
		else:
			return self.__mul__(other.static)


	def _rmul_hamiltonian(self,other): # other * self
		if self.dynamic and other.dynamic:
			new = other.copy()
			return (new.T.__imul__(self.T)).T #lazy implementation
		elif self.dynamic:
			return self.__rmul__(other.static)
		elif other.dynamic:
			return other.__mul__(self.static)
		else:
			return self.__rmul__(other.static)

	def _imul_hamiltonian(self,other): # self *= other
		if self.dynamic and other.dynamic:
			self._is_dense = self._is_dense or other._is_dense

			new_dynamic_ops = {}
			# create new dynamic operators coming from
			# self.static * other
			for func,Hd in iteritems(other._dynamic):
				if _sp.issparse(self.static):
					Hmul = self.static.dot(Hd)
				elif _sp.issparse(Hd):
					Hmul = self.static * Hd
				else:
					Hmul = _np.matmul(self.static,Hd)

				if not _check_almost_zero(Hmul):
					new_dynamic_ops[func] = Hmul



			for func1,H1 in iteritems(self._dynamic):
				for func2,H2 in iteritems(other._dynamic):

					if _sp.issparse(H1):
						H12 = H1.dot(H2)
					elif _sp.issparse(H2):
						H12 = H1 * H2
					else:
						H12 = _np.matmul(H1,H2)

					func12 = func1 * func2

					if func12 in new_dynamic_ops:
						try:
							new_dynamic_ops[func12] += H12
						except NotImplementedError:
							new_dynamic_ops[func12] = new_dynamic_ops[func12] + H12

						try:
							new_dynamic_ops[func12].sum_duplicates()
							new_dynamic_ops[func12].eliminate_zeros()
						except: pass

						if _check_almost_zero(new_dynamic_ops[func12]):
							new_dynamic_ops.pop(func12)
					else:
						if not _check_almost_zero(H12):
							new_dynamic_ops[func12] = H12


			self._dynamic = new_dynamic_ops
			return _shallowcopy(self)
		elif self.dynamic:
			return self.__imul__(other.static)
		elif other.dynamic:
			return (other.T.__imul__(self.static.T)).T
		else:
			return self.__imul__(other.static)





	#####################
	# sparse operations #
	#####################


	def _add_sparse(self,other):

		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype)

		try:
			new._static += other
		except NotImplementedError:
			new._static = new._static + other

		try:
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		new.check_is_dense()
		return new	


	def _iadd_sparse(self,other):

		try:
			self._static += other
		except NotImplementedError:
			self._static = self._static + other

		try:
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(self._static):
			self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)

		self.check_is_dense()
		return self	
	



	def _sub_sparse(self,other):

		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype)

		try:
			new._static -= other
		except NotImplementedError:
			new._static = new._static - other

		try:
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		new.check_is_dense()
		return new	


	def _isub_sparse(self,other):

		try:
			self._static -= other
		except NotImplementedError:
			self._static = self._static - other

		try:
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(self._static):
			self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)

		self.check_is_dense()
		return self




	def _mul_sparse(self,other):

		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype)

		new._static = new._static * other

		try:
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass


		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)


		for func in list(new._dynamic):
			new._dynamic[func] = new._dynamic[func] * other

			try:
				new._dynamic[func].sum_duplicates()
				new._dynamic[func].eliminate_zeros()
			except: pass

			if _check_almost_zero(new._dynamic[func]):
				new._dynamic.pop(func)

		new.check_is_dense()
		return new





	def _rmul_sparse(self,other):
		# Auxellery function to calculate the right-side multipication with another sparse matrix.

		# find resultant type from product
		dtype = _np.result_type(self._dtype, other.dtype)
		# create a copy of the hamiltonian object with the previous dtype
		new=self.astype(dtype)

		# proform multiplication on all matricies of the new hamiltonian object.

		new._static = other * new._static

		try:
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		for func in list(new._dynamic):
			new._dynamic[func] = other.dot(new._dynamic[func])

			try:
				new._dynamic[func].sum_duplicates()
				new._dynamic[func].eliminate_zeros()
			except: pass

			if _check_almost_zero(new._dynamic[func]):
				new._dynamic.pop(func)


		new.check_is_dense()		
		return new




	def _imul_sparse(self,other):


		self._static =self._static * other

		try:	
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		if _check_almost_zero(self._static):
			self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)


		for func in list(self._dynamic):
			self._dynamic[func] = other.dot(self._dynamic[func])

			try:
				self._dynamic[func].sum_duplicates()
				self._dynamic[func].eliminate_zeros()
			except: pass

			if _check_almost_zero(self._dynamic[func]):
				self._dynamic.pop(func)

		self.check_is_dense()
		return _shallowcopy(self)




	#####################
	# scalar operations #
	#####################



	def _mul_scalar(self,other):
		dtype = _np.result_type(self._dtype, other)
		new=self.astype(dtype)


		new=self.copy()
		try:
			new._static *= other
		except NotImplementedError:
			new._static = new._static * other

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		for func in list(new._dynamic):
			try:
				new._dynamic[func] *= other
			except NotImplementedError:
				new._dynamic[func] = new._dynamic[func] * other

			try:
				new._dynamic[func].sum_duplicates()
				new._dynamic[func].eliminate_zeros()
			except: pass

			if _check_almost_zero(new._dynamic[func]):
				new._dynamic.pop(func)

		new.check_is_dense()
		return new







	def _imul_scalar(self,other):
		if not _np.can_cast(other,self._dtype,casting="same_kind"):
			raise TypeError("cannot cast types")

		try:
			self._static *= other
		except NotImplementedError:
			self._static = self._static * other

		if _check_almost_zero(self._static):
			self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)

		for func in list(self._dynamic):
			try:
				self._dynamic[func] *= other
			except NotImplementedError:
				self._dynamic[func] = self._dynamic[func] * other

			try:
				self._dynamic[func].sum_duplicates()
				self._dynamic[func].eliminate_zeros()
			except: pass

			if _check_almost_zero(self._dynamic[func]):
				self._dynamic.pop(func)

		self.check_is_dense()
		return _shallowcopy(self)



	####################
	# dense operations #
	####################


	def _add_dense(self,other):

		dtype = _np.result_type(self._dtype, other.dtype)

		if dtype not in supported_dtypes:
			return NotImplemented

		new=self.astype(dtype)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)

		try:
			new._static += other
		except:
			new._static = new._static + other

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		new.check_is_dense()
		return new



	def _iadd_dense(self,other):

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		try: 
			self._static += other
		except:
			self._static = new._static + other

		if _check_almost_zero(self._static):
			self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)

		self.check_is_dense()

		return _shallowcopy(self)




	def _sub_dense(self,other):

		dtype = _np.result_type(self._dtype, other.dtype)

		if dtype not in supported_dtypes:
			return NotImplemented

		new=self.astype(dtype)


		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)

		try:
			new._static -= other
		except:
			new._static = new._static - other

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		new.check_is_dense()

		return new



	def _isub_dense(self,other):

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		try:
			self._static -= other
		except:
			self._static = self._static - other

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		self.check_is_dense()
		return _shallowcopy(self)


	def _mul_dense(self,other):

		dtype = _np.result_type(self._dtype, other.dtype)

		if dtype not in supported_dtypes:
			return NotImplemented

		new=self.astype(dtype)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)

		new._static = _np.asarray(new._static.dot(other))

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)

		for func in list(new._dynamic):
			new._dynamic[func] = _np.asarray(new._dynamic[func].dot(other))

			if _check_almost_zero(new._dynamic[func]):
				new._dynamic.pop(func)

		new.check_is_dense()

		return new





	def _rmul_dense(self,other):

		dtype = _np.result_type(self._dtype, other.dtype)

		if dtype not in supported_dtypes:
			return NotImplemented

		new=self.astype(dtype)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		if _sp.issparse(new._static):
			new._static = _np.asarray(other * new._static)
		else:
			new._static = _np.asarray(other.dot(new._static))

		if _check_almost_zero(new._static):
			new._static = _sp.csr_matrix(new._shape,dtype=new._dtype)



		for func in list(new._dynamic):
			if _sp.issparse(new._dynamic[func]):
				new._dynamic[func] = _np.asarray(other * new._dynamic[func])
			else:
				new._dynamic[func] = _np.asarray(other.dot(new._dynamic[func]))
			

			if _check_almost_zero(new._dynamic[func]):
				new._dynamic.pop(func)

		new.check_is_dense()

		return new





	def _imul_dense(self,other):

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		self._static = _np.asarray(self._static.dot(other))

		if _check_almost_zero(self._static):
			self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)

		for func in list(self._dynamic):
			self._dynamic[func] = _np.asarray(self._dynamic[func].dot(other))

			if _check_almost_zero(self._dynamic[func]):
				self._dynamic.pop(func)

		self.check_is_dense()

		return _shallowcopy(self)


	
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




def ishamiltonian(obj):
	"""Checks if instance is object of `hamiltonian` class.

	Parameters
	----------
	obj : 
		Arbitraty python object.

	Returns
	-------
	bool
		Can be either of the following:

		* `True`: `obj` is an instance of `hamiltonian` class.
		* `False`: `obj` is NOT an instance of `hamiltonian` class.

	"""
	return isinstance(obj,hamiltonian)




