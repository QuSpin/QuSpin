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
	def __init__(self,input_dict,N=None,basis=None,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**kwargs):
		"""
		this object maps operatators/matricies to keys, which when calling various operations allows to specify the scalar multiples
		in front of these operators.

		--- arguments ---

		input_dict: dictionary (compulsory) this is a dictionary which should contain values which are operator lists like the
					static_list input to the hamiltonian class while the key's correspond to the key vales which you use
					to specify the coupling during other method calls:

					example:
						input_dict = {
										"Jzz": [["zz",Jzz_bonds]] # use "Jzz" key to specify the zz interaction coupling
										"hx" : [["x" ,hx_site ]] # use "hx" key to specify the field strength
									 }
		* N: (optional) number of sites to create the hamiltonian with.

		* shape: (optional) shape to create the hamiltonian with.

		* copy: (optional) weather or not to copy the values from the input arrays. 

		* check_symm: (optional) flag whether or not to check the operator strings if they obey the given symmetries.

		* check_herm: (optional) flag whether or not to check if the operator strings create hermitian matrix. 

		* check_pcon: (optional) flag whether or not to check if the oeprator string whether or not they conserve magnetization/particles. 

		* dtype: (optional) data type to case the matrices with. 

		* kw_args: extra options to pass to the basis class.		

		--- ops_dict attributes ---: '_. ' below stands for 'object. '		

		* _.basis: the basis associated with this hamiltonian, is None if hamiltonian has no basis. 

		* _.ndim: number of dimensions, always 2.
		
		* _.Ns: number of states in the hilbert space.

		* _.get_shape: returns tuple which has the shape of the hamiltonian (Ns,Ns)

		* _.is_dense: return 'True' if the hamiltonian contains a dense matrix as a componnent. 

		* _.dtype: returns the data type of the hamiltonian

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
					raise ValueError('missing argument shape')
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
		if self._basis is not None:
			return self._basis
		else:
			raise AttributeError("object has no attribute 'basis'")

	@property
	def ndim(self):
		return self._ndim
	
	@property
	def Ns(self):
		return self._Ns

	@property
	def get_shape(self):
		return self._shape

	@property
	def is_dense(self):
		return self._is_dense

	@property
	def dtype(self):
		return _np.dtype(self._dtype).name

	@property
	def T(self):
		return self.transpose()

	@property
	def H(self):
		return self.getH()

	def copy(self,dtype=None):
		return _deepcopy(self)


	def transpose(self,copy = False):
		new = _shallowcopy(self)
		for key,op in self._ops_dict.items():
			new._ops_dict[key] = op.transpose()
		return new

	def conjugate(self):
		new = _shallowcopy(self)
		for key,op in self._ops_dict.items():
			new._ops_dict[key] = op.conj()
		return new

	def conj(self):
		return self.conjugate()

	def getH(self,copy=False):
		return self.conj().transpose(copy=copy)

	def astype(self,dtype):
		if dtype not in hamiltonian.supported_dtypes:
			raise ValueError("operator can only be cast to floating point types")
		new = _shallowcopy(self)
		new._dtype = dtype
		for key in self._ops_dict.keys():
			new._ops_dict[key] = self._ops_dict[key].astype(dtype)

		return new


	def tocsr(self,pars={}):
		pars = self._check_scalar_pars(pars)

		H = _sp.csr_matrix(self.get_shape,dtype=self._dtype)

		for key,J in pars.items():
			try:
				H += J*_sp.csr_matrix(self._ops_dict[key])
			except:
				H = H + J*_sp.csr_matrix(self._ops_dict[key])

		return H

	def tocsc(self,pars={}):
		pars = self._check_scalar_pars(pars)

		H = _sp.csc_matrix(self.get_shape,dtype=self._dtype)

		for key,J in pars.items():
			try:
				H += J*_sp.csc_matrix(self._ops_dict[key])
			except:
				H = H + J*_sp.csc_matrix(self._ops_dict[key])

		return H


	
	def todense(self,out=None,pars={}):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a dense matrix evaluated at the desired time.
			This function can overflow memory if not careful.
		"""
		pars = self._check_scalar_pars(pars)

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)
			out = _np.asmatrix(out)

		for key,J in pars.items():
			out += J * self._ops_dict[key]
		
		return out


	def toarray(self,pars={},out=None):
		"""
		args:
			pars, dictionary to evaluate couples at. 
			out, array to output results too.

		description:
			this function simply returns a copy of the Hamiltonian as a dense matrix evaluated at the desired time.
			This function can overflow memory if not careful.
		"""

		pars = self._check_scalar_pars(pars)

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)

		for key,J in pars.items():
			out += J * self._ops_dict[key]
		
		return out


	def __call__(self,**pars):
		pars = self._check_scalar_pars(pars)
		if self.is_dense:
			return self.todense(pars)
		else:
			return self.tocsr(pars)

	def tohamiltonian(self,pars={}):
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


	def aslinearoperator(self,pars={}):
		pars = self._check_scalar_pars(pars)
		matvec = functools.partial(_ops_dict_dot,self,pars)
		rmatvec = functools.partial(_ops_dict_dot,self.H,pars)
		return _sla.LinearOperator(self.get_shape,matvec,rmatvec=rmatvec,matmat=matvec,dtype=self._dtype)		

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

	def matvec(self,V):
		return self.dot(V)

	def rmatvec(self,V):
		return self.H.dot(V)

	def matmat(self,V):
		return self.dot(V)

	def dot(self,V,pars={},check=True):
		"""
		args:
			V, the vector to multiple with
			pars, dictionary to evaluate couples at. 

		description:
			This function does the spare matrix vector multiplication of V with the Hamiltonian evaluated at 
			the specified time. It is faster in this case to multiple each individual parts of the Hamiltonian 
			first, then add all those vectors together.
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
		try:
			V = V.transpose()
		except AttributeError:
			V = _np.asanyarray(V)
			V = V.transpose()
		return (self.transpose().dot(V)).transpose()

	def matrix_ele(self,Vl,Vr,pars={},diagonal=False,check=True):
		"""
		args:
			Vl, the vector to multiple with on left side
			Vr, the vector to multiple with on the right side
			time=0, the time to evalute drive at.

		description:
			This function takes the matrix element of the Hamiltonian at the specified time
			between Vl and Vr.
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


	def trace(self,pars={}):
		pars = self._check_scalar_pars(pars)
		tr = 0.0
		for key,value in self._operator_dict.items():
			try:
				tr += pars[key] * value.trace()
			except AttributeError:
				tr += pars[key] * value.diagonal().sum()
		return tr


	def eigsh(self,pars={},**eigsh_args):

		if self.Ns == 0:
			return _np.array([]),_np.array([[]])

		return _sla.eigsh(self.tocsr(pars),**eigsh_args)


	def eigh(self,pars={},**eigh_args):
		"""
		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values. 
			uses wrapped lapack functions which are contained in module py_lapack
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
		"""
		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values 
			and eigen vectors. uses wrapped lapack functions which are contained in module py_lapack
		"""

		if self.Ns <= 0:
			return _np.asarray([])

		H_dense = self.todense(pars=pars)
		E = _np.linalg.eigvalsh(H_dense,**eigvalsh_args)
		#eigvalsh_args["overwrite_a"] = True
		#E = _la.eigvalsh(H_dense,**eigvalsh_args)
		return E


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
	return isinstance(obj,ops_dict)

	
