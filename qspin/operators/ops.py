#local modules:
from ..basis import spin_basis_1d as _default_basis
from ..basis import isbasis as _isbasis

from .make_hamiltonian import make_static as _make_static
from .make_hamiltonian import make_dynamic as _make_dynamic
from .make_hamiltonian import test_function as _test_function


# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

# needed for exp_op class
from scipy.sparse.linalg import expm_multiply as _expm_multiply

from copy import deepcopy as _deepcopy
import warnings

__all__ = ["hamiltonian","ishamiltonian","commutator","anti_commutator","exp_op","isexp_op"]

def commutator(H1,H2):
	""" This function returns the commutator of two Hamiltonians H1 and H2. """
	if _ishamiltonian(H1) or _ishamiltonian(H2):
		return H1*H2 - H2*H1
	else:
		return H1.dot(H2) - H2.dot(H1)

def anti_commutator(H1,H2):
	""" This function returns the anti-commutator of two Hamiltonians H1 and H2. """
	if _ishamiltonian(H1) or _ishamiltonian(H2):
		return H1*H2 + H2*H1
	else:
		return H1.dot(H2) + H2.dot(H1)




class HamiltonianEfficiencyWarning(Warning):
    pass


#global names:
supported_dtypes=[_np.float32, _np.float64, _np.complex64, _np.complex128]

if hasattr(_np,"float128"): supported_dtypes.append(_np.float128)
if hasattr(_np,"complex256"): supported_dtypes.append(_np.complex256)

supported_dtypes = tuple(supported_dtypes)

def check_static(sub_list):
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
	

def check_dynamic(sub_list):
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


	

def ishamiltonian(obj):
	return isinstance(obj,hamiltonian)

def isexp_op(obj):
	return isinstance(obj,exp_op)


class hamiltonian(object):
	def __init__(self,static_list,dynamic_list,N=None,shape=None,copy=True,check_symm=True,check_herm=True,check_pcon=True,dtype=_np.complex128,**kwargs):
		"""
		This function intializes the Hamtilonian. You can either initialize with symmetries, or an instance of basis.
		Note that if you initialize with a basis it will ignore all symmetry inputs.

		--- arguments ---

		* static_list: (compulsory) list of objects to calculate the static part of hamiltonian operator. The format goes like:

			```python
			static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]
			```
	

		* dynamic_list: (compulsory) list of objects to calculate the dynamic part of the hamiltonian operator.The format goes like:

			```python
			dynamic_list=[[opstr_1,[indx_11,...,indx_1n],func_1,func_1_args],[matrix_2,func_2,func_2_args],...]
			```

			For the dynamic list the ```func``` is the function which goes in front of the matrix or operator given in the same list. ```func_args``` is a tuple of the extra arguments which go into the function to evaluate it like: 
			```python
			f_val = func(t,*func_args)
			```


		* N: (optional) number of sites to create the hamiltonian with.

		* shape: (optional) shape to create the hamiltonian with.

		* copy: (optional) weather or not to copy the values from the input arrays. 

		* check_symm: (optional) flag whether or not to check the operator strings if they obey the given symmetries.

		* check_herm: (optional) flag whether or not to check if the operator strings create hermitian matrix. 

		* check_pcon: (optional) flag whether or not to check if the oeprator string whether or not they conserve magnetization/particles. 

		* dtype: (optional) data type to case the matrices with. 

		* kw_args: extra options to pass to the basis class.

		--- hamiltonian attributes ---: '_. ' below stands for 'object. '

 		* _.ndim: number of dimensions, always 2.
		
		* _.Ns: number of states in the hilbert space.

		* _.get_shape: returns tuple which has the shape of the hamiltonian (Ns,Ns)

		* _.is_dense: return 'True' if the hamiltonian contains a dense matrix as a componnent. 

		* _.dtype: returns the data type of the hamiltonian

		* _.static: return the static part of the hamiltonian 

		* _.dynamic: returns the dynamic parts of the hamiltonian 


		"""

		self._is_dense=False
		self._ndim=2




		if not (dtype in supported_dtypes):
			raise TypeError('hamiltonian does not support type: '+str(dtype))
		else:
			self._dtype=dtype
		


		if type(static_list) in [list,tuple]:
			static_opstr_list=[]
			static_other_list=[]
			for ele in static_list:
				if check_static(ele):
					static_opstr_list.append(ele)
				else:
					static_other_list.append(ele)
		else: 
			raise TypeError('expecting list/tuple of lists/tuples containing opstr and list of indx')

		if type(dynamic_list) in [list,tuple]:
			dynamic_opstr_list=[]
			dynamic_other_list=[]
			for ele in dynamic_list:
				if check_dynamic(ele):
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
				self._dynamic = ()

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
					self._mat_checks(O_a)

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

			n = len(dynamic_other_list)

			for i in xrange(n):
				O,f,f_args = dynamic_other_list[i]
				_test_function(f,f_args)
				if _sp.issparse(O):
					self._mat_checks(O)

					if copy:
						dynamic_other_list[i][0] = dynamic_other_list[i][0].copy().astype(self._dtype)
					else:
						dynamic_other_list[i][0] = dynamic_other_list[i][0].astype(self._dtype)
					
				elif O.__class__ is _np.ndarray:
					self._mat_checks(O)
					self._is_dense=True

					dynamic_other_list[i][0] = _np.asmatrix(dynamic_other_list[i][0],dtype=self._dtype,copy=copy)


				elif O.__class__ is _np.matrix:
					self._mat_checks(O)
					self._is_dense=True

					dynamic_other_list[i][0] = dynamic_other_list[i][0].astype(self._dtype,copy=copy)

				else:
					O_a = _np.asanyarray(O)
					self._mat_checks(O_a)
					self._is_dense=True

					O_a = O_a.astype(self._dtype,copy=copy)

					dynamic_other_list.pop(i)
					dynamic_other_list.insert(i,(O_a,f,f_args))

			self._dynamic += tuple(dynamic_other_list)

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
				self._dynamic = ()


		self.sum_duplicates()

	@property
	def ndim(self):
		return self._ndim
	
	@property
	def Ns(self):
		return self._shape[0]

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
	def static(self):
		return self._static

	@property
	def dynamic(self):
		return self._dynamic

	@property
	def T(self):
		return self.transpose()

	@property
	def H(self):
		return self.getH()



	def sum_duplicates(self):
		"""
		description:
			This function consolidates the list of dynamic, combining matrices which have the same driving function and function arguments.
		"""
		self._dynamic=list(self._dynamic)
		l=len(self._dynamic)
		i=j=0;
		while i < l:
			while j < l:
				if i != j:
					ele1=self._dynamic[i]; ele2=self._dynamic[j]
					if (ele1[1] == ele2[1]) and (ele1[2] == ele2[2]):
						self._dynamic.pop(j)

						if i > j: i -= 1
						self._dynamic.pop(i)
						ele1=list(ele1)

						try:
							ele1[0]+=ele2[0]
						except:
							ele1[0] = ele1[0] + ele2[0]

						try:
							
							ele1[0].sum_duplicates()
							ele1[0].eliminate_zeros()
						except: pass
				
						self._dynamic.insert(i,tuple(ele1))
				l=len(self._dynamic); j+=1
			i+=1;j=0

		l = len(self._dynamic)
		for i in xrange(l):
			try:
				self._dynamic[i][0].tocsr()
				self._dynamic[i][0].sum_duplicates()
				self._dynamic[i][0].eliminate_zeros()
			except: pass

		


		remove=[]
		atol = 100*_np.finfo(self._dtype).eps
		is_dense = False


		if _sp.issparse(self._static):
			if _np.allclose(self._static.data,0,atol=atol):
				self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)
		else:
			if _np.allclose(self._static,0,atol=atol):
				self._static = _sp.csr_matrix(self._shape,dtype=self._dtype)



		for i,(Hd,f,f_args) in enumerate(self._dynamic):
			if _sp.issparse(Hd):
				if _np.allclose(Hd.data,0,atol=atol):
					remove.append(i)
			else:
				if _np.allclose(Hd,0,atol=atol):
					remove.append(i)


		remove.reverse()

		for i in remove:
			self._dynamic.pop(i)

		self._dynamic=tuple(self._dynamic)

		self.check_is_dense()



	def check_is_dense(self):
		is_sparse = _sp.issparse(self._static)
		for Hd,f,f_args in self._dynamic:
			is_sparse *= _sp.issparse(Hd)

		self._is_dense = not is_sparse




	def __SO(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the real time Schrodinger operator -i*H(t)*|V >
		"""

		V_dot = self._static.dot(V)	
		for Hd,f,f_args in self._dynamic:
			V_dot += f(time,*f_args)*(Hd.dot(V))

		return -1j*V_dot

	def __ISO(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the Imaginary time Schrodinger operator -H(t)*|V >
		"""

		V_dot = self._static.dot(V)	
		for Hd,f,f_args in self._dynamic:
			V_dot += f(time,*f_args)*(Hd.dot(V))

		return -V_dot


	def rdot(self,V,time=0,check=True): # V * H(time)
		if self.Ns <= 0:
			return _np.asarray([])

		if ishamiltonian(V):
			raise ValueError("To multiply hamiltonians use '*' operator.")

		if V.ndim > 2:
			raise ValueError("Expecting V.ndim < 3.")


		if not _np.isscalar(time):
			time = _np.asarray(time)
			if V.ndim == 2 and V.shape[0] > 1:
 				if V.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

				if V.shape[0] != len(time):
					raise ValueError("{0} number of vectors do not match length of time vector {1}.".format(V.shape[0],len(time)))
				
				return _np.vstack([self.rdot(v,time=t) for v,t in zip(V[:],time)])

			else:
 				if V.shape[1] != self._shape[0]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
				
				return _np.vstack([self.dot(V,time=t) for t in time])
				
		if not check:
			V_dot = self._static.__rmul__(V)
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd,__rmul__(V))
			return V_dot


		if V.__class__ is _np.ndarray:
			if V.ndim != 2:
				V = V.reshape((1,-1))
				
			if V.shape[1] != self._shape[0]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
	
			V_dot = self._static.__rmul__(V)
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd,__rmul__(V))

		elif _sp.issparse(V):
			if V.shape[1] != self._shape[0]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
	
			V_dot = self._static.__rmul__(V)
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd,__rmul__(V))


		elif V.__class__ is _np.matrix:
			if V.ndim != 2:
				V = V.reshape((1,-1))

			if V.shape[1] != self._shape[0]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			V_dot = self._static.__rmul__(V)
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd,__rmul__(V))

		elif V.__class__ == exp_op: # matrix op right dot

			return V.dot(self.__call__(time),time=time)


		else:
			V = _np.asanyarray(V)

			if V.ndim != 2:
				V = V.reshape((1,-1))

			if V.shape[1] != self._shape[0]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			V_dot = self._static.__rmul__(V)
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd,__rmul__(V))

		return V_dot

		

	def dot(self,V,time=0,check=True):
		"""
		args:
			V, the vector to multiple with
			time=0, the time to evalute drive at.

		description:
			This function does the spare matrix vector multiplication of V with the Hamiltonian evaluated at 
			the specified time. It is faster in this case to multiple each individual parts of the Hamiltonian 
			first, then add all those vectors together.
		"""

		
		if self.Ns <= 0:
			return _np.asarray([])

		if ishamiltonian(V):
			raise ValueError("To multiply hamiltonians use '*' operator.")

		if V.ndim > 2:
			raise ValueError("Expecting V.ndim < 3.")

		if not _np.isscalar(time):
			time = _np.asarray(time)

			if time.ndim > 1:
				raise ValueError("Expecting time.ndim < 2.")

			if V.ndim == 2 and V.shape[1] > 1:
 				if V.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
				
				if V.shape[1] != len(time):
					raise ValueError("{0} number of vectors do not match length of time vector {1}.".format(V.shape[1],len(time)))
				
				V_dot = _np.vstack([self.dot(v,time=t,check=check) for v,t in zip(V.T[:],time)]).T
				return V_dot
			else:
 				if V.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
				
				V_dot = _np.vstack([self.dot(V,time=t,check=check) for t in time]).T
				return V_dot

			

		if not check:
			V_dot = self._static.dot(V)	
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd.dot(V))
			return V_dot




		if V.__class__ is _np.ndarray:
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
	
			V_dot = self._static.dot(V)	
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd.dot(V))


		elif _sp.issparse(V):
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))
	
			V_dot = self._static.dot(V)	
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd.dot(V))
			return V_dot

		elif V.__class__ is _np.matrix:
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			V_dot = self._static.dot(V)	
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd.dot(V))

		elif V.__class__ == exp_op: # matrix op right dot

			return V.rdot(self.__call__(time),time=time)
			"""
			V = V.expm()

			if not _sp.issparse(V):
				raise ValueError("Expecting a sparse matrix for 'V'!")
			
			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			V_dot = self._static.dot(V)	
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd.dot(V))
			return V_dot
			"""
		else:
			V = _np.asanyarray(V)
			if V.ndim not in [1,2]:
				raise ValueError("Expecting 1 or 2 dimensional array")

			if V.shape[0] != self._shape[1]:
				raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V.shape,self._shape))

			V_dot = self._static.dot(V)	
			for Hd,f,f_args in self._dynamic:
				V_dot += f(time,*f_args)*(Hd.dot(V))

		return V_dot





	def matrix_ele(self,Vl,Vr,time=0,diagonal=False,check=True):
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
			return np.array([])

		Vr=self.dot(Vr,time=time,check=check)

		if not check:
			if diagonal:
				return _np.einsum("ij,ij->i",Vl.conj(),Vr)
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
					return _np.einsum("ij,ij->i",Vl.conj(),Vr)
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
					return _np.einsum("ij,ij->i",Vl.conj(),Vr)
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
					return _np.einsum("ij,ij->i",Vl.conj(),Vr)
				else:
					return Vl.conj().dot(Vr)
			elif Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError("matrix dimension mismatch with shapes: {0} and {1}.".format(V1.shape,self._shape))

				return Vl.T.conj().dot(Vr)
			else:
				raise ValueError('Expecting Vl to have ndim < 3')
		

	def project_to(self,proj):
		if isinstance(proj,hamiltonian):
			raise NotImplementedError

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
		if generator:
			return exp_op(other,a=a,time=time,start=start,stop=stop,num=num,endpoint=endpoint,iterate=iterate).sandwich(self)
		else:
			return self.project_to(other)


	def eigsh(self,time=0,**eigsh_args):
		"""
		args:
			time=0, the time to evalute drive at.
			other arguments see documentation: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html
			
		description:
			function which diagonalizes hamiltonian using sparse methods
			solves for eigen values and eigen vectors, but can only solve for a few of them accurately.
			uses the scipy.sparse.linalg.eigsh function which is a wrapper for ARPACK
		"""
		if not _np.isscalar(time):
			raise TypeError('expecting scalar argument for time')

		if self.Ns <= 0:
			return _np.asarray([]), _np.asarray([[]])

		char = _np.dtype(self._dtype).char
		if char == "g":
			H = self.tocsr(time=time).astype(_np.float64)
		elif char == "G": 
			H = self.tocsr(time=time).astype(_np.complex128)
		else:
			H = self.tocsr(time=time)

		return _sla.eigsh(H,**eigsh_args)



	
	def eigh(self,time=0,**eigh_args):
		"""
		args:
			time=0, time to evaluate drive at.

		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values. 
			uses wrapped lapack functions which are contained in module py_lapack
		"""
		eigh_args["overwrite_a"] = True
		
		if not _np.isscalar(time):
			raise TypeError('expecting scalar argument for time')


		if self.Ns <= 0:
			return _np.asarray([]),_np.asarray([[]])

		# fill dense array with hamiltonian
		H_dense = self.todense(time=time)		
		# calculate eigh
		E,H_dense = _la.eigh(H_dense,**eigh_args)
		return E,H_dense


	def eigvalsh(self,time=0,**eigvalsh_args):
		"""
		args:
			time=0, time to evaluate drive at.

		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values 
			and eigen vectors. uses wrapped lapack functions which are contained in module py_lapack
		"""
		eigvalsh_args["overwrite_a"] = True
		
		if not _np.isscalar(time):
			raise TypeError('expecting scalar argument for time')

		if self.Ns <= 0:
			return _np.asarray([])

		H_dense=_np.zeros(self._shape,dtype=self._dtype)
		self.todense(time=time,out=H_dense)

		E = _la.eigvalsh(H_dense,**eigvalsh_args)
		return E





	def evolve(self,v0,t0,times,solver_name="dop853",verbose=False,iterate=False,imag_time=False,**solver_args):
		from scipy.integrate import complex_ode

		if _np.iscomplexobj(times):
			raise ValueError("times must be real number(s).")

		if solver_name in ["dop853","dopri5"]:
			if solver_args.get("nsteps") is None:
				solver_args["nsteps"] = _np.iinfo(_np.int32).max
			if solver_args.get("rtol") is None:
				solver_args["rtol"] = 1E-9
			if solver_args.get("atol") is None:
				solver_args["atol"] = 1E-9



		
		if v0.ndim <= 2:
			v0 = v0.reshape((-1,))
		else:
			raise ValueError("v0 must have ndim <= 2")

		if v0.shape[0] != self.Ns:
			raise ValueError("v0 must have {0} elements".format(self.Ns))

		complex_type = _np.dtype(_np.complex64(1j)*v0[0])
		if imag_time:
			v0 = v0.astype(self.dtype)
			if _np.iscomplexobj(v0):
				solver = ode(self.__ISO)
			else:
				solver = complex_ode(self.__ISO)
		else:
			v0 = v0.astype(complex_type)
			solver = complex_ode(self.__SO)

		

		solver.set_integrator(solver_name,**solver_args)
		solver.set_initial_value(v0, t0)

		if _np.isscalar(times):
			return self._evolve_scalar(solver,v0,t0,times,imag_time)
		else:
			if iterate:
				return self._evolve_iter(solver,v0,t0,times,verbose,imag_time)
			else:
				return self._evolve_list(solver,v0,t0,times,complex_type,verbose,imag_time)

			
		






	def _evolve_scalar(self,solver,v0,t0,time,imag_time):
		from numpy.linalg import norm

		if time == t0:
			return _np.array(v0)
		solver.integrate(time)
		if solver.successful():
			if imag_time: solver._y /= norm(solver._y)
			return _np.array(solver.y)
		else:
			raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(times))	



	def _evolve_list(self,solver,v0,t0,times,complex_type,verbose,imag_time):
		from numpy.linalg import norm

		v = _np.empty((len(times),self.Ns),dtype=complex_type)
		
		for i,t in enumerate(times):
			if t == t0:
				if verbose: print "evolved to time {0}, norm of state {1}".format(t,_np.linalg.norm(solver.y))
				v[i,:] = _np.array(v0)
				continue

			solver.integrate(t)
			if solver.successful():
				if verbose: print "evolved to time {0}, norm of state {1}".format(t,_np.linalg.norm(solver.y))
				if imag_time: solver._y /= norm(solver._y)
				v[i,:] = _np.array(solver.y)
			else:
				raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(t))
				
		return v



	def _evolve_iter(self,solver,v0,t0,times,verbose,imag_time):
		from numpy.linalg import norm

		for i,t in enumerate(times):
			if t == t0:
				if verbose: print "evolved to time {0}, norm of state {1}".format(t,_np.linalg.norm(solver.y))
				yield _np.array(v0)
				continue
				

			solver.integrate(t)
			if solver.successful():
				if verbose: print "evolved to time {0}, norm of state {1}".format(t,_np.linalg.norm(solver.y))
				if imag_time: solver._y /= norm(solver._y)
				yield _np.array(solver.y)
			else:
				raise RuntimeError("failed to evolve to time {0}, nsteps might be too small".format(t))
		

	def tocsr(self,time=0):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a csr_matrix evaluated at the desired time.
		"""
		if self.Ns <= 0:
			return _sp.csr_matrix(_np.asarray([[]]))
		if not _np.isscalar(time):
			raise TypeError('expecting scalar argument for time')


		H = _sp.csr_matrix(self._static)	

		for Hd,f,f_args in self._dynamic:
			Hd = _sp.csr_matrix(Hd)
			try:
				H += Hd * f(time,*f_args)
			except:
				H = H + Hd * f(time,*f_args)


		return H

	def tocsc(self,time=0):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a csr_matrix evaluated at the desired time.
		"""
		if self.Ns <= 0:
			return _sp.csc_matrix(_np.asarray([[]]))
		if not _np.isscalar(time):
			raise TypeError('expecting scalar argument for time')

		H = _sp.csc_matrix(self._static)	
		for Hd,f,f_args in self._dynamic:
			Hd = _sp.csc_matrix(Hd)
			try:
				H += Hd * f(time,*f_args)
			except:
				H = H + Hd * f(time,*f_args)


		return H


	
	def todense(self,time=0,order=None, out=None):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a dense matrix evaluated at the desired time.
			This function can overflow memory if not careful.
		"""

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)
			out = _np.asmatrix(out)

		if _sp.issparse(self._static):
			self._static.todense(order=order,out=out)
		else:
			out[:] = self._static[:]

		for Hd,f,f_args in self._dynamic:
			out += Hd * f(time,*f_args)
		
		return out


	def toarray(self,time=0,order=None, out=None):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a dense matrix evaluated at the desired time.
			This function can overflow memory if not careful.
		"""

		if out is None:
			out = _np.zeros(self._shape,dtype=self.dtype)

		if _sp.issparse(self._static):
			self._static.toarray(order=order,out=out)
		else:
			out[:] = self._static[:]

		for Hd,f,f_args in self._dynamic:
			out += Hd * f(time,*f_args)
		
		return out



	def as_dense_format(self,copy=False):
		if copy:
			return self.copy().asdense()
		else:
			if _sp.issparse(self._static):
				self._static = self._static.todense()
			else:
				self._static = _np.asmatrix(self._static)

			self._dynamic = list(self._dynamic)
			n = len(self._dynamic)
			for i in xrange(n):
				self._dynamic[i] = list(self._dynamic[i])
				if _sp.issparse(self._dynamic[i][0]):
					self._dynamic[i][0] = self._dynamic[i][0].todense()
				else:
					self._dynamic[i][0] = _np.asmatrix(self._dynamic[i][0])

				self._dynamic[i][0] = self._dynamic[i][0].todense()
				self._dynamic[i] = tuple(self._dynamic[i])

			self._dynamic = tuple(self._dynamic)
			return self


	def as_sparse_format(self,fmt,copy=False):
		if type(fmt) is not str:
			raise ValueError("Expecting string for 'fmt'")

		if fmt not in ["csr","csc","dia","bsr"]:
			raise ValueError("'{0}' is not a valid sparse format or does not support arithmetic.".format(fmt))

		if copy:
			return self.copy().asformat_csr(fmt)
		else:
			sparse_constuctor = getattr(_sp,fmt+"_matrix")

			self._static = sparse_constuctor(self._static)
			self._dynamic = list(self._dynamic)
			n = len(self._dynamic)
			for i in xrange(n):
				self._dynamic[i] = list(self._dynamic[i])
				self._dynamic[i][0] = sparse_constuctor(self._dynamic[i][0])
				self._dynamic[i] = tuple(self._dynamic[i])

			self._dynamic = tuple(self._dynamic)
			return self


	def conj(self):
		self._static = self._static.conj()
		self._dynamic = list(self._dynamic)
		n = len(self._dynamic)
		for i in xrange(n):
			self._dynamic[i] = list(self._dynamic[i])
			self._dynamic[i][0] = self._dynamic[i][0].conj()
			self._dynamic[i] = tuple(self._dynamic[i])

		self._dynamic = tuple(self._dynamic)
		return self


	def getH(self,copy=False):
		return self.conj().transpose(copy=copy)


	def transpose(self,copy=False):
		if copy:
			return self.copy().transpose()
		else:
			self._static = self._static.T
			self._dynamic = list(self._dynamic)
			n = len(self._dynamic)
			for i in xrange(n):
				self._dynamic[i] = list(self._dynamic[i])
				self._dynamic[i][0] = self._dynamic[i][0].T
				self._dynamic[i] = tuple(self._dynamic[i])

			self._dynamic = tuple(self._dynamic)
			return self



	def astype(self,dtype,copy=False):
		if copy:
			return self.copy().astype(dtype)
		else:
			self._dtype = dtype
			self._static = self._static.astype(dtype)
			self._dynamic = list(self._dynamic)
			n = len(self._dynamic)
			for i in xrange(n):
				self._dynamic[i] = list(self._dynamic[i])
				self._dynamic[i][0] = self._dynamic[i][0].astype(dtype)
				self._dynamic[i] = tuple(self._dynamic[i])

			self._dynamic = tuple(self._dynamic)
			return self
		




	def copy(self):
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
		for i,(Hd,f,f_args) in enumerate(self._dynamic):
			h_str = Hd.__str__()
			f_args_str = f_args.__str__()
			f_str = f.__name__
			
			string += ("{0}) func: {2}, func_args: {3}, mat: \n{1} \n".format(i,h_str, f_str,f_args_str))

		return string
		

	def __repr__(self):
		if self.is_dense:
			return "<{0}x{1} qspin dense hamiltonian of type '{2}'>".format(*(self._shape[0],self._shape[1],self._dtype))
		else:
			return "<{0}x{1} qspin sprase hamiltonian of type '{2}' stored in {3} format>".format(*(self._shape[0],self._shape[1],self._dtype,self._static.getformat()))


	def __neg__(self): # -self
		self._static = -self._static
		self._dynamic = list(self._dynamic)
		n = len(self._dynamic)
		for i in xrange(n):
			self._dynamic[i][-1] = -self._dynamic[i][-1]

		self._dynamic = tuple(self._dynamic)
		
		return self


	def __call__(self,time):
		if self.is_dense:
			return self.todense(time)
		else:
			return self.tocsr(time)


	##################################
	# symbolic arithmetic operations #
	# currently only have +,-,* like #
	# operators implimented.         #
	##################################



	def __mul__(self,other): # self * other
		if isinstance(other,hamiltonian):			
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
		if isinstance(other,hamiltonian):
#			self._hamiltonian_checks(other,casting="unsafe")
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
		if isinstance(other,hamiltonian):
			self._hamiltonian_checks(other)
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




	def __div__(self,other): # self / other
		if isinstance(other,hamiltonian):			
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
		if isinstance(other,hamiltonian):
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
		if isinstance(other,hamiltonian):
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
		if isinstance(other,hamiltonian):
			self._hamiltonian_checks(other)
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
		if isinstance(other,hamiltonian):
			self._hamiltonian_checks(other,casting="unsafe")
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
		if isinstance(other,hamiltonian):
			self._hamiltonian_checks(other)
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
			if other.shape != self._shape: # only accepts square matricies 
				raise ValueError('shapes do not match')
			if not _np.can_cast(other.dtype,self._dtype,casting=casting):
				raise ValueError('cannot cast types')


	def _hamiltonian_checks(self,other,casting="same_kind"):
			if other._shape != self._shape: # only accepts square matricies 
				raise ValueError('shapes do not match')
			if not _np.can_cast(other.dtype,self._dtype,casting=casting):
				raise ValueError('cannot cast types')
		




	def _add_hamiltonian(self,other): 
#		self._hamiltonian_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)

		new._is_dense = new._is_dense or other._is_dense

		try:
			new._static += other._static 
		except NotImplementedError:
			new._static = new._static + other._static 

		try:
			new._static = new._static.tocsr(copy=False)
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass


		new._dynamic += other._dynamic
		new.sum_duplicates()

		return new




	def _iadd_hamiltonian(self,other):
#		self._hamiltonian_checks(other)
		self._is_dense = self._is_dense or other._is_dense

		try:
			self._static += other._static 
		except NotImplementedError:
			self._static = self._static + other._static 

		try:
			self._static = new._static.tocsr(copy=False)
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		self._dynamic += other._dynamic
		self.sum_duplicates()


		return self




	def _sub_hamiltonian(self,other): 
#		self._hamiltonian_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)

		new._is_dense = new._is_dense or other._is_dense

		try:
			new._static -= other._static 
		except NotImplementedError:
			new._static = new._static - other._static 

		try:
			new._static = new._static.tocsr(copy=False)
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass

		
		a=tuple([(-Hd,f,f_args) for Hd,f,f_args in other._dynamic])
		new._dynamic += a
		new.sum_duplicates()

		return new





	def _isub_hamiltonian(self,other): 
#		self._hamiltonian_checks(other)

		self._is_dense = self._is_dense or other._is_dense

		try:
			self._static -= other._static 
		except NotImplementedError:
			self._static = self._static - other._static 

		try:
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		a=tuple([(-Hd,f,f_args) for Hd,f,f_args in other._dynamic])
		self._dynamic += a
		self.sum_duplicates()
	
		return self


	def _mul_hamiltonian(self,other): # self * other
		if self.dynamic and other.dynamic:
			raise TypeError("unsupported operand type(s) for *: 'hamiltonian' and 'hamiltonian' which both have dynamic parts.")
		elif self.dynamic:
			return self.__mul__(other.static)
		elif other.dynamic:
			return other.__rmul__(self.static)
		else:
			return self.__mul__(other.static)


	def _rmul_hamiltonian(self,other): # other * self
		if self.dynamic and other.dynamic:
			raise TypeError("unsupported operand type(s) for *: 'hamiltonian' and 'hamiltonian' which both have dynamic parts.")
		elif self.dynamic:
			return self.__rmul__(other.static)
		elif other.dynamic:
			return other.__mul__(self.static)
		else:
			return self.__rmul__(other.static)

	def _imul_hamiltonian(self,other): # self *= other
		if self.dynamic and other.dynamic:
			raise TypeError("unsupported operand type(s) for *: 'hamiltonian' and 'hamiltonian' which both have dynamic parts.")
		elif self.dynamic:
			return self.__imul__(other.static)
		elif other.dynamic:
			return other.__rmul__(self.static)
		else:
			return self.__imul__(other.static)





	#####################
	# sparse operations #
	#####################


	def _add_sparse(self,other):
#		self._mat_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)

		try:
			new._static += other
		except NotImplementedError:
			new._static = new._static + other

		try:
			new._static.tocsr()
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass

		new.sum_duplicates()

		return new	


	def _iadd_sparse(self,other):
#		self._mat_checks(other)
		try:
			self._static += other
		except NotImplementedError:
			self._static = self._static + other

		try:
			self._static.tocsr()
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		self.sum_duplicates()

		return self	
	



	def _sub_sparse(self,other):
#		self._mat_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)

		try:
			new._static -= other
		except NotImplementedError:
			new._static = new._static - other

		try:
			new._static.tocsr()
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass

		new.sum_duplicates()

		return new	


	def _isub_sparse(self,other):
#		self._mat_checks(other)
		try:
			self._static -= other
		except NotImplementedError:
			self._static = self._static - other

		try:
			self._static.tocsr()
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass

		self.sum_duplicates()

		return self




	def _mul_sparse(self,other):
#		self._mat_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)

		new._static = new._static * other

		try:
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass

		new._dynamic = list(new._dynamic)
		n = len(new._dynamic)
		
		for i in xrange(n):
			new._dynamic[i] = list(new._dynamic[i])
			new._dynamic[i][0] = new._dynamic[i][0] * other

			try:
				new._dynamic[i][0].tocsr()
				new._dynamic[i][0].sum_duplicates()
				new._dynamic[i][0].eliminate_zeros()
			except: pass

			new._dynamic[i] = tuple(new._dynamic[i])


		new._dynamic = tuple(new._dynamic)

		new.sum_duplicates()

		return new





	def _rmul_sparse(self,other):
#		self._mat_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)


		new._static = other * new._static
		try:
			new._static.tocsr()
			new._static.sum_duplicates()
			new._static.eliminate_zeros()
		except: pass
			

		new._dynamic = list(new._dynamic)
		n = len(new._dynamic)
		for i in xrange(n):
			new._dynamic[i] = list(new._dynamic[i])
			new._dynamic[i][0] = other.dot(new._dynamic[i][0])
			try:
				new._dynamic[i][0].tocsr()
				new._dynamic[i][0].sum_duplicates()
				new._dynamic[i][0].eliminate_zeros()
			except: pass
			new._dynamic[i] = tuple(new._dynamic[i])

		new._dynamic = tuple(new._dynamic)
		new.sum_duplicates()

		return new




	def _imul_sparse(self,other):
#		self._mat_checks(other)

		self._static =self._static * other
		try:	
			self._static.tocsr()
			self._static.sum_duplicates()
			self._static.eliminate_zeros()
		except: pass


		self._dynamic = list(self._dynamic)
		n = len(self._dynamic)
		for i in xrange(n):
			self._dynamic[i] = list(self._dynamic[i])

			self._dynamic[i][0] = self._dynamic[0][i] * other
			try:
				self._dynamic[i][0].tocsr()
				self._dynamic[i][0].sum_duplicates()
				self._dynamic[i][0].eliminate_zeros()
			except: pass

			self._dynamic[i] = tuple(self._dynamic[i])

		self._dynamic = tuple(self._dynamic)

		self.sum_duplicates()

		return self




	#####################
	# scalar operations #
	#####################



	def _mul_scalar(self,other):
		dtype = _np.result_type(self._dtype, other)
		new=self.astype(dtype,copy=True)


		new=self.copy()
		try:
			new._static *= other
		except NotImplementedError:
			new._static = new._static * other

		new._dynamic = list(new._dynamic)
		n = len(new._dynamic)
		
		try:
			for i in xrange(n):
				new._dynamic[i] = list(new._dynamic[i])
				new._dynamic[i][0] *= other
				new._dynamic[i] = tuple(new._dynamic[i])
		except NotImplementedError:
			for i in xrange(n):
				new._dynamic[i] = list(new._dynamic[i])
				new._dynamic[i][0] = new._dynamic[i][0] * other
				new._dynamic[i] = tuple(new._dynamic[i])

		new._dynamic = tuple(new._dynamic)

		new.sum_duplicates()

		return new







	def _imul_scalar(self,other):
		if not _np.can_cast(other,self._dtype,casting="same_kind"):
			raise TypeError("cannot cast types")

		try:
			self._static *= other
		except NotImplementedError:
			self._static = self._static * other

		self._dynamic = list(self._dynamic)
		n = len(self._dynamic)

		try:
			for i in xrange(n):
				self._dynamic[i] = list(self._dynamic[i])
				self._dynamic[i][0] *= other
				self._dynamic[i] = tuple(self._dynamic[i])

		except NotImplementedError:
			for i in xrange(n):
				self._dynamic[i] = list(self._dynamic[i])
				self._dynamic[i][0] = other * self._dynamic[0]
				self._dynamic[i] = tuple(self._dynamic[i])

		self._dynamic = tuple(self._dynamic)

		self.sum_duplicates()

		return self



	####################
	# dense operations #
	####################


	def _add_dense(self,other):
#		self._mat_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)

		try:
			new._static += other
		except:
			new._static = new._static + other

		new.sum_duplicates()
		
		return new	



	def _iadd_dense(self,other):
#		self._mat_checks(other)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		try: 
			self._static += other
		except:
			self._static = new._static + other

		self.sum_duplicates()
		
		return self




	def _sub_dense(self,other):
#		self._mat_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)


		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)

		try:
			new._static -= other
		except:
			new._static = new._static - other

		new.sum_duplicates()
		
		return new



	def _isub_dense(self,other):
#		self._mat_checks(other)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		try:
			self._static -= other
		except:
			self._static = self._static - other

		self.sum_duplicates()
		
		return self





	def _mul_dense(self,other):
#		self._mat_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		new._static = new._static * other

		new._dynamic = list(new._dynamic)
		n = len(new._dynamic)
		for i in xrange(n):
			new._dynamic[i] = list(new._dynamic[i])
			new._dynamic[i][0] = new._dynamic[i][0] * other
			new._dynamic[i] = tuple(new._dynamic[i])

		new._dynamic = tuple(new._dynamic)

		new.sum_duplicates()

		return new





	def _rmul_dense(self,other):
#		self._mat_checks(other,casting="unsafe")
		dtype = _np.result_type(self._dtype, other.dtype)
		new=self.astype(dtype,copy=True)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		if _sp.issparse(new._static):
			new._static = other * new._static
		else:
			new._static = other.dot(new._static)

		new._dynamic = list(new._dynamic)
		n = len(new._dynamic)
		for i in xrange(n):
			new._dynamic[i] = list(new._dynamic[i])
			new._dynamic[i][0] = other * new._dynamic[i][0]
			new._dynamic[i] = tuple(new._dynamic[i])

		new._dynamic = tuple(new._dynamic)

		new.sum_duplicates()

		return new





	def _imul_dense(self,other):
#		self._mat_checks(other)

		if not self._is_dense:
			self._is_dense = True
			warnings.warn("Mixing dense objects will cast internal matrices to dense.",HamiltonianEfficiencyWarning,stacklevel=3)


		self._static = self._static.dot(other)

		self._dynamic = list(self._dynamic)
		n = len(self._dynamic)
		for i in xrange(n):
			self._dynamic[i] = list(self._dynamic[i])
			self._dynamic[i][0] = self._dynamic[i][0].dot(other)
			self._dynamic[i] = tuple(self._dynamic[i])


		self._dynamic = tuple(self._dynamic)

		self.sum_duplicates()

		return self


	
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







class exp_op(object):
	def __init__(self, O, a = 1.0, time = 0.0, start = None, stop = None, num = None, endpoint = None, iterate = False):

		if not _np.isscalar(a):
			raise TypeError('expecting scalar argument for a')


		self._a=_np.complex128(a)
		self._time=time

		self._start=start
		self._stop=stop
		self._num=num
		self._endpoint=endpoint
		self._iterate=iterate

		if self._iterate:
			if [self._start, self._stop] == [None, None]:
				raise ValueError("'iterate' can only be True with time discretization. must specify 'start' and 'stop' points.")

		 	self._grid, self._step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)
		else:
			if [self._start, self._stop] == [None, None]:
				if self._num != None:
					raise ValueError("unexpected arguement 'num'.")

				if self._endpoint != None:
					raise ValueError("unexpected arguement 'endpoint'.")


				self._grid = None
				self._step = None
			else:

				self._grid, self._step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)

		if ishamiltonian(O):
			self._O = O
		else:
			self._O = hamiltonian([O], [])
	
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
			if [self._grid, self._step] == [None, None]:
				raise ValueError("grid must be set in order to set iterate to be True.")

		self._iterate = Value
		

	def get_mat(self,time=0.0):

		if self.O.is_dense:
			return _np.linalg.expm(self._a * self.O.todense(time))
		else:
			return _sp.linalg.expm(self._a * self.O.tocsc(time))

	def dot(self, other, time=0.0):

		is_sp = False
		is_ham = False

		if ishamiltonian(other):
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

		M = self._a * self.O(time)
		if self._iterate:
			if is_ham:
				return _hamiltonian_iter_dot(M, other, self._step, self._grid)
			else:
				return _iter_dot(M, other, self.step, grid)

		else:
			if [self._grid, self._step] == [None, None]:
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
					return _expm_multiply(M, other, start=self._start, stop=self._stop, num=self._num, endpoint=self._endpoint)

	def rdot(self, other, time=0.0):

		is_sp = False
		is_ham = False

		if ishamiltonian(other):
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

		M = (self._a * self.O(time)).T
		if self._iterate:
			if is_ham:
				return _hamiltonian_iter_rdot(M, other.T, self._step, self._grid)
			else:
				return _iter_rdot(M, other.T, self._step, self._grid)
		else:
			if [self._step, self._grid] == [None, None]:

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
					return _expm_multiply(M, other.T, start=self._start, stop=self._stop, num=self._num, endpoint=self._endpoint).T

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

			if is_ham:
				mat_iter = _hamiltonian_iter_sandwich(M, other, self._step, self._grid)
			else:
				mat_iter = _iter_sandwich(M, other, self._step, self._grid)

			return mat_iter
		else:
			if [self._step, self._grid] == [None, None]:

				other = self.dot(other,time=time)
				other = self.H.rdot(other,time=time)
				return other

			else:
				if is_ham:
					mat_iter = _hamiltonian_iter_sandwich(M, other, self._step, self._grid)
				else:
					mat_iter = _iter_sandwich(M, other, self._step, self._grid)

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

	yield other.transpose(copy=True)

	M *= step
	for t in grid[1:]:
		other =  _hamiltonian_rdot(M, other)
		yield other.transpose(copy=True)


def _hamiltonian_iter_sandwich(M, other, step, grid):
	if grid[0] != 0:
		M *= grid[0]
		other = _hamiltonian_dot(M, other)
		other = _hamiltonian_rdot(M, other.T.conj()).T.conj()
		M /= grid[0]

	yield other.copy()

	M *= step
	for t in grid[1:]:
		other = _hamiltonian_dot(M, other)
		other = _hamiltonian_rdot(M, other.T.conj()).T.conj()
		yield other.copy()








