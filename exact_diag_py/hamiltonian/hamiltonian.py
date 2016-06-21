#local modules:
from ..basis import basis1d as _default_basis
from ..basis import spin_photon as _spin_photon_basis

from ..basis import isbasis as _isbasis

from make_hamiltonian import make_static as _make_static
from make_hamiltonian import make_dynamic as _make_dynamic
from make_hamiltonian import test_function as _test_function

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

from copy import deepcopy as _deepcopy
import warnings


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
				if (type(sub_sub_list) in [list,tuple]) and (len(sub_sub_list) > 1):
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
				if (type(sub_sub_list) in [list,tuple]) and (len(sub_sub_list) > 1):
					for element in sub_sub_list:
						if not _np.isscalar(element): raise TypeError('expecting scalar elements of indx')
				else: raise TypeError('expecting list for indx') 
		else: raise TypeError('expecting a list of one or more indx')
		if not hasattr(sub_list[2],"__call__"): raise TypeError('expecting callable object for driving function')
		if type(sub_list[3]) not in [list,tuple]: raise TypeError('expecting list for function arguements')
		return True
	elif (type(sub_list) in [list,tuple]) and (len(sub_list) == 3): 
		if not hasattr(sub_list[1],"__call__"): raise TypeError('expecting callable object for driving function')
		if type(sub_list[2]) not in [list,tuple]: raise TypeError('expecting list for function arguements')
		return False
	else:
		raise TypeError('expecting list with object, driving function, and function arguements')


	




class hamiltonian(object):
	def __init__(self,static_list,dynamic_list,L=None,shape=None,pauli=True,photon=False,Ntot=None,n_ph=0,copy=True,dtype=_np.complex128,**kwargs):
		"""
		This function intializes the Hamtilonian. You can either initialize with symmetries, or an instance of basis1d.
		Note that if you initialize with a basis it will ignore all symmetry inputs.
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

		# if any operator strings present must get basis.
		if static_opstr_list or dynamic_opstr_list:
			# check if user input basis
			basis=kwargs.get('basis')	

			# if not
			if basis is None: 
				if L is None: # if L is missing 
					raise Exception('if opstrs in use, arguement L needed for basis class')

				if type(L) is not int: # if L is not int
					raise TypeError('argument L must be integer')

				if Ntot is not None:
					if type(Ntot) is not int:
						raise TypeError('argument Ntot must be integer')

				if n_ph is not None:
					if type(n_ph) is not int:
						raise TypeError('argument Ntot must be integer')

				if not photon:
					basis=_default_basis(L,**kwargs)
				else:
					basis=_spin_photon_basis(L,Ntot,n_ph,**kwargs)

			elif not _isbasis(basis):
				raise TypeError('expecting instance of basis class for arguement: basis')

			self._static=_make_static(basis,static_opstr_list,dtype,pauli)
			self._dynamic=_make_dynamic(basis,dynamic_opstr_list,dtype,pauli)
			self._shape = self._static.shape



		if static_other_list or dynamic_other_list:
			if not hasattr(self,"_shape"):
				found = False
				if shape is None: # if no shape arguement found, search to see if the inputs have shapes.
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

					dynamic_other_list[i][0] = dynamic_other_list[i][0].astype(self._dtype,copy=copy)


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
					raise ValueError('missing arguement shape')
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

	def sum_duplicates(self):
#		print self._dynamic
		"""
		description:
			This function consolidates the list of dynamic, combining matrices which have the same driving function and function arguements.
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
		atol = 10*_np.finfo(self._dtype).eps


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



	def tocsr(self,time=0,dtype=None):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a csr_matrix evaluated at the desired time.
		"""
		if self.Ns <= 0:
			return _sp.csr_matrix(_np.asarray([[]]))
		if not _np.isscalar(time):
			raise TypeError('expecting scalar arguement for time')

		if dtype is None:
			dtype = self._dtype


		if _sp.issparse(self._static):
			H=self._static.tocsr(copy=True)
		else:
			H = _sp.csr_matrix(self._static)	

		for Hd,f,f_args in self._dynamic:
			if _sp.issparse(Hd):
				try:
					H += Hd.tocsr() * f(time,*f_args)
				except:
					H = H + Hd.tocsr() * f(time,*f_args)

			else:
				Hd = _sp.csr_matrix(Hd)
				try:
					H += Hd * f(time,*f_args)
				except:
					H = H + Hd * f(time,*f_args)


		return H


	def todense(self,time=0,order="C", out=None):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a dense matrix evaluated at the desired time.
			This function can overflow memory if not careful.
		"""
		if self.Ns <= 0:
			return _np.asarray([[]])
		if not _np.isscalar(time):
			raise TypeError('expecting scalar arguement for time')

		if out is None:
			if _sp.issparse(self._static):
				H=self._static.todense()
			else:
				H = _np.array(self._static)

			for Hd,f,f_args in self._dynamic:
				if _sp.issparse(Hd):
					H = H + (Hd * f(time,*f_args))
				else:		
					H += Hd * f(time,*f_args)

			return H
		else:
			if _sp.issparse(self._static):
				self._static.todense(out=out)
			else:
				H = _np.copyto(out,self._static,casting='same_kind')

			for Hd,f,f_args in self._dynamic:
				if _sp.issparse(Hd):
					out = out + (Hd * f(time,*f_args))
				else:		
					out += (Hd * f(time,*f_args))		



	def __SO(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the Imaginary time Schrodinger operator -i*H(t)*|V >
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


	def expm_multiply(self,V,a=-1j,time=0,**linspace_args):
		if self.Ns <= 0:
			return _np.asarray([])
		if not _np.isscalar(time):
			raise TypeError('expecting scalar arguement for time')
		if not _np.isscalar(a):
			raise TypeError('expecting scalar arguement for a')

		return _sp.linalg.expm_multiply(a*self.tocsr(time),V,**linspace_args)



	def dot(self,V,time=0):
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
		if not _np.isscalar(time):
			raise TypeError('expecting scalar arguement for time')

		if V.__class__ is _np.ndarray:
			if V.shape[0] != self._shape[1]:
				raise ValueError('dimension mismatch')
	
			return self._dot_nocheck(V)

		elif _sm.issparse(V):
			if V.shape[0] != self._shape[1]:
				raise ValueError('dimension mismatch')
	
			return self._dot_nocheck(V)			

		elif V.__class__ is _np.matrix:
			if V.shape[0] != self._shape[1]:
				raise ValueError('dimension mismatch')

			return self._dot_nocheck(V)

		else:
			V = _np.asanyarray(V)

			if V.shape[0] != self._shape[1]:
				raise ValueError('dimension mismatch')

			return self._dot_nocheck(V)	





	def me(self,Vl,Vr,time=0):
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
		if not _np.isscalar(time):
			raise TypeError('expecting scalar arguement for time')

		Vr=self.dot(Vr,time=time)
		
		if Vl.__class__ is _np.ndarray:
			if Vl.ndim == 1:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError('dimension mismatch')

				return Vl.conj().dot(Vr)
			elif Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError('dimension mismatch')

				return Vl.T.conj().dot(Vr)
			else:
				raise ValueError('Expecting Vl to have ndim < 3')

		elif Vl.__class__ is _np.matrix:
			if Vl.ndim == 1:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError('dimension mismatch')

				return Vl.conj().dot(Vr)
			elif Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError('dimension mismatch')

				return Vl.H.dot(Vr)
			else:
				raise ValueError('Expecting Vl to have ndim < 3')

		elif _sm.issparse(Vl):
			if Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError('dimension mismatch')

				return Vl.H.dot(Vr)
			else:
				raise ValueError('Expecting Vl to have ndim < 3')

		else:
			Vl = _np.asanyarray(Vl)
			if Vl.ndim == 1:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError('dimension mismatch')

				return Vl.conj().dot(Vr)
			elif Vl.ndim == 2:
				if Vl.shape[0] != self._shape[1]:
					raise ValueError('dimension mismatch')

				return Vl.T.conj().dot(Vr)
			else:
				raise ValueError('Expecting Vl to have ndim < 3')

		
	def _dot_nocheck(self,V):
		V_dot = self._static.dot(V)	
		for Hd,f,f_args in self._dynamic:
			V_dot += f(time,*f_args)*(Hd.dot(V))
		return V_dot
		




	def eigsh(self,time=0,**eigsh_args):
		"""
		args:
			time=0, the time to evalute drive at.
			other arguements see documentation: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html
			
		description:
			function which diagonalizes hamiltonian using sparse methods
			solves for eigen values and eigen vectors, but can only solve for a few of them accurately.
			uses the scipy.sparse.linalg.eigsh function which is a wrapper for ARPACK
		"""
		if not _np.isscalar(time):
			raise TypeError('expecting scalar arguement for time')

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
			raise TypeError('expecting scalar arguement for time')


		if self.Ns <= 0:
			return _np.asarray([]),_np.asarray([[]])

		# allocate space
		H_dense=_np.zeros(self._shape,dtype=self._dtype)
		# fill dense array with hamiltonian
		self.todense(time=time,out=H_dense)
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
			raise TypeError('expecting scalar arguement for time')

		if self.Ns <= 0:
			return _np.asarray([])

		H_dense=_np.zeros(self._shape,dtype=self._dtype)
		self.todense(time=time,out=H_dense)

		E = _la.eigvalsh(H_dense,overwrite_a=True)
		return E


	def astype(self,dtype,copy=False):
		if copy:
			return self.copy().astype(dtype)
		else:
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


	def __str__(self):
		string = "static mat: \n{0}\n\n\ndynamic:\n".format(self._static.__str__())
		for i,(Hd,f,f_args) in enumerate(self._dynamic):
			h_str = Hd.__str__()
			f_args_str = f_args.__str__()
			f_str = f.__name__
			
			string += ("{0} func: {2}, func_args: {3}, mat: \n{1} \n".format(i,h_str, f_str,f_args_str))

		return string
		

	def __repr__(self):
		if self.is_dense:
			return "<{0}x{1} exact_diag_py dense hamiltonian of type '{2}'>".format(*(self._shape[0],self._shape[1],self._dtype))
		else:
			return "<{0}x{1} exact_diag_py sprase hamiltonian of type '{2}' stored in {3} format>".format(*(self._shape[0],self._shape[1],self._dtype,self._static.getformat()))


	def __neg__(self): # -self
		self._static = -self._static
		self._dynamic = list(self._dynamic)
		n = len(self._dynamic)
		for i in xrange(n):
			self._dynamic[i][-1] = -self._dynamic[i][-1]

		self._dynamic = tuple(self._dynamic)
		
		return self


	##################################
	# symbolic arithmetic operations #
	# currently only have +,-,* like #
	# operators implimented.         #
	##################################



	def __mul__(self,other): # self * other
		if isinstance(other,hamiltonian):
			return self._mul_hamiltonian(other)

		elif _sp.issparse(other):
			return self._mul_sparse(other)

		elif _np.isscalar(other):
			return self._mul_scalar(other)

		elif other.__class__ == _np.ndarray:
			return self._mul_dense(other)

		elif other.__class__ == _np.matrix:
			return self._mul_dense(other)

		else:
			other = _np.asanyarray(other)
			return self._mul_dense(other)






	def __rmul__(self,other): # other * self
		if isinstance(other,hamiltonian):
			return self._rmul_hamiltonian(other)

		elif _sp.issparse(other):
			return self._rmul_sparse(other)

		elif _np.isscalar(other):
			return self._mul_scalar(other)

		elif other.__class__ == _np.ndarray:
			return self._rmul_dense(other)

		elif other.__class__ == _np.matrix:
			return self._rmul_dense(other)

		else:
			other = _np.asanyarray(other)
			return self._rmul_dense(other)







	def __imul__(self,other): # self *= other
		if isinstance(other,hamiltonian):

			return self._imul_hamiltonian(other)

		elif _sp.issparse(other):

			return self._imul_sparse(other)

		elif _np.isscalar(other):
			return self._imul_scalar(other)

		elif other.__class__ == _np.ndarray:
			return self._imul_dense(other)

		elif other.__class__ == _np.matrix:
			return self._imul_dense(other)

		else:
			other = _np.asanyarray(other)
			return self._imul_dense(other)






	def __add__(self,other): # self + other
		if isinstance(other,hamiltonian):
			return self._add_hamiltonian(other)

		elif _sp.issparse(other):
			return self._add_sparse(other)
			
		elif _np.isscalar(other):
			raise NotImplementedError('hamiltonian does not support addition by scalar')

		elif other.__class__ == _np.ndarray:
			return self._add_dense(other)

		elif other.__class__ == _np.matrix:
			return self._add_dense(other)

		else:
			other = _np.asanyarray(other)
			return self._add_dense(other)





	def __radd__(self,other): # other + self
		return self.__add__(other)






	def __iadd__(self,other): # self += other
		if isinstance(other,hamiltonian):
			return self._iadd_hamiltonian(other)

		elif _sp.issparse(other):
			return self._iadd_sparse(other)

		elif _np.isscalar(other):
			raise NotImplementedError('hamiltonian does not support addition by scalar')

		elif other.__class__ == _np.ndarray:
			return self._iadd_dense(other)

		else:
			other = _np.asanyarray(other)			
			return self._iadd_dense(other)






	def __sub__(self,other): # self - other
		if isinstance(other,hamiltonian):
			return self._sub_hamiltonian(other)

		elif _sp.issparse(other):
			return self._sub_sparse(other)

		elif _np.isscalar(other):
			raise NotImplementedError('hamiltonian does not support addition by scalar')

		elif other.__class__ == _np.ndarray:
			return self._sub_dense(other)

		else:
			other = _np.asanyarray(other)
			return self._sub_dense(other)



	def __rsub__(self,other): # other - self
		# NOTE: because we use signed types this is possble
		return self.__sub__(other).__neg__()




	def __isub__(self,other): # self -= other
		if isinstance(other,hamiltonian):
			return self._isub_hamiltonian(other)

		elif _sp.issparse(other):			
			return self._isub_sparse(other)

		elif _np.isscalar(other):
			raise NotImplementedError('hamiltonian does not support addition by scalar')

		elif other.__class__ == _np.ndarray:
			return self._isub_dense(other)

		else:
			other = _np.asanyarray(other)
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
		self._hamiltonian_checks(other,casting="unsafe")
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
		self._hamiltonian_checks(other)
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
		self._hamiltonian_checks(other,casting="unsafe")
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
		self._hamiltonian_checks(other)

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
		return NotImplemented

	def _rmul_hamiltonian(self,other): # other * self
		return NotImplemented

	def _imul_hamiltonian(self,other): # self *= other
		return NotImplemented




	#####################
	# sparse operations #
	#####################


	def _add_sparse(self,other):
		self._mat_checks(other,casting="unsafe")
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
		self._mat_checks(other)
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
		self._mat_checks(other,casting="unsafe")
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
		self._mat_checks(other)
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
		self._mat_checks(other,casting="unsafe")
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
		self._mat_checks(other,casting="unsafe")
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
		self._mat_checks(other)

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
		self._mat_checks(other,casting="unsafe")
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
		self._mat_checks(other)

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
		self._mat_checks(other,casting="unsafe")
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
		self._mat_checks(other)

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
		self._mat_checks(other,casting="unsafe")
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
		self._mat_checks(other,casting="unsafe")
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
		self._mat_checks(other)

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


