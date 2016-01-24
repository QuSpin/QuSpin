#local modules:
from ..basis import basis1d as _basis1d
from make_hamiltonian import make_static as _make_static
from make_hamiltonian import make_dynamic as _make_dynamic

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import numpy as _np

from inspect import isfunction as _isfunction
from copy import deepcopy as _deepcopy


#global names:
supported_dtypes=(_np.float32, _np.float64, _np.complex64, _np.complex128)

"""
def static_opstr(sub_list):
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
		if _isspmatrix(sub_list[0]:
			A = sub_list[0]
		else:
			A = _np.asarray(sub_list[0])
		
		if A.ndim != 2: raise ValueError('expecting square matrix')
		if A.shape[0] != A.shape[1]: raise ValueError('expecting square matrix')
		if not _isfunction(sub_list[1]): raise TypeError('expecting callable object for driving function')
		if type(sub_list[2]) not in [list,tuple]: raise TypeError('expecting list for function arguements')
		return False


def dynamic_opstr(sub_list):
	if (type(sub_list) in [list,tuple]):
		if len(sub_list) == 4:
			if type(sub_list[0]) is not str: raise TypeError('expecting string type for opstr')
			if type(sub_list[1]) in [list,tuple]:
				for sub_sub_list in sub_list[1]:
					if (type(sub_sub_list) in [list,tuple]) and (len(sub_sub_list) > 1):
						for element in sub_sub_list:
							if not _np.isscalar(element): raise TypeError('expecting scalar elements of indx')
				else: raise TypeError('expecting list for indx') 
			else: raise TypeError('expecting a list of one or more indx')
			if not _isfunction(sub_list[2]): raise TypeError('expecting callable object for driving function')
			if type(sub_list[3]) not in [list,tuple]: raise TypeError('expecting list for function arguements')
			return True
		if len(sub_list) == 3:
			if _isspmatrix(sub_list[0]:
				A = sub_list[0]
			else:
				A = _np.asarray(sub_list[0])
			
			if A.ndim != 2: raise ValueError('expecting square matrix')
			if A.shape[0] != A.shape[1]: raise ValueError('expecting square matrix')
			if not _isfunction(sub_list[1]): raise TypeError('expecting callable object for driving function')
			if type(sub_list[2]) not in [list,tuple]: raise TypeError('expecting list for function arguements')
			return False
		else:
			raise ValueError('unrecognized hamiltonian constructor usage')
	else: 
"""



	




class hamiltonian:
	def __init__(self,static_list,dynamic_list,L,pauli=True,dtype=_np.complex128,**basis_params):
		"""
		This function intializes the Hamtilonian. You can either initialize with symmetries, or an instance of basis1d.
		Note that if you initialize with a basis it will ignore all symmetry inputs.
		"""

		basis=basis_params.get('basis')
		if basis is None: basis=_basis1d(L,**basis_params)

		if type(L) is not int:
			raise TypeError('expecting integer for L')

		if not isinstance(basis,_basis1d):
			raise TypeError('expecting instance of basis class for basis')
		if not (dtype in supported_dtypes):
			raise TypeError('hamiltonian does not support type: '+str(dtype))

		if type(static_list) in [list,tuple]:
			for sub_list in static_list:
				if (type(sub_list) in [list,tuple]) and (len(sub_list) == 2):
					if type(sub_list[0]) is not str: raise TypeError('expecting string type for opstr')
					if type(sub_list[1]) in [list,tuple]:
						for sub_sub_list in sub_list[1]:
							if (type(sub_sub_list) in [list,tuple]) and (len(sub_sub_list) > 1):
								for element in sub_sub_list:
									if not _np.isscalar(element): raise TypeError('expecting scalar elements of indx')
							else: raise TypeError('expecting list for indx') 
					else: raise TypeError('expecting a list of one or more indx')
				else: raise TypeError('expecting list containing opstr and list of indx')
		else: raise TypeError('expecting list/tuple of lists/tuples containing opstr and list of indx')

		if type(dynamic_list) in [list,tuple]:
			for sub_list in dynamic_list:
				if (type(sub_list) in [list,tuple]) and (len(sub_list) == 4):
					if type(sub_list[0]) is not str: raise TypeError('expecting string type for opstr')
					if type(sub_list[1]) in [list,tuple]:
						for sub_sub_list in sub_list[1]:
							if (type(sub_sub_list) in [list,tuple]) and (len(sub_sub_list) > 1):
								for element in sub_sub_list:
									if not _np.isscalar(element): raise TypeError('expecting scalar elements of indx')
							else: raise TypeError('expecting list for indx') 
					else: raise TypeError('expecting a list of one or more indx')
					if not _isfunction(sub_list[2]): raise TypeError('expecting callable object for driving function')
					if type(sub_list[3]) not in [list,tuple]: raise TypeError('expecting list for function arguements')
				else: raise TypeError('expecting list containing opstr, list of one or more indx, a callable function, and list function args')
		else: raise TypeError('expecting list/tuple of lists/tuples containing opstr and list of indx, functions, and function args')


		self.L = L
		self.Ns = basis.Ns
		self.blocks = basis.blocks
		self.dtype = dtype
		if self.Ns > 0:
			self.static=_make_static(basis,static_list,dtype,pauli)
			self.dynamic=_make_dynamic(basis,dynamic_list,dtype,pauli)
			self.shape=(self.Ns,self.Ns)
			self.sum_duplicates()



	def sum_duplicates(self):
		"""
		description:
			This function consolidates the list of dynamic, combining matrices which have the same driving function and function arguements.
		"""
		self.dynamic=list(self.dynamic)
		l=len(self.dynamic)
		i=j=0;
		while i < l:
			while j < l:
				if i != j:
					ele1=self.dynamic[i]; ele2=self.dynamic[j]
					if (ele1[0] == ele2[0]) and (ele1[1] == ele2[1]):
						self.dynamic.pop(j)
						i=self.dynamic.index(ele1)
						self.dynamic.pop(i)
						ele1=list(ele1)
						ele1[2]+=ele2[2]
						self.dynamic.insert(i,tuple(ele1))
				l=len(self.dynamic); j+=1
			i+=1;j=0
		self.dynamic=tuple(self.dynamic)



	def tocsr(self,time=0):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a csr_matrix evaluated at the desired time.
		"""
		if self.Ns <= 0:
			return _csr_matrix(_np.asarray([[]]))
		if not _np.isscalar(time):
			raise TypeError('expecting scalar arguement for time')

		H=self.static	
		for f,f_args,Hd in self.dynamic:
			H += Hd*f(time,*f_args)

		return H


	def todense(self,time=0,order=None, out=None):
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

		return self.tocsr(time=time).todense(order=order,out=out)



	def __SO(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. This is the Imaginary time Schrodinger operator -i*H(t)*|V >
		"""

		V=_np.asarray(V)
		V_dot = self.static.dot(V)	
		for f,f_args,Hd in self.dynamic:
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

		V=_np.asarray(V)
		V_dot = self.static.dot(V)	
		for f,f_args,Hd in self.dynamic:
			V_dot += f(time,*f_args)*(Hd.dot(V))

		return -V_dot





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

		V=_np.asarray(V)
		V_dot = self.static.dot(V)	
		for f,f_args,Hd in self.dynamic:
			V_dot += f(time,*f_args)*(Hd.dot(V))

		return V_dot





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
			return None
		if not _np.isscalar(time):
			raise TypeError('expecting scalar arguement for time')


		Vl=_np.asarray(Vl).T.conj()
		Vr=_np.asarray(Vr)
		Vr=self.dot(Vr,time=time)
		me=Vl.dot(Vr)
		return me





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

		return _sla.eigsh(self.tocsr(time=time),**eigsh_args)




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

		H_dense=_np.zeros((self.Ns,self.Ns),dtype=self.dtype)
		self.todense(time=time,out=H_dense)

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

		H_dense=_np.zeros((self.Ns,self.Ns),dtype=self.dtype)
		self.todense(time=time,out=H_dense)

		E = _la.eigvalsh(H_dense,overwrite_a=True)
		return E





	def __add__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: 
				raise ValueError("cannot add Hamiltonians of dimensions: {0} {1}".format(self.shape,other.shape))
			if not _np.can_cast(other.dtype,self.dtype): 
				raise TypeError("cannot cast to proper type")
			new=_deepcopy(self)

			new.static = new.static + other.static
			new.static.sum_duplicates()
			new.static.eliminate_zeros()

			new.dynamic+=other.dynamic
			new.sum_duplicates()
			return new
		else:
			raise NotImplementedError


	def __iadd__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: 
				raise ValueError("cannot add Hamiltonians of dimensions: {0} {1}".format(self.shape,other.shape))
			if not _np.can_cast(other.dtype,self.dtype): 
				raise TypeError("cannot cast to proper type")

			self.static = self.static + other.static
			self.static.sum_duplicates()
			self.static.eliminate_zeros()

			self.dynamic+=other.dynamic
			self.sum_duplicates()
			return self
		else:
			raise NotImplementedError


	def __sub__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: 
				raise ValueError("cannot add Hamiltonians of dimensions: {0} {1}".format(self.shape,other.shape))
			if not _np.can_cast(other.dtype,self.dtype): 
				raise TypeError("cannot cast to proper type")
			new=deepcopy(self)

			new.static = new.static - other.static
			new.static.sum_duplicates()
			new.static.eliminate_zeros()


			
			a=tuple([(ele[0],-ele[1]) for ele in other.dynamic])
			new.dynamic+=a
			new.sum_duplicates()
			return new
		else:
			raise NotImplementedError


	def __isub__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: 
				raise ValueError("cannot add Hamiltonians of dimensions: {0} {1}".format(self.shape,other.shape))
			if not _np.can_cast(other.dtype,self.dtype): 
				raise TypeError("cannot cast to proper type")

			self.static = self.static - other.static
			self.static.sum_duplicates()
			self.static.eliminate_zeros()
			
			a=tuple([(ele[0],-ele[1]) for ele in other.dynamic])
			self.dynamic+=a
			self.sum_duplicates()
			return self
		else:
			raise NotImplementedError

	
	def __eq__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns:
				return False

			compare = self.static != other.static			
			if compare.nnz > 0:
				return False

			for e1,e2 in zip(self.dynamic,other.dynamic):
				f1,f1_args,Hd1=e1
				f2,f2_args,Hd2=e2

				if f1 != f2:
					return False
				if f1_args != f2_args:
					return False

				compare = Hd1 != Hd2
				if compare.nnz > 0:
					return False		

			return True
		else:
			return False

	def __ne__(self,other):
		return not self.__eq__(other)
	

	



