#local modules:
from basis import basis1d as _basis1d

# used to diagonalize hermitian and symmetric matricies
from py_lapack import eigh as _eigh
from py_lapack import eigvalsh as _eigvalsh

#python 2.7 modules
from scipy.linalg import norm as _norm

# the final version the sparse matrices are stored as, good format for dot produces with vectors.
from scipy.sparse import csr_matrix as _csr_matrix

# needed for the sparse linear algebra packages
from scipy.sparse.linalg  import eigsh as _eigsh

# ode solver used in evolve wave function.
from scipy.integrate import complex_ode as _complex_ode
from scipy.integrate import ode as _ode

import numpy as _np

from copy import deepcopy as _deepcopy

import warnings


#global names:

supported_dtypes=(_np.float32, _np.float64, _np.complex64, _np.complex128)


def _make_static(basis,static_list,dtype):
	"""
	args:
		static=[[opstr_1,indx_1],...,[opstr_n,indx_n]], list of opstr,indx to add up for static piece of Hamiltonian.
		dtype = the low level C-type which the matrix should store its values with.
	returns:
		H: a csr_matrix representation of the list static

	description:
		this function takes the list static and creates a list of matrix elements is coordinate format. it does
		this by calling the basis method Op which takes a state in the basis, acts with opstr and returns a matrix 
		element and the state which it is connected to. This function is called for every opstr in list static and for every 
		state in the basis until the entire hamiltonian is mapped out. It takes those matrix elements (which need not be 
		sorted or even unique) and creates a coo_matrix from the scipy.sparse library. It then converts this coo_matrix
		to a csr_matrix class which has optimal sparse matrix vector multiplication.
	"""
	Ns=basis.Ns
	H=_csr_matrix(([],([],[])),shape=(Ns,Ns),dtype=dtype) 
	for opstr,bonds in static_list:
		for bond in bonds:
			J=bond[0]
			indx=bond[1:]
			ME,row,col = basis.Op(J,dtype,opstr,indx)
			Ht=_csr_matrix((ME,(row,col)),shape=(Ns,Ns),dtype=dtype) 
			H+=Ht
			del Ht
			H.sum_duplicates() # sum duplicate matrix elements
			H.eliminate_zeros() # remove all zero matrix elements
		
	return H 




def _test_function(func,func_args):
	t=1.0
	func_val=func(t,*func_args)
	if not _np.isscalar(func_val):
		raise TypeError("function must return scaler values")
	if type(func_val) is complex:
		warnings.warn("driving function returing complex value, dynamic hamiltonian will no longer be hermitian object.",UserWarning) 




def _make_dynamic(basis,dynamic_list,dtype):
	"""
	args:
	dynamic=[[opstr_1,indx_1,func_1,func_1_args],...,[opstr_n,indx_n,func_1,func_2_args]], list of opstr,indx and functions to drive with
	dtype = the low level C-type which the matrix should store its values with.

	returns:
	tuple((func_1,H_1),...,(func_n,H_n))

	H_i: a csr_matrix representation of opstr_i,indx_i
	func_i: callable function of time which is the drive term in front of H_i

	description:
		This function works the same as static, but instead of adding all of the elements 
		of the dynamic list together, it returns a tuple which contains each individual csr_matrix 
		representation of all the different driven parts. This way one can construct the time dependent 
		Hamiltonian simply by looping over the tuple returned by this function. 
	"""
	Ns=basis.Ns
	dynamic=[]
	if dynamic_list:
		H=_csr_matrix(([],([],[])),shape=(Ns,Ns),dtype=dtype)
		for opstr,bonds,f,f_args in dynamic_list:
			if _np.isscalar(f_args): raise TypeError("function arguements must be iterable")
			_test_function(f,f_args)
			for bond in bonds:
				J=bond[0]
				indx=bond[1:]
				ME,row,col = basis.Op(J,dtype,opstr,indx)
				Ht=_csr_matrix((ME,(row,col)),shape=(Ns,Ns),dtype=dtype) 
				H+=Ht
				del Ht
				H.sum_duplicates() # sum duplicate matrix elements
				H.eliminate_zeros() # remove all zero matrix elements
		
			dynamic.append((f,f_args,H))

	return tuple(dynamic)










class hamiltonian:
	def __init__(self,static_list,dynamic_list,l,dtype=_np.complex128,**basis_params):
		"""
		This function intializes the Hamtilonian. You can either initialize with symmetries, or an instance of Basis1D.
		Note that if you initialize with a basis it will ignore all symmetry inputs.
		"""
		basis=basis_params.get('basis')
		if basis is None: basis=_basis1d(l,**basis_params)

		if not isinstance(basis,_basis1d):
			raise TypeError('basis is not instance of Basis1D')
		if not (dtype in supported_dtypes):
			raise TypeError('Hamiltonian1D does not support type: '+str(dtype))


		self.l=l
		self.Ns=basis.Ns
		self.dtype=dtype
		if self.Ns > 0:
			self.static=_make_static(basis,static_list,dtype)
			self.dynamic=_make_dynamic(basis,dynamic_list,dtype)
			self.shape=(self.Ns,self.Ns)
			self.sum_duplicates()



	def sum_duplicates(self):
		"""
		description:
			This function consolidates the list of dynamic, combining matrices which have the same driving function.
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
			raise NotImplementedError

		H=self.static	
		for f,f_args,Hd in self.dynamic:
			H += Hd*f(time,*f_args)

		return H


	def todense(self,time=0):
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
			raise NotImplementedError

		return self.tocsr(time=time).todense()



	def __dot_nocheck(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. it is private and the implimentation below is recomended 
			for external use. 
		"""

		V=_np.asarray(V)
		V_dot = self.static.dot(V)	
		for f,f_args,Hd in self.dynamic:
			V_dot += f(time,*f_args)*(Hd.dot(V))

		return V_dot




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
			raise NotImplementedError

		return self.__dot_nocheck(time,V)











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
			raise NotImplementedError


		Vl=_np.asarray(Vl)
		Vr=_np.asarray(Vr)
		HVr=self.__dot_nocheck(time,Vr)
		me=_np.vdot(Vl,HVr)
		return _np.asscalar(me)





	def eigsh(self,time=0,k=6,sigma=None,which='SA',maxiter=10000):
		"""
		args:
			time=0, the time to evalute drive at.
			other arguements see documentation: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.linalg.eigsh.html
			
		description:
			function which diagonalizes hamiltonian using sparse methods
			solves for eigen values and eigen vectors, but can only solve for a few of them accurately.
			uses the scipy.sparse.linalg.eigsh function which is a wrapper for ARPACK
		"""
		if self.Ns <= 0:
			return _np.asarray([]), _np.asarray([[]])

		return _eigsh(self.tocsr(time=time),k=k,sigma=sigma,which=which,maxiter=maxiter)





	def eigh(self,time=0):
		"""
		args:
			time=0, time to evaluate drive at.

		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values. 
			uses wrapped lapack functions which are contained in module py_lapack
		"""
		if self.Ns <= 0:
			return _np.asarray([]),_np.asarray([[]])

		return _eigh(self.todense(time=time))





	def eigvalsh(self,time=0):
		"""
		args:
			time=0, time to evaluate drive at.

		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values 
			and eigen vectors. uses wrapped lapack functions which are contained in module py_lapack
		"""
		if self.Ns <= 0:
			return _np.asarray([])

		return _eigvalsh(self.todense(time=time),JOBZ='N')

	def __add__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")
			new=_deepcopy(self)

			new.static+=other.static
			new.static.sum_duplicates()
			new.static.eliminate_zeros()

			new.dynamic+=other.dynamic
			new.sum_duplicates()
			return new
		else:
			raise Exception("Not Implimented")


	def __iadd__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")

			self.static+=other.static
			self.static.sum_duplicates()
			self.static.eliminate_zeros()

			self.dynamic+=other.dynamic
			self.sum_duplicates()
			return self
		else:
			raise Exception("Not Implimented")


	def __sub__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")
			new=deepcopy(self)

			new.static-=other.static
			new.static.sum_duplicates()
			new.static.eliminate_zeros()


			
			a=tuple([(ele[0],-ele[1]) for ele in other.dynamic])
			new.dynamic+=a
			new.sum_duplicates()
			return new
		else:
			raise Exception("Not Implimented")


	def __isub__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")

			self.static-=other.static
			self.static.sum_duplicates()
			self.static.eliminate_zeros()
			
			a=tuple([(ele[0],-ele[1]) for ele in other.dynamic])
			self.dynamic+=a
			self.sum_duplicates()
			return self
		else:
			raise Exception("Not Implimented")


	
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
	

	



