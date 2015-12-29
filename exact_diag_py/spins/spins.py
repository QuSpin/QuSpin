#local modules:
from ..basis import basis1d as _basis1d
from make_hamiltonian import make_static as _make_static
from make_hamiltonian import make_dynamic as _make_dynamic

#python 2.7 modules
from scipy.linalg import norm as _norm

# need linear algebra packages
from scipy.sparse.linalg  import eigsh as _eigsh
from scipy.linalg import eigh as _eigh
from scipy.linalg import eigvalsh as _eigvalsh

import numpy as _np
from copy import deepcopy as _deepcopy

import warnings


#global names:
supported_dtypes=(_np.float32, _np.float64, _np.complex64, _np.complex128)



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
			raise NotImplementedError

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
			raise NotImplementedError

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
			raise NotImplementedError

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

		H_dense=_np.zeros((self.Ns,self.Ns),dtype=self.dtype)
		self.todense(time=time,out=H_dense)

		E,H_dense = _eigh(H_dense,overwrite_a=True)
		return E,H_dense





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

		if self.Ns <= 0:
			return _np.asarray([]),_np.asarray([[]])

		H_dense=_np.zeros((self.Ns,self.Ns),dtype=self.dtype)
		self.todense(time=time,out=H_dense)

		E = _eigvalsh(H_dense,overwrite_a=True)
		return E





	def __add__(self,other):
		if isinstance(other,hamiltonian):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")
			new=_deepcopy(self)

			new.static = new.static + other.static
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

			self.static = self.static + other.static
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

			new.static = new.static - other.static
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

			self.static = self.static - other.static
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
	

	



