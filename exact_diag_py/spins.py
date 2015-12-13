import sys # needed for sys.stop('error message')
#local modules:
from basis import basis1d
from py_lapack import eigh # used to diagonalize hermitian and symmetric matricies

#python 2.7 modules
from scipy.linalg import norm as _norm
from scipy.sparse import csr_matrix	# the final version the sparse matrices are stored as, good format for dot produces with vectors.
from scipy.sparse.linalg  import eigsh	# needed for the sparse linear algebra packages
from scipy.integrate import complex_ode,ode	# ode solver used in evolve wave function.
from numpy import isscalar,asscaler, vdot, asarray, array, float32, float64, complex64, complex128
from copy import deepcopy



#global names:
supported_dtypes=(float32, float64, complex64, complex128)


def make_static(B,static_list,dtype):
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

	H=csr_matrix(([],([],[])),shape=(B.Ns,B.Ns),dtype=dtype) 
	for alist in static_list:
		opstr=alist[0]
		bonds=alist[1]
		for bond in bonds:
			J=bond[0]
			indx=bond[1:]
			ME,row,col = B.Op(J,dtype,opstr,indx)
			Ht=csr_matrix((ME,(row,col)),shape=(B.Ns,B.Ns),dtype=dtype) 
			H+=Ht
			del Ht
			H.sum_duplicates() # sum duplicate matrix elements
			H.eliminate_zeros() # remove all zero matrix elements
		
	return H 








def make_dynamic(B,dynamic_list,dtype):
	"""
	args:
	dynamic=[[opstr_1,indx_1,func_1],...,[opstr_n,indx_n,func_1]], list of opstr,indx and functions to drive with
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
	_dynamic=[]
	if dynamic_list:
		H=csr_matrix(([],([],[])),shape=(B.Ns,B.Ns),dtype=dtype)
		for alist in dynamic_list:
			opstr=alist[0]
			bonds=alist[1]
			for bond in bonds:
				J=bond[0]
				indx=bond[1:]
				ME,row,col = B.Op(J,dtype,opstr,indx)
				Ht=csr_matrix((ME,(row,col)),shape=(B.Ns,B.Ns),dtype=dtype) 
				H+=Ht
				del Ht
				H.sum_duplicates() # sum duplicate matrix elements
				H.eliminate_zeros() # remove all zero matrix elements
		
			_dynamic.append((alist[2],H))

	return tuple(_dynamic)










class hamiltonian:
	def __init__(self,static_list,dynamic_list,l,dtype=complex128,**basis_params):
		"""
		This function intializes the Hamtilonian. You can either initialize with symmetries, or an instance of Basis1D.
		Note that if you initialize with a basis it will ignore all symmetry inputs.
		"""
		basis=basis_params.get('basis')
		if basis is None: basis=basis1d(l,**basis_params)

		if not isinstance(basis,basis1d):
			raise TypeError('basis is not instance of Basis1D')
		if not (dtype in supported_dtypes):
			raise TypeError('Hamiltonian1D does not support type: '+str(dtype))


		self.l=l
		self.Ns=basis.Ns
		self.dtype=dtype
		if self.Ns > 0:
			self._static=make_static(basis,static_list,dtype)
			self._dynamic=make_dynamic(basis,dynamic_list,dtype)
			self.shape=(self.Ns,self.Ns)
			self.sum_duplicates()



	def sum_duplicates(self):
		"""
		description:
			This function consolidates the list of dynamic, combining matrices which have the same driving function.
		"""
		self._dynamic=list(self._dynamic)
		l=len(self._dynamic)
		i=j=0;
		while i < l:
			while j < l:
				if i != j:
					ele1=self._dynamic[i]; ele2=self._dynamic[j]
					if ele1[0] == ele2[0]:
						self._dynamic.pop(j)
						i=self._dynamic.index(ele1)
						self._dynamic.pop(i)
						ele1=list(ele1)
						ele1[1]+=ele2[1]
						self._dynamic.insert(i,tuple(ele1))
				l=len(self._dynamic); j+=1
			i+=1;j=0
		self._dynamic=tuple(self._dynamic)



	def tocsr(self,time=0):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a csr_matrix evaluated at the desired time.
		"""
		if self.Ns <= 0:
			return csr_matrix(asarray([[]]))
		if not isscalar(time):
			raise NotImplementedError

		H=self._static	
		for ele in self._dynamic:
			H += ele[1]*ele[0](time)

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
			return asarray([[]])
		if not isscalar(time):
			raise NotImplementedError

		return self.tocsr(time=time).todense()





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
			return asarray([])
		if not isscalar(time):
			raise NotImplementedError

		V=asarray(V)
		V_dot = self._static.dot(V)	
		for ele in self._dynamic:
			J=ele[0](time)
			V_dot += J*(ele[1].dot(V))

		return V_dot



	def __dot_ode(self,time,V):
		"""
		args:
			V, the vector to multiple with
			time, the time to evalute drive at.

		description:
			This function is what get's passed into the ode solver. it is private and the above implimentation is recomended 
			for external use. 
		"""
		if self.Ns <= 0:
			return asarray([])
		if not isscalar(time):
			raise NotImplementedError

		V=asarray(V)
		V_dot = self._static.dot(V)	
		for ele in self._dynamic:
			J=ele[0](time)
			V_dot += J*(ele[1].dot(V))

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

		Vl=asarray(Vl)
		Vr=asarray(Vr)
		HVr=self.dot(Vr,time=time)
		me=vdot(Vl,HVr)
		return asscaler(me)





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
			return asarray([]), asarray([[]])

		return eigsh(self.tocsr(time=time),k=k,sigma=sigma,which=which,maxiter=maxiter)





	def eigh(self,time=0):
		"""
		args:
			time=0, time to evaluate drive at.

		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values. 
			uses wrapped lapack functions which are contained in module py_lapack
		"""
		if self.Ns <= 0:
			return asarray([]),asarray([[]])

		return eigh(self.todense(time=time))





	def eigvalsh(self,time=0):
		"""
		args:
			time=0, time to evaluate drive at.

		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values 
			and eigen vectors. uses wrapped lapack functions which are contained in module py_lapack
		"""
		if self.Ns <= 0:
			return asarray([])

		return eigh(self.todense(time=time),JOBZ='N')





	def evolve(self,v0,t0,time,real_time=True,verbose=False,**integrator_params):
		"""
		args:
			v0, intial wavefunction to evolve.
			t0, intial time 
			time, iterable or scalar, or time to evolve v0 to
			real_time, evolve real or imaginary time
			verbose, print times out as you evolve
			**integrator_params, the parameters used for the dop853 explicit rung-kutta solver.
			see documentation http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.integrate.ode.html

		description:
			This function uses complex_ode to evolve an input wavefunction.
		"""
		if self.Ns <= 0:
			return asarray([])

		v0=asarray(v0)

		if real_time:
			solver=complex_ode(-1j*self.__dot_ode)
		else:
			if dtype in [_np.float32,_np.float64]:
				solver=ode(-self.__dot_ode)
			else:
				solver=complex_ode(-self.__dot_ode)

		solver.set_integrator("dop853",**integrator_params)
		solver.set_initial_value(v0,t=t0)
		
		if isscalar(time):
			if time==t0: return v0
			solver.integrate(time)
			if solver.successful():
				return solver.y
			else:
				raise RuntimeError('failed to integrate')		
		else:
			sol=[]
			for t in time:
				if verbose: print t
				if t==t0: 
					sol.append(v0)
					continue
				solver.integrate(t)
				if solver.successful():
					sol.append(solver.y)
				else:
					raise RuntimeError('failed to integrate')
			return sol




	# possiply impliment this in fortran using naive csr matrix vector dot product, might speed things up,
	# but maybe not since the exponential taylor converges pretty quickly. 
	def Exponential(self,V,z,time=0,n=1,atol=10**(-8)):
		"""
		args:
			V, vector to apply the matrix exponential on.
			a, the parameter in the exponential exp(aH)V
			time, time to evaluate drive at.
			n, the number of steps to break the expoential into exp(aH/n)^n V
			error, if the norm the vector of the taylor series is less than this number
			then the taylor series is truncated.

		description:
			this function computes exp(zH)V as a taylor series in zH. not useful for long time evolution.

		"""
		if self.Ns <= 0:
			return asarray([])
		if not isscaler(time):
			raise NotImplementedError

		if n <= 0: raise ValueError('n must be > 0')

		H=self.tocsr(time=time)

		V=asarray(V)
		for j in xrange(n):
			V1=array(V)
			e=1.0; i=1		
			while e > atol:
				V1=(z/(n*i))*self.dot(V1,time=time)
				V+=V1
				if i%2 == 0:
					e=norm(V1)
				i+=1
		return V



	def __add__(self,other):
		if isinstance(other,Hamiltonian1D):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")
			new=deepcopy(self)

			new._static+=other._static
			new._static.sum_duplicates()
			new._static.eliminate_zeros()

			new._dynamic+=other._dynamic
			new.sum_duplicates()
			return new
		else:
			raise Exception("Not Implimented")


	def __iadd__(self,other):
		if isinstance(other,Hamiltonian1D):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")

			self._static+=other._static
			self._static.sum_duplicates()
			self._static.eliminate_zeros()

			self._dynamic+=other._dynamic
			self.sum_duplicates()
			return self
		else:
			raise Exception("Not Implimented")


	def __sub__(self,other):
		if isinstance(other,Hamiltonian1D):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")
			new=deepcopy(self)

			new._static-=other._static
			new._static.sum_duplicates()
			new._static.eliminate_zeros()


			
			a=tuple([(ele[0],-ele[1]) for ele in other._dynamic])
			new._dynamic+=a
			new.sum_duplicates()
			return new
		else:
			raise Exception("Not Implimented")


	def __isub__(self,other):
		if isinstance(other,Hamiltonian1D):
			if self.Ns != other.Ns: raise Exception("cannot add Hamiltonians of different dimensions")

			self._static-=other._static
			self._static.sum_duplicates()
			self._static.eliminate_zeros()


			
			a=tuple([(ele[0],-ele[1]) for ele in other._dynamic])
			self._dynamic+=a
			self.sum_duplicates()
			return self
		else:
			raise Exception("Not Implimented")


	



