import sys # needed for sys.stop('error message')
#local modules:
from Basis import Basis1D
from py_lapack import eigh # used to diagonalize hermitian and symmetric matricies

#python 2.7 modules
from scipy.linalg import norm
from scipy.sparse import coo_matrix	# needed as the initial format that the Hamiltonian matrices are stored as
from scipy.sparse import csr_matrix	# the final version the sparse matrices are stored as, good format for dot produces with vectors.
from scipy.sparse.linalg  import eigsh, LinearOperator	# needed for the sparse linear algebra packages
from scipy.integrate import complex_ode	# ode solver used in evolve function.

from numpy import pi, asarray, array, int32, int64, float32, float64, complex64, complex128, dot
from collections import Iterable


#global names:
supported_dtypes=(float32, float64, complex64, complex128)



def StaticH(B,static,dtype):
	"""
	args:
		static=[[opstr_1,indx_1],...,[opstr_n,indx_n]], list of opstr,indx to add up for static piece of Hamiltonian.
		dtype = the low level C-type which the matrix should store its values with.
	returns:
		H: a csr_matrix representation of the list static

	description:
		this function takes the list static and creates a list of matrix elements is coordinate format. it does
		this by calling the basis method Op which takes a state in the basis, acts with opstr and returns a matrix 
		element and the state which it is connected to. This function is called for ever opstr in static and for every 
		state in the basis until the entire hamiltonian is mapped out. It takes those matrix elements (which need not be 
		sorted or even unique) and creates a coo_matrix from the scipy.sparse library. It then converts this coo_matrix
		to a csr_matrix class which has optimal sparse matrix vector multiplication.
	"""
	ME_list=[] # this is a list which stores the matrix elements as lists [[row,col,ME],...] for the whole hamiltonian. 
	st=xrange(B.Ns) # iterator which loops over the index of the reference states of the basis B.
	for i in xrange(len(static)): 
		List=static[i]
		opstr=List[0]
		bonds=List[1]
		for bond in bonds:
			J=bond[0]
			indx=bond[1:]
			ME_list.extend(map(lambda x:B.Op(J,x,opstr,indx),st))

	if static: # if static is not an empty list []:
		# there is no way to tranpose a list so we must convert to array, this process will convert all parts of the list to the most compatible type.
		ME_list=asarray(ME_list).T.tolist() # transpose list so that it is now [[row,...],[col,...],[ME,...]] which is how coo_matrix is constructed.
		ME_list[1]=map( lambda a:int(abs(a)), ME_list[1]) # convert the indices back to integers 
		ME_list[2]=map( lambda a:int(abs(a)), ME_list[2])	# convert the indices back to integers
		H=coo_matrix((ME_list[0],(ME_list[1],ME_list[2])),shape=(B.Ns,B.Ns),dtype=dtype) # construct coo_matrix
		H=H.tocsr() # convert to csr_matrix
		H.sum_duplicates() # sum duplicate matrix elements
		H.eliminate_zeros() # remove all zero matrix elements
		return H 
	else: # else return None which indicates there is no static part of Hamiltonian.
		return None








def DynamicHs(B,dynamic,dtype):
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
	Dynamic_Hs=[]
	st=[ k for k in xrange(B.Ns) ]
	for i in xrange(len(dynamic)):
		ME_list=[]
		List=dynamic[i]
		opstr=List[0]
		bonds=List[1]
		for bond in bonds:
			J=bond[0]
			indx=bond[1:]
			ME_list.extend(map(lambda x:B.Op(J,x,opstr,indx),st))
	
		ME_list=asarray(ME_list).T.tolist()
		ME_list[1]=map( lambda a:int(abs(a)), ME_list[1])
		ME_list[2]=map( lambda a:int(abs(a)), ME_list[2])
		H=coo_matrix((ME_list[0],(ME_list[1],ME_list[2])),shape=(B.Ns,B.Ns),dtype=dtype)
		H=H.tocsr()
		H.sum_duplicates()
		H.eliminate_zeros()
		Dynamic_Hs.append((List[2],H))

	return tuple(Dynamic_Hs)










class Hamiltonian1D:
	# initialize with given symmetries
	def __init__(self,static,dynamic,Length,Nup=None,kblock=None,a=1,zblock=None,pblock=None,pzblock=None,dtype=complex64):
		if dtype not in supported_dtypes:
			raise TypeError("Hamiltonian1D doesn't support type: "+str(dtype))

		B=Basis1D(Length,Nup=Nup,kblock=kblock,a=a,zblock=zblock,pblock=pblock,pzblock=pzblock)

		self.Ns=B.Ns
		self.dtype=dtype
		if self.Ns > 0:
			self.Static_H=StaticH(B,static,dtype)
			self.Dynamic_Hs=DynamicHs(B,dynamic,dtype)




	# initialize with a basis B
	def __init__(self,static,dynamic,B,dtype=complex64):
		if dtype not in supported_dtypes:
			raise TypeError("Hamiltonian1D doesn't support type: "+str(dtype))
			
		self.Ns=B.Ns
		self.dtype=dtype
		if self.Ns > 0:
			self.Static_H=StaticH(B,static,dtype)
			self.Dynamic_Hs=DynamicHs(B,dynamic,dtype)





	def tocsr(self,time=0):
		"""
		args:
			time=0, the time to evalute drive at.

		description:
			this function simply returns a copy of the Hamiltonian as a csr_matrix evaluated at the desired time.
		"""
		if self.Ns <= 0:
			return csr_matrix(asarray([[]]))

		if self.Static_H != None: # if there is a static Hamiltonian...
			H=self.Static_H	
			for ele in self.Dynamic_Hs:
				H += ele[1]*ele[0](time)
		else: # if there isn't...
			for ele in self.Dynamic_Hs:
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
			return matrix([])

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
			return array([])

		V=asarray(V)
		if self.Static_H != None: # if there is a static Hamiltonian...
			Vnew = self.Static_H.dot(V)	
			for ele in self.Dynamic_Hs:
				J=ele[0](time)
				Vnew += J*(ele[1].dot(V))
		else: # if there isn't...
			for ele in self.Dynamic_Hs:
				J=ele[0](time)
				Vnew += J*(ele[1].dot(V))

		return Vnew





	def MatrixElement(self,Vl,Vr,time=0):
		"""
		args:
			Vl, the vector to multiple with on left side
			Vr, the vector to multiple with on the right side
			time=0, the time to evalute drive at.

		description:
			This function takes the matrix element of the Hamiltonian at the specified time
			between Vl and Vr.
		"""
		if self.Ns <=0:
			return array([])

		Vl=asarray(Vl)
		Vr=asarray(Vr)
		HVr=self.dot(Vr,time=time)
		ME=dot(Vl.T.conj(),HVr)
		return ME





	def SparseEV(self,time=0,k=6,sigma=None,which='SA',maxiter=10000):
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
			return array([]), array([[]])

		return eigsh(self.tocsr(time=time),k=k,sigma=sigma,which=which,maxiter=maxiter)





	def DenseEE(self,time=0):
		"""
		args:
			time=0, time to evaluate drive at.

		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values. 
			uses wrapped lapack functions which are contained in module py_lapack
		"""
		
		if self.Ns <= 0:
			return array([])

		return eigh(self.todense(time=time),JOBZ='N')





	def DenseEV(self,time=0):
		"""
		args:
			time=0, time to evaluate drive at.

		description:
			function which diagonalizes hamiltonian using dense methods solves for eigen values 
			and eigen vectors. uses wrapped lapack functions which are contained in module py_lapack
		"""
		if self.Ns <= 0:
			return array([]), array([[]])

		return eigh(self.todense(time=time))





	def evolve(self,v0,t0,time,real_time=True,verbose=False,integrator='dop853',**integrator_params):
		"""
		args:
			v0, intial wavefunction to evolve.
			t0, intial time 
			time, iterable, or time to evolve v0 to
			real_time, evolve real or imaginary time
			verbose, print times out as you evolve
			integrator, the type of integrator to use for complex_ode
			**integrator_params, the parameters used for the particular integrator
			see documentation http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.integrate.ode.html

		description:
			This function uses complex_ode to evolve an input wavefunction.

		"""
		if self.Ns <= 0:
			return array([])

		if real_time:
			solver=complex_ode(lambda t,y:-1j*self.dot(y,time=t))
		else:
			solver=complex_ode(lambda t,y:-self.dot(y,time=t))

		solver.set_initial_value(v0,t=t0)
		solver.set_integrator(integrator, **integrator_params)
		
		if isinstance(time,Iterable):
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
					raise Exception('failed to integrate')
			return sol

		else:
			if time==t0: return v0
			solver.integrate(time)
			if solver.successful():
				return solver.y
			else:
				raise Exception('failed to integrate')





	def Exponential(self,V,a,time=0,n=1,error=10**(-15)):
		"""
		args:
			V, vector to apply the matrix exponential on.
			a, the parameter in the exponential exp(aH)V
			time, time to evaluate drive at.
			n, the number of steps to break the expoential into exp(aH/n)^n V
			error, if the norm the vector of the taylor series is less than this number
			then the taylor series is truncated.

		description:
			this function computes exp(aH)V as a taylor series in aH.

		"""
		if self.Ns <= 0:
			return array([])

		if n <= 0: raise Exception("n must be >= 0")

		for j in xrange(n):
			V1=V
			e=1.0; i=1		
			while e > error:
				V1=(a/(n*i))*self.dot(V1,time=time)
				V+=V1
				if i%2 == 0:
					e=norm(V1)
				i+=1
		return V














