import sys # needed for sys.stop('error message')
#local modules:
from Basis import *
from py_lapack import eigh # used to diagonalize hermitian and symmetric matricies

#python 2.7 modules
from scipy.linalg import norm
from scipy.sparse import coo_matrix	# needed as the initial format that the Hamiltonian matrices are stored as
from scipy.sparse import csr_matrix	# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse.linalg  as sla	# needed for the sparse linear algebra packages
from scipy.integrate import complex_ode	# ode solver used in evolve function.

from numpy import pi, asarray, array, int32, int64, float32, float64, complex64, complex128, dot


supported_dtypes=(float32, float64, complex64, complex128)


def StaticH(B,static,dtype):
	ME_list=[]
	st=xrange(B.Ns)
	for i in xrange(len(static)):
		List=static[i]
		opstr=List[0]
		bonds=List[1]
		for bond in bonds:
			J=bond[0]
			indx=bond[1:]
			ME_list.extend(map(lambda x:B.Op(J,x,opstr,indx),st))

	if static:
		ME_list=asarray(ME_list).T.tolist()
		ME_list[1]=map( lambda a:int(abs(a)), ME_list[1])
		ME_list[2]=map( lambda a:int(abs(a)), ME_list[2])
		H=coo_matrix((ME_list[0],(ME_list[1],ME_list[2])),shape=(B.Ns,B.Ns),dtype=dtype)
		H=H.tocsr()
		H.sum_duplicates()
		H.eliminate_zeros()
		return H
	else:
		return None








def DynamicHs(B,dynamic,dtype):
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
	def __init__(self,static,dynamic,Length,Nup=None,kblock=None,a=1,zblock=None,pblock=None,pzblock=None,dtype=complex64):
		if dtype not in supported_dtypes:
			raise TypeError("Hamiltonian1D doesn't support type: "+str(dtype))

		# testing blocks for basis
		if (type(kblock) is int):
			if (type(zblock) is int) or (type(pblock) is int) or (type(pzblock) is int):
				raise BasisError("Translation, spin inversion, and parity symmetries are not implimented at this time.")
			else:
				B=PeriodicBasis1D(Length,Nup=Nup,kblock=kblock,a=a)
				if (dtype != complex128) and (dtype != complex64):
					print "Hamiltonian1D: using momentum states requires complex values: setting dtype to complex64"
					dtype=complex64
		elif (type(zblock) is int) or (type(pblock) is int) or (type(pzblock) is int):
			B=OpenBasis1D(Length,Nup=Nup,zblock=zblock,pblock=pblock,pzblock=pzblock)
		else:
			B=Basis(Length,Nup=Nup)
		
		self.Ns=B.Ns
		if self.Ns > 0:
			self.Static_H=StaticH(B,static,dtype)
			self.Dynamic_Hs=DynamicHs(B,dynamic,dtype)





	def todense(self,time=0):

		if self.Ns <= 0:
			return matrix([])

		if self.Static_H != None: # if there is a static Hamiltonian...
			H=self.Static_H	
			for ele in self.Dynamic_Hs:
				H += J*ele[1]*ele[0](time)
		else: # if there isn't...
			for ele in self.Dynamic_Hs:
				H += J*ele[1]*ele[0](time)

		return H.todense()





	def dot(self,V,time=0):
		if self.Ns <= 0:
			return array([])

		if self.Static_H != None: # if there is a static Hamiltonian...
			Vnew = self.Static_H.dot(V)	
			for ele in self.Dynamic_Hs:
				J=ele[0](time)
				Vnew += J*(ele[1].dot(Vnew))
		else: # if there isn't...
			for ele in self.Dynamic_Hs:
				J=ele[0](time)
				Vnew += J*(ele[1].dot(Vnew))

		return Vnew
	





	def MatrixElement(self,Vl,Vr,time=0):
		Vl=asarray(Vl)
		Vr=asarray(Vr)
		HVr=self.dot(Vr,time=time)
		ME=dot(Vl.T.conj(),HVr)
		return ME


	def SparseEV(self,time=0,k=6,sigma=None,which='SA',maxiter=None):

		if self.Ns <= 0:
			return array([]), matrix([])

		if self.Static_H != None: # if there is a static Hamiltonian...
			H=self.Static_H	
			for ele in self.Dynamic_Hs:
				H += J*ele[1]*ele[0](time)
		else: # if there isn't...
			for ele in self.Dynamic_Hs:
				H += J*ele[1]*ele[0](time)

		return sla.eigsh(H,k=k,sigma=sigma,which=which,maxiter=maxiter)
	


	def DenseEE(self,time=0):

		if self.Ns <= 0:
			return array([])

		if self.Static_H != None: # if there is a static Hamiltonian...
			H=self.Static_H	
			for ele in self.Dynamic_Hs:
				H += J*ele[1]*ele[0](time)
		else: # if there isn't...
			for ele in self.Dynamic_Hs:
				H += J*ele[1]*ele[0](time)

		return eigh(H.todense(),JOBZ='N')




	def DenseEV(self,time=0):

		if self.Ns <= 0:
			return array([]), matrix([])

		if self.Static_H != None: # if there is a static Hamiltonian...
			H=self.Static_H	
			for ele in self.Dynamic_Hs:
				H += J*ele[1]*ele[0](time)
		else: # if there isn't...
			for ele in self.Dynamic_Hs:
				H += J*ele[1]*ele[0](time)

		return eigh(H.todense())



	def evolve(self,v0,t0,t,real_time=True,**integrator_params):
		if real_time:
			solver=complex_ode(lambda t,y:-1j*self.dot(y,time=t))
		else:
			solver=complex_ode(lambda t,y:-self.dot(y,time=t))

		solver.set_initial_value(v0,t=t0)
		solver.set_integrator('dop853', **integrator_params)
		solver.integrate(t)
		if solver.successful():
			return solver.y
		else:
			raise Exception('failed to integrate')

		


	def Exponential(self,V,dt,time=0,n=1,error=10**(-15)):
		if self.Ns <= 0:
			return array([])

		if n <= 0: raise Exception("n must be >= 0")

		for j in xrange(n):
			V1=V
			e=1.0; i=1		
			while e > error:
				V1=(-1j*dt/(n*i))*self.dot(V1,time=time)
				V+=V1

				if i%2 == 0:
					e=norm(V1)
				i+=1
		return V














