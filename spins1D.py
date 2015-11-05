import sys # needed for sys.stop('error message')
#local modules:
from Basis import *
from py_lapack import eigh # used to diagonalize hermitian and symmetric matricies

#python 2.7 modules
from scipy.linalg import norm
from scipy.sparse import coo_matrix	# needed as the initial format that the Hamiltonian matrices are stored as
from scipy.sparse import csr_matrix	# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse.linalg  as sla	# needed for the sparse linear algebra packages
from scipy.integrate import odeint	# ode solver used in evolve function.

from numpy import pi, asarray, array, int32, int64, float32, float64, complex64, complex128



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
		Dynamic_Hs.append(H)

	return Dynamic_Hs










class Hamiltonian1D:
	def __init__(self,static,dynamic,Length,Nup=None,kblock=None,a=1,zblock=None,pblock=None,pzblock=None,dtype=complex64):
		if dtype not in [float32, float64, complex64, complex128]:
			raise TypeError("Hamiltonian1D doesn't support type: "+str(dtype))

		# testing blocks for basis
		if (type(kblock) is int):
			if (type(zblock) is int) or (type(pblock) is int) or (type(pzblock) is int):
				raise BasisError("Translation, spin inversion, and parity symmetries are not implimented at this time.")
			else:
				self.B=PeriodicBasis1D(Length,Nup=Nup,kblock=kblock,a=a)
				if (dtype != complex128) and (dtype != complex64):
					print "Hamiltonian1D: using momentum states requires complex values: setting dtype to complex64"
					dtype=complex64
		elif (type(zblock) is int) or (type(pblock) is int) or (type(pzblock) is int):
			self.B=OpenBasis1D(Length,Nup=Nup,zblock=zblock,pblock=pblock,pzblock=pzblock)
		else:
			self.B=Basis(Length,Nup=Nup)

		self.Static=static
		self.Dynamic=dynamic
		
		self.Ns=self.B.Ns
		if self.Ns > 0:
			self.Static_H=StaticH(self.B,static,dtype)
			self.Dynamic_Hs=DynamicHs(self.B,dynamic,dtype)

	def return_H(self,time=0):
		if self.Ns**2 > sys.maxsize:
			raise MemoryError
		if self.Ns <= 0:
			return matrix([])
		if self.Static: # if there is a static Hamiltonian...
			H=self.Static_H	
			for i in xrange(len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]
		else: # if there isn't...
			J=self.Dynamic[0][2](time)
			H=J*self.Dynamic_Hs[0]
			for i in xrange(1,len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]

		return H.todense()

	def dot(self,V,time=0):
		if self.Ns <= 0:
			return array([])
		if self.Static: # if there is a static Hamiltonian...
			Vnew = self.Static_H.dot(V)	
			for i in xrange(len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				Vnew += J*self.Dynamic_Hs[i].dot(V)
		else: # if there isn't...
			J=self.Dynamic[0][2](time)
			Vnew=J*self.Dynamic_Hs[i].dot(V)
			for i in xrange(1,len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				Vnew += J*self.Dynamic_Hs[i].dot(V)
		return Vnew
	


	def MatrixElement(self,Vl,Vr,time=0):
		
		HVr=self.dot(Vr,time=time)
		ME=dot(Vl.T.conj(),HVr)
		return ME


	def SparseEV(self,time=0,n=6,sigma=None,which='SA',maxiter=None):
		if self.Ns <= 0:
			return array([]), matrix([])
		if self.Static: # if there is a static Hamiltonian...
			H=self.Static_H	
			for i in xrange(len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]
		else: # if there isn't...
			J=self.Dynamic[0][2](time)
			H=J*self.Dynamic_Hs[0]
			for i in xrange(1,len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]

		return sla.eigsh(H,k=n,sigma=sigma,which=which,maxiter=maxiter)
	


	def DenseEE(self,time=0):
		if self.Ns**2 > sys.maxsize:
			raise MemoryError
		if self.Ns <= 0:
			return array([])
		if self.Static: # if there is a static Hamiltonian...
			H=self.Static_H	
			for i in xrange(len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]
		else: # if there isn't...
			J=self.Dynamic[0][2](time)
			H=J*self.Dynamic_Hs[0]
			for i in xrange(1,len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]

		return eigh(H.todense(),JOBZ='N')

	def DenseEV(self,time=0):
		if self.Ns**2 > sys.maxsize:
			raise MemoryError
		if self.Ns <= 0:
			return array([]), matrix([])
		if self.Static: # if there is a static Hamiltonian...
			H=self.Static_H	
			for i in xrange(len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]
		else: # if there isn't...
			J=self.Dynamic[0][2](time)
			H=J*self.Dynamic_Hs[0]
			for i in xrange(1,len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]

		return eigh(H.todense())



	def evolve(self,v0,time,full_output=0, rtol=None, atol=None, h0=0.0, hmax=0.0, hmin=0.0,
						 ixpr=0, mxstep=0, mxhnil=0, mxordn=12, mxords=5, printmessg=0):
		return odeint(self.dot,v0,time,full_output=full_output, rtol=rtol, atol=atol, h0=h0, hmax=hmax, hmin=hmin,
						 ixpr=ixpr, mxstep=mxstep, mxhnil=mxhnil, mxordn=mxordn, mxords=mxords, printmessg=printmessg)


	def Exponential(self,V,dt,time=0,n=1,error=10**(-15)):
		if self.Ns <= 0:
			return array([])
		if self.Static: # if there is a static Hamiltonian...
			H=self.Static_H	
			for i in xrange(len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]
		else: # if there isn't...
			J=self.Dynamic[0][2](time)
			H=J*self.Dynamic_Hs[0]
			for i in xrange(1,len(self.Dynamic)):
				J=self.Dynamic[i][2](time)
				H=H+J*self.Dynamic_Hs[i]

		if n <= 0: n=1
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		for j in xrange(n):
			V1=V
			e=1.0; i=1		
			while e > error:
				V1=(-1j*dt/(n*i))*csr_matrix.dot(H,V1)
				V+=V1

				if i%2 == 0:
					e=norm(V1)
				i+=1
		return V














