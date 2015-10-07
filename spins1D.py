import sys # needed for sys.stop('error message')
#local modules:

from Basis import *
from py_lapack import eigh # used to diagonalize hermitian and symmetric matricies

#python 2.7 modules
#from memory_profiler import profile # needed for the @profile functions which profile the memory usage of a particular function.
from itertools import repeat

from scipy import linalg as la # imported this for use of eigenvalue functions
from scipy.sparse import coo_matrix	# needed as the initial format that the Hamiltonian matrices are stored as
from scipy.sparse import csr_matrix	# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse.linalg  as sla	# needed for the sparse linear algebra packages

from numpy import pi 
from numpy import asarray, sqrt, array, dot
from numpy.linalg import norm # imported this to calculate the norm of vectors when doing error analysis
from numpy import int32, int64, float32, float64, complex64, complex128




# classes for exceptions:

class StaticHError(Exception):
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message

class DynamicHError(Exception):
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message








def StaticH(B,static,dtype):
	ME_list=[]
	st=xrange(B.Ns)
	for i in xrange(len(static)):
		List=static[i]
		if List[0] == "z":
			for bond in List[1]:
				if bond[0] != 0:
					J=repeat(bond[0], B.Ns)
					i=repeat(bond[1], B.Ns)
					j=repeat(bond[2], B.Ns)
					ME_list.extend(map(lambda J,st,i,j:B.findSz(J,st,i,j),J,st,i,j))
		elif List[0] == 'xy':
			for bond in List[1]:
				if bond[0] != 0:
					J=repeat(bond[0], B.Ns)
					i=repeat(bond[1], B.Ns)
					j=repeat(bond[2], B.Ns)
					ME_list.extend(map(lambda J,st,i,j:B.findSxy(J,st,i,j),J,st,i,j))
		elif List[0] == 'h':
			for H in enumerate(List[1]):
				if H[1][2] != 0:
					i=repeat(H[0], B.Ns)
					h=repeat(H[1][2], B.Ns)
					ME_list.extend(map(lambda h,st,i:B.findhz(h,st,i),h,st,i))
				if B.Mcon == False:
					if H[1][0] != 0 or H[1][1] != 0:
						i=repeat(H[0], B.Ns)
						hx=repeat(H[1][0], B.Ns)
						hy=repeat(H[1][1], B.Ns)
						ME_list.extend(map(lambda hx,hy,st,i:B.findhxy(hx,hy,st,i),hx,hy,st,i))
		elif List[0] == 'const':
			for H in List[1]:
				ME_list.extend([[H,s,s] for s in st])
		else:
			raise StaticHError("StaticH doesn't support symbol: "+str(List[0])) 

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
		if List[0] == "z":
			for bond in List[1]:
				if bond[0] != 0:
					J=repeat(bond[0], B.Ns)
					i=repeat(bond[1], B.Ns)
					j=repeat(bond[2], B.Ns)
					ME_list.extend(map(lambda J,st,i,j:B.findSz(J,st,i,j),J,st,i,j))
		elif List[0] == 'xy':
			for bond in List[1]:
				if bond[0] != 0:
					J=repeat(bond[0], B.Ns)
					i=repeat(bond[1], B.Ns)
					j=repeat(bond[2], B.Ns)
					ME_list.extend(map(lambda J,st,i,j:B.findSxy(J,st,i,j),J,st,i,j))
		elif List[0] == 'h':
			for H in enumerate(List[1]):
				if H[1][2] != 0:
					i=repeat(H[0], B.Ns)
					h=repeat(H[1][2], B.Ns)
					ME_list.extend(map(lambda h,st,i:B.findhz(h,st,i),h,st,i))
				if B.Mcon == False:
					if H[1][0] != 0 or H[1][1] != 0:
						i=repeat(H[0], B.Ns)
						hx=repeat(H[1][0], B.Ns)
						hy=repeat(H[1][1], B.Ns)
						ME_list.extend(map(lambda hx,hy,st,i:B.findhxy(hx,hy,st,i),hx,hy,st,i))
		elif List[0] == 'const':
			for H in enumerate(List[1]):
				ME_list.extend([[H[1],s,s] for s in st])
		else:
			raise DynamicHError("DynamicHs doesn't support symbol: "+str(List[0])) 

	
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
				raise BasisError("Hamiltonian1D: Translation, spin inversion, and parity symmetries are not compatible at this time.")
			else:
				self.B=PeriodicBasisT(Length,Nup=Nup,kblock=kblock,a=a)
				if (dtype != complex128) and (dtype != complex64):
					print "Hamiltonian1D: using momentum states requires complex values: setting dtype to complex64"
					dtype=complex64
		elif (type(zblock) is int) or (type(pblock) is int) or (type(pzblock) is int):
			self.B=OpenBasisPZ(Length,Nup=Nup,zblock=zblock,pblock=pblock,pzblock=pzblock)
		else:
			self.B=Basis(Length,Nup=Nup)

		self.Static=static
		self.Dynamic=dynamic
		
		self.Ns=self.B.Ns
		if self.Ns > 0:
			self.Static_H=StaticH(self.B,static,dtype=dtype)
			self.Dynamic_Hs=DynamicHs(self.B,dynamic,dtype=dtype)

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
	
	def MatrixElement(self,Vl,Vr,time=0):
		if self.Ns <= 0:
			return None
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

		HVr=csr_matrix.dot(H,Vr)
		ME=dot(Vl.T.conj(),HVr)
		return ME


	def dot(self,V,time=0):
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
		return csr_matrix.dot(H,V)


	def SparseEV(self,time=0,n=6,sigma=None,which='SA',tol=0,maxiter=None):
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

		return sla.eigsh(H,k=n,sigma=sigma,which=which,tol=tol)
	


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
		#return la.eigvalsh(H.todense())

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
		#return la.eigh(H.todense())





	def Evolve(self,V,dt,time=0,n=1,error=10**(-15)):
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














