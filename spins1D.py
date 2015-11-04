import sys # needed for sys.stop('error message')
#local modules:

from Basis import *
from py_lapack import eigh # used to diagonalize hermitian and symmetric matricies

#python 2.7 modules
#from memory_profiler import profile # needed for the @profile functions which profile the memory usage of a particular function.
from itertools import repeat

<<<<<<< HEAD
=======
from array import array as vec

import py_lapack
>>>>>>> OBC
from scipy import linalg as la # imported this for use of eigenvalue functions
from scipy.sparse import coo_matrix	# needed as the initial format that the Hamiltonian matrices are stored as
from scipy.sparse import csr_matrix	# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse.linalg  as sla	# needed for the sparse linear algebra packages

from numpy import pi 
<<<<<<< HEAD
from numpy import asarray, sqrt, array, dot
from numpy.linalg import norm # imported this to calculate the norm of vectors when doing error analysis
from numpy import int32, int64, float32, float64, complex64, complex128
=======
from numpy import  * # importing functions from numpy so that they can be used in evolve function.

from inspect import currentframe, getframeinfo #allows to grap current line number


"""
Note: the 1D basis can be used for any dimension if the momentum states are not used.
Later we shall add a more general version of the basis class with functionality for arbitrary lattices.
"""


# check binary operations
def int2bin(n,L):
	""" Convert an integer n to a binary vector 
	padded with zeros up to the appropriate length L """

	return (((fliplr(n,L) & (1 << np.arange(L)))) > 0).astype(int)

def bin2int(v):
	""" Convert a binary vector v to an integer """

	return np.sum([v[i]*2**i for i in range(len(v))])

def ncr(n, r):
# this function calculates n choose r used to find the total number of basis states when the magnetization is conserved.
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom



def shift(int_type,shift,period):
# this functino is used to shift the bits of an integer by 'shift' bits.
# it is used when constructing the momentum states
	Imax=2**period-1
	if int_type==0 or int_type==Imax:
		return int_type
	else:
		if shift < 0:
			shift=-shift
			return ((int_type & (2**period-1)) >> shift%period) | (int_type << (period-(shift%period)) & (2**period-1))
		else:
			return (int_type << shift%period) & (2**period-1) | ((int_type & (2**period-1)) >> (period-(shift%period)))

def fliplr(int_type,length):
# this function flips the bits of an integer around the centre, e.g. 1010 -> 0101
# (generator of) parity symmetry
    return sum(1<<(length-1-i) for i in range(length) if int_type>>i&1)

def flip_all(int_type,length):
# this function flips all bits
# (generator of) inversion symmetry
    lower = 0;
    upper = length;
    return int_type^((1<<upper)-1)&~((1<<lower)-1)

def CheckState(kblock,L,s,T=1):
# this is a function defined in Ander's paper. It is used to check if the integer inputed is the marker state for a state with momentum k.
	t=s
	for i in xrange(1,L+1,T):
		t = shift(t,-T,L)
		if t < s:
			return -1
		elif t==s:
			if kblock % (L/i) != 0: return -1 # need to check the shift condition 
			return i
 
		
"""
def CheckStateP(pblock,L,s,T=1):
# parity Check_State
    t = s
    i = 1
    t = fliplr(t,L)
    if t < s:
    	return -1
    elif t == s:
    	if pblock == -1: return -1
    	return i 
    return 2	

def CheckStateZ(zblock,L,s,T=1):
# inversion Check_State
    t = s
    i = 2 #alternative one can put i=0 but then one needs to modify the basis class to accept it
    t = flip_all(t,L)
    if t < s:
    	return -1
    #no compatibility condition for inversion symmetry  
    return i			


def CheckStatePandZ(pz,s,L,rpz=2):

	t=s
	#print pz, s, L
	#print int2bin(t,L), int2bin(s,L)
	t=fliplr(t,L)
	t=flip_all(t,L)
	
	if t==s:
		if pz != -1:
			rpz=4
		else:
			rpz=-1
	elif t > s:
		rpz=2
	else:
		rpz=-1

	return t,rpz

"""


def CheckStatePZ(pz,s,L,rpz=2):
	t=s
	t=fliplr(t,L)
	t=flip_all(t,L)
	if t==s:
		if pz != -1:
			rpz*=2
		else:
			rpz=-1*abs(rpz)
	elif t > s:
		rpz*=1
	else:
		rpz=-1*abs(rpz)

	return rpz
		

def CheckStateP(p,s,L,rp=2):
	t=s
	t=fliplr(t,L)
	if t == s:
		if p != -1:
			rp*=2
		else:
			rp=-1*abs(rp)
	elif t > s: 
		rp*=1
	else:
		rp=-1*abs(rp)

	return rp;


def CheckStateZ(z,s,L,rz=2):
	t=s
	t=flip_all(t,L)
	if t > s:
		rz*=1;
	else:
		rz=-1*abs(rz)
	return rz;



class Basis1D:
	def __init__(self,L,Nup=None,pblock=None,zblock=None,pzblock=None,a=1):
		self.L=L
		if type(Nup) is int:
			if Nup <=0 or Nup >= L: sys.exit("Basis1D error: Nup must fall inbetween 0 and L")
			self.Nup=Nup
			self.Mcon=True
			self.Ns=ncr(L,Nup)
			self.a=a
			zbasis=vec('L')
			s=sum([2**i for i in xrange(0,Nup)])
#			print int2bin(s,L)
			zbasis.append(s)
			for i in xrange(self.Ns-1):
				t = (s | (s - 1)) + 1
				s = t | ((((t & -t) / (s & -s)) >> 1) - 1) 
#				print int2bin(s,L)
				zbasis.append(s)
		else:
			self.Ns=2**L
			self.a=a
			self.Mcon=False
			zbasis=xrange(self.Ns)

		#print pblock, zblock	


		if type(pblock) is int and type(zblock) is int:
			self.Pcon = True
			self.Zcon = True
			self.PZcon = False
			self.p = pblock
			self.z = zblock
			self.pz = pblock*zblock
			self.Npz = []
			self.basis = []
			for s in zbasis:
				rpz = CheckStateZ(zblock,s,self.L)
				rpz = CheckStateP(pblock,s,self.L,rp=rpz)
				rpz = CheckStatePZ(pblock*zblock,s,self.L,rpz=rpz)
#				print rpz, int2bin(s,self.L)
				if rpz > 0:
					self.basis.append(s)
					self.Npz.append(rpz)
			self.Ns=len(self.basis)
		elif type(pblock) is int:
			self.Pcon = True
			self.Zcon = False
			self.PZcon = False
			self.p = pblock
			self.z = zblock
			self.Np = []
			self.basis = []
			for s in zbasis:
				rp=CheckStateP(pblock,s,self.L)
#				print rp, int2bin(s,self.L)
				if rp > 0:
					self.basis.append(s)
					self.Np.append(rp)
			self.Ns=len(self.basis)
		elif type(zblock) is int:
			self.Pcon = False
			self.Zcon = True
			self.PZcon = False
			self.z = zblock
			self.basis = []
			for s in zbasis:
				rz=CheckStateZ(zblock,s,self.L)
#				print rz, int2bin(s,self.L)
				if rz > 0:
					self.basis.append(s)
			self.Ns=len(self.basis)
		elif type(pzblock) is int:
			self.PZcon = True
			self.Zcon = False
			self.Pcon = False
			self.pz = pzblock
			self.Npz = []
			self.basis = []
			for s in zbasis:
				rpz = CheckStatePZ(pzblock,s,self.L)
#				print rpz, int2bin(s,self.L)
				if rpz > 0:
					self.basis.append(s)
					self.Npz.append(rpz)
			self.Ns=len(self.basis)	
		else: 
			self.Pcon=False
			self.Zcon=False
			self.PZcon=False
			self.basis=zbasis 	




	def FindZstate(self,s):
		if self.Mcon:
			#cf = currentframe()
			#print "There was Kcon here; Line:", cf.f_lineno 
			bmin=0;bmax=self.Ns-1
			while True:
				b=(bmin+bmax)/2
				if s < self.basis[b]:
					bmax=b-1
				elif s > self.basis[b]:
					bmin=b+1
				else:
					return b
				if bmin > bmax:
					return -1
		else: return s

	def RefState(self,s):
		t=s; r=s; g=0; q=0; qg=0;
		if self.Pcon and self.Zcon:
			t = flip_all(t,self.L)
			if t < r:
				r=t; g=1;q=0;
			t=s
			t = fliplr(t,self.L)
			if t < r:
				r=t; q=1; g=0;
			t=flip_all(t,self.L)
			if t < r:
				r=t; q=1; g=1;
		elif self.Pcon:
			t = fliplr(t,self.L)
			if t < s:
				r=t; q=1;
		elif self.Zcon:
			t = flip_all(t,self.L)
			if t < s:
				r=t; g=1;
		elif self.PZcon:
			t = fliplr(t,self.L)
			t = flip_all(t,self.L)
			if t < s:
				r=t; qg=1;		

		return r,q,g,qg


def findSz(B,J,st,i,j):
	s1=B.basis[st]
	s2=exchangeBits(s1,i,j)
	if s1 == s2:
		return [0.25*J,st,st]
	else:
		return [-0.25*J,st,st]


def findSxy(B,J,st,i,j):
	s1=B.basis[st]
	s2=exchangeBits(s1,i,j)
	if s1 == s2:
		return [0,st,st]
	else:
		if B.Pcon or B.Zcon or B.PZcon:
			s2,q,g,qg=B.RefState(s2)
			stt=B.FindZstate(s2)
#			print st,int2bin(s1,B.L),int2bin(exchangeBits(s1,i,j),B.L), stt,int2bin(s2,B.L), q, g, [i,j]
			if stt >= 0:
				#print s1,s2
				if B.Pcon and B.Zcon:
					ME = sqrt( float(B.Npz[stt])/B.Npz[st]   )*0.5*J*B.p**(q)*B.z**(g)
				elif B.Pcon:
						ME = sqrt( float(B.Np[stt])/(B.Np[st]) )*0.5*J*B.p**(q)
				elif B.Zcon:
					ME =  0.5*J*B.z**(g)
				elif B.PZcon:
					ME = sqrt( float(B.Npz[stt])/B.Npz[st]   )*0.5*J*B.pz**(qg)		
			else:
				ME = 0.0
				stt = 0		
		else:
			stt=B.FindZstate(s2)
			ME=0.5*J
		#print [ME,st,stt]
		return [ME,st,stt]




def findhz(B,h,st,i):
	s1=B.basis[st]
	if testBit(s1,i) == 1:
		return [-0.5*h,st,st]
	else:
		return [0.5*h,st,st]




def findhxy(B,hx,hy,st,i):
	s1=B.basis[st]
	s2=flipBit(s1,i)
	if testBit(s2,i) == 1:
		if B.Pcon or B.Zcon:
			s2,q,g=B.RefState(s2)
			stt=B.FindZstate(s2)
			if stt >= 0:
				if B.Pcon and B.Zcon:
					ME=sqrt( float(B.Npz[stt])/B.Npz[st] )*0.5*(hx-1j*hy)*B.p**(q)*B.z**(g)
				elif B.Pcon:
					ME=sqrt(float(B.Np[stt])/(B.Np[st]))*0.5*(hx-1j*hy)*B.p**(q)
				elif B.Zcon:
					ME=0.5*(hx-1j*hy)*B.z**(g)
				elif B.PZcon:
				 	ME=sqrt( float(B.Npz[stt])/B.Npz[st] )*0.5*(hx-1j*hy)*B.pz**(qg)
			else:
				ME = 0.0
				stt=0
		else:
			stt=B.FindZstate(s2)
			ME=-0.5*(hx-1j*hy)
	else:
		if B.Pcon or B.Zcon:
			s2,q,g=B.RefState(s2)
			stt=B.FindZstate(s2)
			if stt >= 0:
				if B.Pcon and B.Zcon:
					ME=sqrt( float(B.Npz[stt])/B.Npz[st] )*0.5*(hx+1j*hy)*B.p**(q)*B.z**(g)
				elif B.Pcon:
					ME=sqrt( float(B.Np[stt])/(B.Np[st]))*0.5*(hx+1j*hy)*B.p**(q)
				elif B.Zcon:
					ME=0.5*(hx+1j*hy)*B.z**(g)
				elif B.PZcon:
				 	ME=sqrt( float(B.Npz[stt])/B.Npz[st] )*0.5*(hx+1j*hy)*B.pz**(qg)
			else: 
				ME=0.0
				stt=0
		else:
			stt=B.FindZstate(s2)
			ME=-0.5*(hx+1j*hy)
		
	return [ME,st,stt]

>>>>>>> OBC




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
<<<<<<< HEAD
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
=======
#	@profile(precision=6)
	def __init__(self,static,dynamic,Length,Nup=None,pblock=None,zblock=None,pzblock=None,a=1,dtype=np.complex128):
		self.Static=static
		self.Dynamic=dynamic
		self.B=Basis1D(Length,Nup=Nup,pblock=pblock,zblock=zblock,pzblock=pzblock,a=a)
		self.Ns=self.B.Ns
		if self.Ns > 0:
			self.Static_H=StaticH1D(self.B,static,dtype=dtype)
			self.Dynamic_Hs=DynamicHs1D(self.B,dynamic,dtype=dtype)


	def return_H(self,time=0):
		if self.Ns**2 > sys.maxsize:
			sys.exit('Hamiltonian1D: dense matrix is too large to create')
		if self.Ns <= 0:
			return matrix([])
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		return H.todense()
	
	def MatrixElement(self,Vl,Vr,time=0):
		if self.Ns <= 0:
			return None
		t=time
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		HVr=csr_matrix.dot(H,Vr)
		ME=np.dot(Vl.T.conj(),HVr)
		return ME[0,0]
>>>>>>> OBC

		return H.todense()

	def dot(self,V,time=0):
		if self.Ns <= 0:
			return array([])
<<<<<<< HEAD
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
=======
		t=time
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		HV=csr_matrix.dot(H,V)
		return HV


	def SparseEV(self,time=0,n=None,sigma=None,which='SA'):
		if self.Ns <= 0:
			return array([]), matrix([])
		t=time
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		return sla.eigsh(H,k=n,sigma=sigma,which=which)
>>>>>>> OBC
	


	def DenseEE(self,time=0):
		if self.Ns**2 > sys.maxsize:
<<<<<<< HEAD
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
=======
			sys.exit('Hamiltonian1D: dense matrix is too large to create. Full diagonalization is not possible')
		if self.Ns <= 0:
			return array([])
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		Hdense=H.todense()
		return py_lapack.eigh(Hdense,JOBZ='N')

#	@profile(precision=3)
	def DenseEV(self,time=0):
		if self.Ns**2 > sys.maxsize:
			sys.exit('Hamiltonian1D: dense matrix is too large to create. Full diagonalization is not possible')
		if self.Ns <= 0:
			return array([]), matrix([])
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]	
		Hdense=H.todense()
		return py_lapack.eigh(Hdense)
>>>>>>> OBC





	def Evolve(self,V,dt,time=0,n=1,error=10**(-15)):
		if self.Ns <= 0:
			return array([])
<<<<<<< HEAD
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

=======
		t=time
		H=self.Static_H
>>>>>>> OBC
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














