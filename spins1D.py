import sys # needed for sys.stop('error message')
#local modules:
from BitOps import * # loading modules for bit operations.

#python 2.7 modules
import operator as op # needed to calculate n choose r in function ncr(n,r).
from memory_profiler import profile # needed for the @profile functions which profile the memory usage of a particular function.
from itertools import repeat

from array import array as vec

from scipy import linalg as la # imported this for use of eigenvalue functions
from numpy.linalg import norm # imported this to calculate the norm of vectors when doing error analysis
from scipy.sparse import coo_matrix	# needed as the initial format that the Hamiltonian matrices are stored as
from scipy.sparse import csr_matrix	# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse.linalg  as sla	# needed for the sparse linear algebra packages

import numpy as np # needed for general mathematical operators on vectors/matrices
from numpy import pi 
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




def CheckStatePZ(z,p,s,L,rpz=-1):

	t=s
	t=flip_all(t,L)
	print int2bin(t,L), int2bin(s,L)
	if t > s:
		rpz=2
	else:
		rpz=-1;

	if rpz != -1:
		t=s
		t=fliplr(t,L)
		if t > s:
			rpz=2
		elif t==s:
			if p != -1:
				rpz=8
				return t, rpz 
			else:
				rpz=-1
		else:
			rpz=-1;

	if rpz != -1:
		t=s
		t=fliplr(t,L)
		t=flip_all(t,L)
		if t==s:
			if z*p != -1:
				rpz=8
			else:
				rpz=-1
		elif t > s:
			rpz=4
		else:
			rpz=-1


	return t,rpz
	

def CheckStatePandZ(pz,s,L,rpz=-1):

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
		

'''

def CheckStateP(p,s,L,rp=-1):
	t=s
	t=fliplr(t,L)
	if t == s:
		if p != -1:
			rp=4;
		else:
			rp=-1;
	elif t > s and rp == -1: 
		rp=2;

	return t,rp;


def CheckStateZ(z,s,L,rz=-1):
	t=s
	t=flip_all(t,L)
	if t > s and rz == -1:
		rz=2;

	return t,rz;
'''


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
		   self.Npz = []
		   self.basis = []
		   for s in zbasis:
#					print int2bin(s,self.L)
#		   	   rp = CheckStateP(pblock,self.L,s)
#		   	   rz = CheckStateZ(zblock,self.L,s)
					t,rpz = CheckStatePZ(pblock,zblock,s,self.L)
#					print rpz
					if rpz > 0:
						self.basis.append(s)
						self.Npz.append(rpz)
#					if rpz == 1:
#						self.Npz.append(8)
#					else:
#						self.Npz.append(4)
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
				rp=CheckStateP(pblock,self.L,s,T=a)
				#print rp
				if rp > 0:
					self.basis.append(s)
					if rp == 1:
						self.Np.append(4)
					else:
						self.Np.append(2)
					#print s
				self.Ns=len(self.basis)
		elif type(zblock) is int:
			self.Pcon = False
			self.Zcon = True
			self.PZcon = False
			self.p = pblock
			self.z = zblock
			#self.Rz = []
			self.basis = []
			#print zbasis
			for s in zbasis:
				rz=CheckStateZ(zblock,self.L,s,T=a)
				if rz > 0:
					#self.Rz.append(rz)
					self.basis.append(s)
					#print s
				self.Ns=len(self.basis)
		elif type(pzblock) is int:
			self.PZcon = True
			self.Zcon = False
			self.Pcon = False
			self.pz = pzblock
			self.Npz = []
			self.basis = []
			#print zbasis
			for s in zbasis:
				t, rpz = CheckStatePandZ(pzblock,s,self.L)
				#print [rpz],s,t
				if rpz > 0:
					#self.Rz.append(rz)
					self.basis.append(s)
					self.Npz.append(rpz)
					#print s
				self.Ns=len(self.basis)	
		else: 
			self.Pcon=False
			self.Zcon=False
			self.PZcon=False
			self.basis=zbasis 	
		#print self.Rz	

		#for s in zbasis:
		#		print s, int2bin(s,self.L)




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





def StaticH1D(B,static,dtype=np.complex128):
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
					ME_list.extend(map(lambda J,st,i,j:findSz(B,J,st,i,j),J,st,i,j))
		elif List[0] == 'xy':
			for bond in List[1]:
				if bond[0] != 0:
					J=repeat(bond[0], B.Ns)
					i=repeat(bond[1], B.Ns)
					j=repeat(bond[2], B.Ns)
					ME_list.extend(map(lambda J,st,i,j:findSxy(B,J,st,i,j),J,st,i,j))
		elif List[0] == 'h':
			for H in enumerate(List[1]):
				if H[1][2] != 0:
					i=repeat(H[0], B.Ns)
					h=repeat(H[1][2], B.Ns)
					ME_list.extend(map(lambda h,st,i:findhz(B,h,st,i),h,st,i))
				if B.Mcon == False:
					if H[1][0] != 0 or H[1][1] != 0:
						i=repeat(H[0], B.Ns)
						hx=repeat(H[1][0], B.Ns)
						hy=repeat(H[1][1], B.Ns)
						ME_list.extend(map(lambda hx,hy,st,i:findhxy(B,hx,hy,st,i),hx,hy,st,i))
				elif H[1][0] != 0 or H[1][1] != 0 :
					sys.exit("StaticH1D warning: attemping to put non-magnetization conserving operators when this is an assumed symmetry.")
		elif List[0] == 'const':
			for H in enumerate(List[1]):
				ME_list.extend([[H[1],st,st] for s in st])
		else:
			sys.exit('StaticH: operator symbol not recognized')

	ME_list=asarray(ME_list).T.tolist()
	ME_list[1]=map( lambda a:int(abs(a)), ME_list[1])
	ME_list[2]=map( lambda a:int(abs(a)), ME_list[2])
	H=coo_matrix((ME_list[0],(ME_list[1],ME_list[2])),shape=(B.Ns,B.Ns),dtype=dtype)
	H=H.tocsr()
	H.sum_duplicates()
	H.eliminate_zeros()
	return H







def DynamicHs1D(B,dynamic,dtype=np.complex128):
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
					ME_list.extend(map(lambda J,st,i,j:findSz(B,J,st,i,j),J,st,i,j))
		elif List[0] == 'xy':
			for bond in List[1]:
				if bond[0] != 0:
					J=repeat(bond[0], B.Ns)
					i=repeat(bond[1], B.Ns)
					j=repeat(bond[2], B.Ns)
					ME_list.extend(map(lambda J,st,i,j:findSz(B,J,st,i,j),J,st,i,j))
		elif List[0] == 'h':
			for H in enumerate(List[1]):
				if H[1][2] != 0:
					i=repeat(H[0], B.Ns)
					h=repeat(H[1][2], B.Ns)
					ME_list.extend(map(lambda h,st,i:findhz(B,h,st,i),h,st,i))
				if B.Mcon == False:
					if H[1][0] != 0 or H[1][1] != 0:
						i=repeat(H[0], B.Ns)
						hx=repeat(H[1][0], B.Ns)
						hy=repeat(H[1][1], B.Ns)
						ME_list.extend(map(lambda hx,hy,st,i:findhxy(B,hx,hy,st,i),hx,hy,st,i))
				elif H[1][0] != 0 or H[1][1] != 0 :
					sys.exit("StaticH1D warning: attemping to put non-magnetization conserving operators when this is an assumed symmetry.")
		elif List[0] == 'const':
			for H in enumerate(List[1]):
				ME_list.extend([[H[1],st,st] for s in st])
		else:
			sys.exit('DynamicH: operator symbol not recognized')

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
#	@profile(precision=6)
	def __init__(self,static,dynamic,Length,Nup=None,pblock=None,zblock=None,pzblock=None,a=1,dtype=np.complex128):
		if type(pblock) is int: 
			if dtype != np.complex128 and dtype != np.complex64:
				print "Hamiltonian1D: using momentum states requires complex values: setting dtype to complex64"
				dtype=np.complex64
		self.Static=static
		self.Dynamic=dynamic
		self.B=Basis1D(Length,Nup=Nup,pblock=pblock,zblock=zblock,pzblock=pzblock,a=a)
		self.Ns=self.B.Ns
		self.Static_H=StaticH1D(self.B,static,dtype=dtype)
		self.Dynamic_Hs=DynamicHs1D(self.B,dynamic,dtype=dtype)


	def return_H(self,time=0):
		if self.Ns**2 > sys.maxsize:
			sys.exit('Hamiltonian1D: dense matrix is too large to create')
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		return H.todense()
	
	def MatrixElement(self,Vl,Vr,time=0):
		t=time
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		HVr=csr_matrix.dot(H,Vr)
		ME=np.dot(Vl.T.conj(),HVr)
		return ME[0,0]


	def dot(self,V,time=0):
		t=time
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		HV=csr_matrix.dot(H,V)
		return HV


	def SparseEV(self,time=0,n=None,sigma=None,which='SA'):
		t=time
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		return sla.eigsh(H,k=n,sigma=sigma,which=which)
	


	def DenseEE(self,time=0):
		if self.Ns**2 > sys.maxsize:
			sys.exit('Hamiltonian1D: dense matrix is too large to create. Full diagonalization is not possible')
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]	
		return la.eigvalsh(H.todense(),overwrite_a=False,overwrite_b=False)

#	@profile(precision=3)
	def DenseEV(self,time=0):
		if self.Ns**2 > sys.maxsize:
			sys.exit('Hamiltonian1D: dense matrix is too large to create. Full diagonalization is not possible')
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]	
		return la.eigh(H.todense(),overwrite_a=True,overwrite_b=True)





	def Evolve(self,V,dt,time=0,n=1,error=10**(-15)):
		t=time
		H=self.Static_H
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














