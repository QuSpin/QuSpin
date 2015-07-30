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



"""
Note: the 1D basis can be used for any dimension if the momentum states are not used.
Later we shall add a more general version of the basis class with functionality for arbitrary lattices.
"""

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




class Basis1D:
	def __init__(self,L,Nup=None,kblock=None,a=1):
		self.L=L
		if type(Nup) is int:
			if Nup <=0 or Nup >= L: sys.exit("Basis1D error: Nup must fall inbetween 0 and L")
			self.Nup=Nup
			self.Mcon=True
			self.Ns=ncr(L,Nup)
			self.a=a
			zbasis=vec('L')
			s=sum([2**i for i in xrange(0,Nup)])
			zbasis.append(s)
			for i in xrange(self.Ns-1):
				t = (s | (s - 1)) + 1
				s = t | ((((t & -t) / (s & -s)) >> 1) - 1) 
				zbasis.append(s)
		else:
			self.Ns=2**L
			self.a=a
			self.Mcon=False
			zbasis=xrange(self.Ns)

		if type(kblock) is int:
			self.kblock=kblock
			self.k=2*pi*a*kblock/L
			self.Kcon=True
			self.R=vec('I')
			self.basis=vec('L')
			for s in zbasis:
				r=CheckState(kblock,L,s,T=a)
				if r > 0:
					self.R.append(r)
					self.basis.append(s)
			self.Ns=len(self.basis)
		else: 
			self.Kcon=False
			self.basis=zbasis
			


	def FindZstate(self,s):
		if self.Kcon or self.Mcon:
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
		t=s; r=s; l=0;
		for i in xrange(1,self.L+1,self.a):
			t=shift(t,-self.a,self.L)
			if t < r:
				r=t; l=i;

		return r,l






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
		if B.Kcon:
			s2,l=B.RefState(s2)
			stt=B.FindZstate(s2)
			if stt >= 0:
				ME=sqrt(float(B.R[st])/B.R[stt])*0.5*J*exp(-1j*B.k*l)
			else:
				ME=0.0
				stt=0
		else:
			stt=B.FindZstate(s2)
			ME=0.5*J

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
		if B.Kcon:
			s2,l=B.RefState(s2)
			stt=B.FindZstate(s2)
			if stt >= 0:
				ME=-sqrt(float(B.R[st])/B.R[stt])*0.5*(hx-1j*hy)*exp(-1j*B.k*l)
			else: 
				ME=0.0
				stt=0
		else:
			stt=B.FindZstate(s2)
			ME=-0.5*(hx-1j*hy)
	else:
		if B.Kcon:
			s2,l=B.RefState(s2)
			stt=B.FindZstate(s2)
			if stt >= 0:
				ME=-sqrt(float(B.R[st])/B.R[stt])*0.5*(hx+1j*hy)*exp(-1j*B.k*l)
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
	def __init__(self,static,dynamic,Length,Nup=None,kblock=None,a=1,dtype=np.complex128):
		if type(kblock) is int: 
			if dtype != np.complex128 and dtype != np.complex64:
				print "Hamiltonian1D: using momentum states requires complex values: setting dtype to complex64"
				dtype=np.complex64
		self.Static=static
		self.Dynamic=dynamic
		self.B=Basis1D(Length,Nup=Nup,kblock=kblock,a=a)
		self.Static_H=StaticH1D(self.B,static,dtype=dtype)
		self.Dynamic_Hs=DynamicHs1D(self.B,dynamic,dtype=dtype)

	@profile(precision=6)
	def return_H(self,time=0):
		if self.B.Ns**2 > sys.maxsize:
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
	

	@profile(precision=6)
	def DenseEE(self,time=0):
		if self.B.Ns**2 > sys.maxsize:
			sys.exit('Hamiltonian1D: dense matrix is too large to create. Full diagonalization is not possible')
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		denseH=H.todense()	
		return la.eigvalsh(denseH,check_finite=False,overwrite_a=True,overwrite_b=True)

	@profile(precision=6)
	def DenseEV(self,time=0):
		if self.B.Ns**2 > sys.maxsize:
			sys.exit('Hamiltonian1D: dense matrix is too large to create. Full diagonalization is not possible')
		H=self.Static_H
		for i in xrange(len(self.Dynamic)):
			J=self.Dynamic[i][2](time)
			H=H+J*self.Dynamic_Hs[i]
		denseH=H.todense()		
		return la.eigh(denseH,check_finite=False,overwrite_a=True,overwrite_b=True)





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














