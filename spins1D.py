import sys # needed for sys.stop('error message')
#local modules:
from BitOps import * # loading modules for bit operations.

#python 2.7 modules
import operator as op # needed to calculate n choose r in function ncr(n,r).
#from memory_profiler import profile # needed for the @profile functions which profile the memory usage of a particular function.
from itertools import repeat

from array import array as vec
from py_lapack import eigh
from scipy import linalg as la # imported this for use of eigenvalue functions
from numpy.linalg import norm # imported this to calculate the norm of vectors when doing error analysis
from scipy.sparse import coo_matrix	# needed as the initial format that the Hamiltonian matrices are stored as
from scipy.sparse import csr_matrix	# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse.linalg  as sla	# needed for the sparse linear algebra packages
from numpy import int32, int64, float32, float64, complex64, complex128

import numpy as np # needed for general mathematical operators on vectors/matrices
from numpy import pi 
from numpy import  * # importing functions from numpy so that they can be used in evolve function.




# classes for exceptions:

class BasisError(Exception):
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message


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





def ncr(n, r):
# this function calculates n choose r used to find the total number of basis states when the magnetization is conserved.
    r = min(r, n-r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n-r, -1))
    denom = reduce(op.mul, xrange(1, r+1))
    return numer//denom




# Parent Class Basis: This class is the basic template for all other basis classes. It only has Magnetization symmetry as an option.
# all 'child' classes will inherit its functionality, but that functionality can be overwritten in the child class.
# Basis classes must have the functionality of finding the matrix elements built in. This way, the function for constructing 
# the hamiltonian is universal and the basis object takes care of constructing the correct matrix elements based on its internal symmetry. 
class Basis:
	def __init__(self,L,Nup=None):
		self.L=L
		if type(Nup) is int:
			if Nup <=0 or Nup >= L: sys.exit("Basis1D error: Nup must fall inbetween 0 and L")
			self.Nup=Nup
			self.Mcon=True 
			self.symm=True # Symmetry exsists so one must use the search functionality when calculating matrix elements
			self.Ns=ncr(L,Nup) 
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
			self.symm=False # No symmetries here. at all so each integer corresponds to the number in the hilbert space.
			zbasis=xrange(self.Ns)

		self.basis=zbasis


	def FindZstate(self,s):
		if self.symm:
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

	def findSz(self,J,st,i,j):
		s1=self.basis[st]
		s2=exchangeBits(s1,i,j)
		if s1 == s2:
			return [0.25*J,st,st]
		else:
			return [-0.25*J,st,st]

	def findSxy(self,J,st,i,j):
		s1=self.basis[st]
		s2=exchangeBits(s1,i,j)
		if s1 == s2:
			return [0,st,st]
		else:
			stt=self.FindZstate(s2)
			ME=0.5*J
		return [ME,st,stt]

	def findhz(self,h,st,i):
		s1=self.basis[st]
		if testBit(s1,i) == 1:
			return [-0.5*h,st,st]
		else:
			return [0.5*h,st,st]

	def findhxy(self,hx,hy,st,i):
		if not self.Mcon:
			raise BasisError('transverse field terms present when Magnetization is conserved.')
		s1=self.basis[st]
		s2=flipBit(s1,i)
		if testBit(s2,i) == 1:
			stt=B.FindZstate(s2)
			ME=-0.5*(hx-1j*hy)
		else:
			stt=B.FindZstate(s2)
			ME=-0.5*(hx+1j*hy)
				
		return [ME,st,stt]








# First child class, this is the momentum conserving basis:
# this functions are needed for constructing the momentum states:


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


def CheckStateT(kblock,L,s,T=1):
# this is a function defined in Ander's paper. It is used to check if the integer inputed is the marker state for a state with momentum k.
	t=s
	for i in xrange(1,L+1,T):
		t = shift(t,-T,L)
		if t < s:
			return -1
		elif t==s:
			if kblock % (L/i) != 0: return -1 # need to check the shift condition 
			return i


class BasisT(Basis):
	def __init__(self,L,Nup=None,kblock=None,a=1):
		Basis.__init__(self,L,Nup)
		self.a=a
		zbasis=self.basis
		if type(kblock) is int:
			self.kblock=kblock
			self.k=2*pi*a*kblock/L
			self.Kcon=True
			self.symm=True # even if Mcon=False there is a symmetry therefore we must search through basis list.
			self.R=vec('I')
			self.basis=vec('L')
			for s in zbasis:
				r=CheckStateT(kblock,L,s,T=a)
				if r > 0:
					self.R.append(r)
					self.basis.append(s)
			self.Ns=len(self.basis)
		else: 
			self.Kcon=False # do not change symm to False since there may be Magnetization conservation.
		

	def RefState(self,s):
		t=s; r=s; l=0;
		for i in xrange(1,self.L+1,self.a):
			t=shift(t,-self.a,self.L)
			if t < r:
				r=t; l=i;

		return r,l

	# don't need to override diagonal matrix elements
	# overriding off diagonal matrix element functions for this more specific basis.
	# if there are no conservation laws specified then it calls the parent class function.
	def findSxy(self,J,st,i,j):
		if self.Kcon:
			s1=self.basis[st]
			s2=exchangeBits(s1,i,j)
			if s1 == s2:
				ME=0.0;	stt=st
			else:
				s2,l=self.RefState(s2)
				stt=self.FindZstate(s2)
				if stt >= 0:
					ME=sqrt(float(self.R[st])/self.R[stt])*0.5*J*exp(-1j*self.k*l)
				else:
					ME=0.0;	stt=0
			return [ME,st,stt]
		else:
			return Basis.findSxy(self,J,st,i,j)


	def findhxy(self,hx,hy,st,i):
		if not self.Mcon:
			raise BasisError('transverse field terms present when Magnetization is conserved.')
		if self.Kcon:
			s1=self.basis[st]
			s2=flipBit(s1,i)
			updown=testBit(s2,i)
			s2,l=self.RefState(s2)
			stt=self.FindZstate(s2)
			if stt >= 0:
				if updown == 1:
					ME=-sqrt(float(self.R[st])/self.R[stt])*0.5*(hx-1j*hy)*exp(-1j*self.k*l)
				else:
					ME=-sqrt(float(self.R[st])/self.R[stt])*0.5*(hx+1j*hy)*exp(-1j*self.k*l)
			else: 
				ME=0.0
				stt=0
			return [ME,st,stt]
		else:
			return Basis.findhxy(self,hx,hy,st,i)


		


	





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



class OpenBasisPZ(Basis):
	def __init__(self,L,Nup=None,pblock=None,zblock=None,pzblock=None):
		Basis.__init__(self,L,Nup)
		zbasis=self.basis


		if (type(pblock) is int) and (type(zblock) is int):
			self.Pcon = True
			self.Zcon = True
			self.PZcon = True
			self.p = pblock
			self.z = zblock
			self.pz = pblock*zblock
			if (type(pzblock) is int) and (self.pz != self.p*self.z):
				print "OpenBasisPZ wanring: contradiction between pzblock and pblock*zblock, assuming the block denoted by pblock and zblock" 
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


	def findSxy(self,J,st,i,j):
		if self.Pcon or self.Zcon or self.PZcon:
			s1=self.basis[st]
			s2=exchangeBits(s1,i,j)
			if s1 == s2:
				ME=0; stt=st
			else:
				s2,q,g,qg=self.RefState(s2)
				stt=self.FindZstate(s2)
#				print st,int2bin(s1,B.L),int2bin(exchangeBits(s1,i,j),B.L), stt,int2bin(s2,B.L), q, g, [i,j]
				if stt >= 0:
					if self.Pcon and self.Zcon:
						ME = sqrt( float(self.Npz[stt])/self.Npz[st]   )*0.5*J*self.p**(q)*self.z**(g)
					elif self.Pcon:
							ME = sqrt( float(self.Np[stt])/(self.Np[st]) )*0.5*J*self.p**(q)
					elif self.Zcon:
						ME =  0.5*J*self.z**(g)
					elif self.PZcon:
						ME = sqrt( float(self.Npz[stt])/self.Npz[st]   )*0.5*J*self.pz**(qg)		
				else:
					ME = 0.0
					stt = 0	
			return [ME,st,stt]	
		else:
			return Basis.findSxy(self,J,st,i,j)



	def findhxy(B,hx,hy,st,i):
		s1=B.basis[st]
		s2=flipBit(s1,i)
		if testBit(s2,i) == 1:
			if B.Pcon or B.Zcon or B.PZcon:
				s1=B.basis[st]
				s2=flipBit(s1,i)
				updown=testBit(s2,i)
				s2,q,g=B.RefState(s2)
				stt=B.FindZstate(s2)
				if stt >= 0:
					if B.Pcon and B.Zcon:
						if updown==1:
							ME=-sqrt( float(B.Npz[stt])/B.Npz[st] )*0.5*(hx-1j*hy)*B.p**(q)*B.z**(g)
						else:
							ME=-sqrt( float(B.Npz[stt])/B.Npz[st] )*0.5*(hx+1j*hy)*B.p**(q)*B.z**(g)
					elif B.Pcon:
						if updown==1:
							ME=-sqrt(float(B.Np[stt])/(B.Np[st]))*0.5*(hx-1j*hy)*B.p**(q)
						else:
							ME=-sqrt(float(B.Np[stt])/(B.Np[st]))*0.5*(hx+1j*hy)*B.p**(q)
					elif B.Zcon:
						if updown==1:
							ME=-0.5*(hx-1j*hy)*B.z**(g)
						else:
							ME=-0.5*(hx+1j*hy)*B.z**(g)
					elif B.PZcon:
						if updown==1:
					 		ME=-sqrt( float(B.Npz[stt])/B.Npz[st] )*0.5*(hx-1j*hy)*B.pz**(qg)
						else:
							ME=-sqrt( float(B.Npz[stt])/B.Npz[st] )*0.5*(hx+1j*hy)*B.pz**(qg)
				else:
					ME = 0.0
					stt=0
				return [ME,st,stt]
			else:
				return Basis.findhxy(self,hx,hy,st,i)























"""
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

"""



def StaticH(B,static,dtype=np.complex128):
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
			for H in enumerate(List[1]):
				ME_list.extend([[H[1],st,st] for s in st])
		else:
			raise StaticHError("StaticH doesn't support symbol: "+List[0]) 

	if static:
		ME_list=asarray(ME_list).T.tolist()
		ME_list[1]=map( lambda a:int(abs(a)), ME_list[1])
		ME_list[2]=map( lambda a:int(abs(a)), ME_list[2])
		H=coo_matrix((ME_list[0],(ME_list[1],ME_list[2])),shape=(B.Ns,B.Ns),dtype=dtype)
		H=H.tocsr()
		H.sum_duplicates()
		H.eliminate_zeros()
	return H








def DynamicHs(B,dynamic,dtype=np.complex128):
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
			raise DynamicHError("DynamicHs doesn't support symbol: "+List[0]) 

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
				self.B=BasisT(Length,Nup=Nup,kblock=kblock,a=a)
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
		self.Static_H=StaticH(self.B,static,dtype=dtype)
		self.Dynamic_Hs=DynamicHs(self.B,dynamic,dtype=dtype)

	def return_H(self,time=0):
		if self.Ns**2 > sys.maxsize:
			raise MemoryError
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
		ME=np.dot(Vl.T.conj(),HVr)
		return ME[0,0]


	def dot(self,V,time=0):
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

		HV=csr_matrix.dot(H,V)
		return HV


	def SparseEV(self,time=0,n=None,sigma=None,which='SA'):
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

		return sla.eigsh(H,k=n,sigma=sigma,which=which)
	


	def DenseEE(self,time=0):
		if self.Ns**2 > sys.maxsize:
			raise MemoryError
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
		denseH=H.todense()

		return eigh(H.todense(),JOBZ='N')

	def DenseEV(self,time=0):
		if self.Ns**2 > sys.maxsize:
			raise MemoryError
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

		denseH=H.todense()
		return eigh(denseH)





	def Evolve(self,V,dt,time=0,n=1,error=10**(-15)):
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














