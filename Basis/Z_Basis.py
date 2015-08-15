from BitOps import * # loading modules for bit operations.
import operator as op # needed to calculate n choose r in function ncr(n,r).
from array import array as vec



class BasisError(Exception):
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




