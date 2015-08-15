from BitOps import * # loading modules for bit operations.
from Z_Basis import Basis, BasisError

from array import array as vec
from numpy import pi, sqrt

# First child class, this is the momentum conserving basis:
# this functions are needed for constructing the momentum states:
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

# Basis class:
class PeriodicBasisT(Basis):
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


		


