# python 2.7 modules
import operator as op # needed to calculate n choose r in function ncr(n,r).
# array is a very nice data structure which stores values in a c or fortran like array, saving memory, but has all features of a list.
# it is not good for array type operations like multiplication, for those use numpy arrays.
from array import array as vec
from multiprocessing import Manager
from functools import partial

# local modules
from SpinOps import SpinOp # needed to act with opstr
from BitOps import * # loading modules for bit operations.

# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)





class BasisError(Exception):
	# this class defines an exception which can be raised whenever there is some sort of error which we can
	# see will obviously break the code. 
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
		# This is the constructor of the class Basis:
		#		L: length of chain
		#		Nup: number of up spins if restricting magnetization sector. 
		self.L=L
		if type(Nup) is int:
			if Nup < 0 or Nup > L: raise BasisError("0 <= Nup <= "+str(L))
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




	def Op(self,J,opstr,indx,st):
		s1=self.basis[st]
		ME,s2=SpinOp(s1,opstr,indx)
		stt=self.FindZstate(s2)
		return [J*ME,st,stt]












