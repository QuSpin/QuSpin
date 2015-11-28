# python 2.7 modules
import operator as op # needed to calculate n choose r in function ncr(n,r).
# array is a very nice data structure which stores values in a c or fortran like array, saving memory, but has all features of a list.
# it is not good for array type operations like multiplication, for those use numpy arrays.
from array import array as vec
from multiprocessing import Manager
from functools import partial
from bisect import bisect_left

# local modules
from SpinOps import SpinOp # needed to act with opstr
from BitOps import * # loading modules for bit operations.
from Basis_fortran import *

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
		if L>30: raise BasisError("L must be less than 31") 
		self.L=L
		if type(Nup) is int:
			if Nup < 0 or Nup > L: raise BasisError("0 <= Nup <= "+str(L))
			self.Nup=Nup
			self.conserved="M"
			self.Ns=ncr(L,Nup) 
			s0=sum([2**i for i in xrange(0,Nup)])
			mbasis=make_m_basis(s0,self.Ns)
			
#			mbasis=vec("l")			
#			s=sum([2**i for i in xrange(0,Nup)])
#			mbasis.append(s)
#			for i in xrange(self.Ns-1):
#				t = (s | (s - 1)) + 1
#				s = t | ((((t & -t) / (s & -s)) >> 1) - 1) 
#				mbasis.append(s)
		else:
			self.conserved=""
			self.Ns=2**L
			self.Mcon=False
			self.symm=False # No symmetries here. at all so each integer corresponds to the number in the hilbert space.
			mbasis=xrange(self.Ns)

		self.basis=mbasis


	def FindZstate(self,s):	
		if self.conserved:
			i = bisect_left(self.basis, s)
			if i != self.Ns and self.basis[i] == s:
				return i
			else:
				return -1
		else: return s




	def Op(self,J,opstr,indx,st):
		s1=self.basis[st]
		ME,s2=SpinOp(s1,opstr,indx)
		stt=self.FindZstate(s2)
		if stt >= 0:
			return [J*ME,st,stt]
		else:
			return [0,st,st]










