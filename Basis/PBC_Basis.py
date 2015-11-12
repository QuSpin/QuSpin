from BitOps import * # loading modules for bit operations.
from SpinOps import SpinOp
from Z_Basis import Basis, BasisError
from array import array as vec
from numpy import pi,exp, sqrt


# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)



def CheckStateT(kblock,L,s,T=1):
	# this is a function defined in [1]
	# It is used to check if the integer inputed is a reference state for a state with momentum k.
	#		kblock: the number associated with the momentum (i.e. k=2*pi*kblock/L)
	#		L: length of the system
	#		s: integer which represents a spin config in Sz basis
	#		T: number of sites to translate by, not 1 if the unit cell on the lattice has 2 sites in it.
	t=s
	for i in xrange(1,L+1,T):
		t = shift(t,-T,L)
		if t < s:
			return -1
		elif t==s:
			if kblock % (L/i) != 0: return -1 # need to check the shift condition 
			return i



# child class of Basis, this is the momentum conserving basis:
# because it is a child class of Basis, it inherits its methods like FindZstate which searches basis for states
class PeriodicBasis1D(Basis):
	def __init__(self,L,Nup=None,kblock=None,a=1):
		# This function in the constructor of the class:
		#		L: length of the chain
		#		Nup: number of up spins if restricting magnetization sector. 
		#		kblock: the number associated with the momentum block which basis is restricted to (i.e. k=2*pi*kblock/L)
 		#		a: number of lattice spaces between unit cells.

		Basis.__init__(self,L,Nup) # this calls the initialization of the basis class which initializes the basis list given Nup and Mcon/symm
		zbasis=self.basis # take initialized basis from Basis class and store in separate array to access, then overwrite basis.
		self.Pcon=False
		self.Zcon=False
		self.PZcon=False
		self.pblock=None
		self.zblock=None
		self.pzblock=None


		# if symmetry is needed, the reference states must be found.
		# This is done through the CheckState function. Depending on
		# the symmetry, a different function must be used. Also if multiple
		# symmetries are used, the Checkstate functions be called
		# sequentially in order to check the state for all symmetries used.
		if type(kblock) is int:
			if kblock < 0 or kblock >= L: raise BasisError("0<= kblock < "+str(L))
			self.a=a
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
		# this function takes an integer s which represents a spin configuration in the Sz basis, then tries to find its 
		# reference state depending on the symmetries specified by the user. it does this by applying the various symmetry 
		# operations on the state and seeing whether a smaller integer is produced. This smaller integer by definition is the
		# reference state.
		# it returns r which is the reference state. l is the number of times the translation operator had to act.
		# This information is needed to calculate the matrix element s between states in this basis [1].
		t=s; r=s; l=0;
		for i in xrange(1,self.L+1,self.a):
			t=shift(t,-self.a,self.L)
			if t < r:
				r=t; l=i;

		return r,l


	
	def Op(self,J,st,opstr,indx):	
		# This function find the matrix elemement and state which opstr creates
		# after acting on an inputed state index.
		#		J: coupling in front of opstr
		#		st: index of a local state in the basis for which the opstor will act on
		#		opstr: string which contains a list of operators which  
		#		indx: a list of ordered indices which tell which operator in opstr live on the lattice.

		if self.Kcon: # if the user wants to use momentum basis, special care must be taken [1]
			s1=self.basis[st]
			ME,s2=SpinOp(s1,opstr,indx)
			s2,l=self.RefState(s2)
			stt=self.FindZstate(s2) # if reference state not found in basis, this is not a valid matrix element.
			if stt >= 0:
				ME *= sqrt(float(self.R[st])/self.R[stt])*J*exp(-1j*self.k*l)
			else:
				ME=0.0;	stt=st
			return [ME,st,stt]
		else: # else, no special care is needed, just use the equivilant method from Basis class 
			return Basis.Op(self,J,st,opstr,indx)
		
		


