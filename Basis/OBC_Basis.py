# python 2.7 modules
from array import array as vec
from numpy import sqrt,ones,asarray,int32
# local modules
from BitOps import * # loading modules for bit operations.
from SpinOps import SpinOp
from Basis import Basis, BasisError
from Basis_fortran import *

# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)


def CheckStatePZ(pz,s,L,rpz=2):
	t=s
	t=flip_lr(t,L)
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
	t=flip_lr(t,L)
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




def RefState_Z(s,L):
	r=s; g=0; q=0; qg=0;
	t = flip_all(s,L)
	if t < s:
		r=t; g=1;
	return r,q,g,qg


def RefState_P(s,L):
	r=s; g=0; q=0; qg=0;
	t = flip_lr(s,L)
	if t < s:
		r=t; q=1;		
	return r,q,g,qg

def RefState_PZ(s,L):
	r=s; g=0; q=0; qg=0;
	t = flip_lr(s,L)
	t = flip_all(t,L)
	if t < s:
		r=t; qg=1;		

	return r,q,g,qg


def RefState_P_Z(s,L):
	t=s; r=s; g=0; q=0; qg=0;
	t = flip_all(t,L)
	if t < r:
		r=t; g=1;q=0;
	t=s
	t = flip_lr(t,L)
	if t < r:
		r=t; q=1; g=0;
	t=flip_all(t,L)
	if t < r:
		r=t; q=1; g=1;
	
	return r,q,g,qg

RefState={"":(lambda s,L:s,0,0,0),"M":(lambda s,L:s,0,0,0),"Z":RefState_Z,"M & Z":RefState_Z,"P":RefState_P,"M & P":RefState_P,"PZ":RefState_PZ,"M & PZ":RefState_PZ,"P & Z":RefState_P_Z,"M & P & Z":RefState_P_Z}




class OpenBasis1D(Basis):
	def __init__(self,L,Nup=None,pblock=None,zblock=None,pzblock=None):
		# This function in the constructor of the class:
		#		L: length of the chain
		#		Nup: number of up spins if restricting magnetization sector. 
		#		pblock: the number associated with parity quantum number of the block
		#		zblock: the number associated with spin inversion quantum number of the block
		#		pzblock: the number associated with parity + spin inversion quantum number of the block

		#	Note: the PZ block assumes the Hamiltonian is invariant under the total transformation PZ, 
		#				but not each transformation separately.

		Basis.__init__(self,L,Nup) # this calls the initialization of the basis class which initializes the basis list given Nup and Mcon/symm
		self.kblock=None
		self.a=1
		self.basis=asarray(self.basis,dtype=int32)

		# if symmetry is needed, the reference states must be found.
		# This is done through the CheckState function. Depending on
		# the symmetry, a different function must be used. Also if multiple
		# symmetries are used, the Checkstate functions be called
		# sequentially in order to check the state for all symmetries used.
		if (type(pblock) is int) and (type(zblock) is int):
			if abs(pblock) != 1:
				raise BasisError("pblock must be either +/- 1")
			if abs(zblock) != 1:
				raise BasisError("zblock must be either +/- 1")
			if self.conserved:
				self.conserved += " & P & Z"
			else:
				self.conserved += "P & Z"

			self.pblock = pblock
			self.zblock = zblock
			self.pzblock = 1
			self.N=make_p_z_basis(L,self.basis,pblock,zblock)
			self.basis=self.basis[self.basis != -1]
			self.N=self.N[self.N != -1]
			self.Ns=len(self.basis)
		elif type(pblock) is int:
			if abs(pblock) != 1:
				raise BasisError("pblock must be either +/- 1")
			if self.conserved:
				self.conserved += " & P"
			else:
				self.conserved = "P"
			self.pblock = pblock
			self.zblock = 1
			self.pzblock = 1
			self.N=make_p_basis(L,self.basis,pblock)
			self.basis=self.basis[self.basis != -1]
			self.N=self.N[self.N != -1]
			self.Ns=len(self.basis)
		elif type(zblock) is int:
			if abs(zblock) != 1:
				raise BasisError("zblock must be either +/- 1")
			if self.conserved:
				self.conserved += " & Z"
			else:
				self.conserved += "Z"

			self.pblock = 1
			self.zblock = zblock
			self.pzblock = 1
			self.N=make_z_basis(L,self.basis)
			self.basis=self.basis[self.basis != -1]
			self.N=self.N[self.N != -1]
			self.Ns=len(self.basis)
		elif type(pzblock) is int:
			if abs(pzblock) != 1:
				raise BasisError("pzblock must be either +/- 1")
			if self.conserved:
				self.conserved += " & PZ"
			else:
				self.conserved += "PZ"

			self.pblock = 1
			self.zblock = 1
			self.pzblock = pzblock
			self.N=make_pz_basis(L,self.basis,pzblock)
			self.basis=self.basis[self.basis != -1]
			self.N=self.N[self.N != -1]
			self.Ns=len(self.basis)	
		else: 
			N = ones((self.Ns,),dtype=int32)
			self.pblock = 1
			self.zblock = 1
			self.pzblock = 1


	def Op(self,J,opstr,indx,st):
		# This function find the matrix elemement and state which opstr creates
		# after acting on an inputed state index.
		#		J: coupling in front of opstr
		#		st: index of a local state in the basis for which the opstor will act on
		#		opstr: string which contains a list of operators which  
		#		indx: a list of ordered indices which tell which operator in opstr live on the lattice.
		s1=self.basis[st]
		ME,s2=SpinOp(s1,opstr,indx)
		s2,q,g,qg=RefState[self.conserved](s2,self.L)
		stt=self.FindZstate(s2)
		if stt >= 0: 
			ME *= sqrt( float(self.N[stt])/self.N[st])*J*(self.pblock**q)*(self.zblock**g)*(self.pzblock**qg)	
			return [ME,st,stt]	
		else:
			return [0,st,st]	
		






