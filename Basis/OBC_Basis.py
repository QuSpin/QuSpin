from BitOps import * # loading modules for bit operations.
from SpinOps import SpinOp
from Z_Basis import Basis, BasisError

from array import array as vec
from numpy import sqrt





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
			self.symm = True
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
			self.symm = True
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
			self.symm = True
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
			self.symm = True
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


	def Op(self,J,st,opstr,indx,pauli=False):
		if self.Pcon or self.Zcon or self.PZcon:
			s1=self.basis[st]
			ME,s2=SpinOp(s1,opstr,indx,pauli=pauli)
			s2,q,g,qg=self.RefState(s2)
			stt=self.FindZstate(s2)
			#print st,int2bin(s1,self.L),int2bin(exchangeBits(s1,i,j),self.L), stt,int2bin(s2,self.L), q, g, [i,j]
			if stt >= 0:
				if self.Pcon and self.Zcon:
					ME *= sqrt( float(self.Npz[stt])/self.Npz[st])*J*self.p**(q)*self.z**(g)
				elif self.Pcon:
					ME *= sqrt( float(self.Np[stt])/(self.Np[st]))*J*self.p**(q)
				elif self.Zcon:
					ME *=  0.5*J*self.z**(g)
				elif self.PZcon:
					ME *= sqrt( float(self.Npz[stt])/self.Npz[st] )*J*self.pz**(qg)		
			else:
				ME = 0.0
				stt = st
			return [ME,st,stt]	
		else:
			return Basis.Op(self,J,st,opstr,indx,pauli=pauli)






