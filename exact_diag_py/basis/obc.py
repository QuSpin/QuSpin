# python 2.7 modules
from numpy import int32 as _index_type
from numpy import array as _array
import numpy as _np
# local modules
from base import base, BasisError,ncr

from constructors.constructors import *

from constructors import make_z_basis,make_m_z_basis
from constructors import make_p_basis,make_m_p_basis
from constructors import make_pz_basis,make_m_pz_basis
from constructors import make_p_z_basis,make_m_p_z_basis





# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)


# this is a dictionary which given a set of symmetries links to a function which does the correction actino for that set of symmtries.
op={"Z":op_z,
		"M & Z":op_z,
		"P":op_p,
		"M & P":op_p,
		"PZ":op_pz,
		"M & PZ":op_pz,
		"P & Z":op_p_z,
		"M & P & Z":op_p_z}



class obc(base):
	def __init__(self,L,**blocks):
		# This function in the constructor of the class:
		#		L: length of the chain
		#		Nup: number of up spins if restricting magnetization sector. 
		#		pblock: the number associated with parity quantum number of the block
		#		zblock: the number associated with spin inversion quantum number of the block
		#		pzblock: the number associated with parity + spin inversion quantum number of the block

		#	Note: the PZ block assumes the Hamiltonian is invariant under the total transformation PZ, 
		#				but not each transformation separately.
		Nup=blocks.get("Nup")
		zblock=blocks.get("zblock")
		pblock=blocks.get("pblock")
		pzblock=blocks.get("pzblock")
		a=blocks.get("a")
		self.blocks=blocks
		if a is None:
			a=1
			blocks["a"]=1

		if L>30: raise NotImplementedError('basis can only be constructed for L<31')
		self.L=L
		if type(Nup) is int:
			if Nup < 0 or Nup > L: raise BasisError("0 <= Nup <= %d" % L)
			self.Nup=Nup
			self.conserved="M"
			self.Ns=ncr(L,Nup) 
		else:
			self.conserved=""
			self.Ns=2**L


		if (type(pblock) is int) and (abs(pblock) != 1):
			raise BasisError("pblock must be either +/- 1")
		if (type(zblock) is int) and (abs(zblock) != 1):
			raise BasisError("zblock must be either +/- 1")
		if (type(pzblock) is int) and (abs(pzblock) != 1):
			raise BasisError("pzblock must be either +/- 1")

		if ((type(zblock) is int) or (type(pzblock) is int)) and (type(Nup) is int) and (Nup != L/2):
			raise BasisError("Spin inversion symmetry only works for Nup=L/2")

		if(L >= 10): frac = 0.6
		else: frac = 0.7
		# if symmetry is needed, the reference states must be found.
		# This is done through the CheckState function. Depending on
		# the symmetry, a different function must be used. Also if multiple
		# symmetries are used, the Checkstate functions be called
		# sequentially in order to check the state for all symmetries used.
		if (type(pblock) is int) and (type(zblock) is int):
			if self.conserved: self.conserved += " & P & Z"
			else: self.conserved += "P & Z"
			self.Ns = int(_np.ceil(self.Ns*0.5*frac))
			
			self.basis = _np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty((self.Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_p_z_basis(L,Nup,pblock,zblock,self.N,self.basis)
			else:
				self.Ns = make_p_z_basis(L,pblock,zblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]



		elif type(pblock) is int:
			if self.conserved: self.conserved += " & P"
			else: self.conserved = "P"
			self.Ns = int(_np.ceil(self.Ns*frac))
			
			self.basis = _np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty((self.Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_p_basis(L,Nup,pblock,self.N,self.basis)
			else:
				self.Ns = make_p_basis(L,pblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]



		elif type(zblock) is int:
			if self.conserved: self.conserved += " & Z"
			else: self.conserved += "Z"
			self.Ns = int(_np.ceil(self.Ns*1.0))

			
			self.basis = _np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty((self.Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_z_basis(L,Nup,self.N,self.basis)
			else:
				self.Ns = make_z_basis(L,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]
				
		elif type(pzblock) is int:
			if self.conserved: self.conserved += " & PZ"
			else: self.conserved += "PZ"
			self.Ns = int(_np.ceil(self.Ns*frac))
			
			self.basis = _np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty((self.Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_pz_basis(L,Nup,pzblock,self.N,self.basis)
			else:
				self.Ns = make_pz_basis(L,pzblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]
		else: 
			raise BasisError("if no symmetries used use base class")	

	def Op(self,J,dtype,opstr,indx):
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')

		return  op[self.conserved](self.N,self.basis,opstr,indx,J,self.L,dtype,**self.blocks)
		






