# python 2.7 modules
from numpy import int32 as _index_type
from numpy import array as _array
import numpy as _np
# local modules
from base import base, BasisError,ncr

from constructors import op_t,op_t_z,op_t_p,op_t_pz,op_t_p_z

from constructors import make_t_basis,make_m_t_basis
from constructors import make_t_z_basis,make_m_t_z_basis
from constructors import make_t_p_basis,make_m_t_p_basis
from constructors import make_t_pz_basis,make_m_t_pz_basis
from constructors import make_t_p_z_basis,make_m_t_p_z_basis






# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)


# this is a dictionary which given a set of symmetries links to a function which does the correction actino for that set of symmtries.
op={"T":op_t,
		"M & T":op_t,
		"T & Z":op_t_z,
		"M & T & Z":op_t_z,
		"T & P":op_t_p,
		"M & T & P":op_t_p,
		"T & PZ":op_t_pz,
		"M & T & PZ":op_t_pz,
		"T & P & Z":op_t_p_z,
		"M & T & P & Z":op_t_p_z}




class pbc(base):
	def __init__(self,L,**blocks):
		# This function in the constructor of the class:
		#		L: length of the chain
		#		Nup: number of up spins if restricting magnetization sector. 
		#		kblock: number which represents the momentum block 
		Nup=blocks.get("Nup")
		kblock=blocks.get("kblock")
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


		if (type(zblock) is int) and (abs(zblock) != 1):
			raise ValueError("zblock must be +/- 1.")

		if (type(pblock) is int) and (abs(pblock) != 1):
			raise ValueError("pblock must be +/- 1.")

		if (type(pzblock) is int) and (abs(pzblock) != 1):
			raise ValueError("pzblock must be +/- 1.")

		if (type(Nup) is int) and ((type(zblock) is int) or (type(pzblock) is int)):
			if (L % 2) != 0:
				raise ValueError("spin inversion symmetry must be used with even number of sites.")
			if Nup != L/2:
				raise ValueError("spin inversion symmetry only reduces the 0 magnetization sector.")

		if(L >= 10): frac = 0.6
		else: frac = 0.7

		if L > 1: L_m = L-1
		else: L_m = 1

		#base.__init__(self,L,Nup) # this calls the initialization of the basis class which initializes the basis list given Nup and Mcon/symm
		# if symmetry is needed, the reference states must be found.
		# This is done through via the fortran constructors.
		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & P & Z"
			else: self.conserved = "T & P & Z"
			self.Ns = int(_np.ceil(self.Ns*a*(1.1)/float(L_m)))

			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			self.m=_np.empty(self.basis.shape,dtype=_np.int16)
			if (type(Nup) is int):
				self.Ns = make_m_t_p_z_basis(L,Nup,pblock,zblock,kblock,a,self.N,self.m,self.basis)
			else:
				self.Ns = make_t_p_z_basis(L,pblock,zblock,kblock,a,self.N,self.m,self.basis)

			self.N = self.N[:self.Ns]
			self.m = self.m[:self.Ns]
			self.basis = self.basis[:self.Ns]

		elif (type(kblock) is int) and (type(pzblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & PZ"
			else: self.conserved = "T & PZ"
			self.Ns = int(_np.ceil(self.Ns*a*(1.1)/float(L_m)))

			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			self.m=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_t_pz_basis(L,Nup,pzblock,kblock,a,self.N,self.m,self.basis)
			else:
				self.Ns = make_t_pz_basis(L,pzblock,kblock,a,self.N,self.m,self.basis)

			self.N = self.N[:self.Ns]
			self.m = self.m[:self.Ns]
			self.basis = self.basis[:self.Ns]

		elif (type(kblock) is int) and (type(pblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & P"
			else: self.conserved = "T & P"
			self.Ns = int(_np.ceil(self.Ns*a*(1.1)/float(L_m)))


			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			self.m=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_t_p_basis(L,Nup,pblock,kblock,a,self.N,self.m,self.basis)
			else:
				self.Ns = make_t_p_basis(L,pblock,kblock,a,self.N,self.m,self.basis)

			self.N = self.N[:self.Ns]
			self.m = self.m[:self.Ns]
			self.basis = self.basis[:self.Ns]

		elif (type(kblock) is int) and (type(zblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & Z"
			else: self.conserved = "T & Z"
			self.Ns = int(_np.ceil((frac*self.Ns*a)/float(L_m)))

			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			self.m=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_t_z_basis(L,Nup,zblock,kblock,a,self.N,self.m,self.basis)
			else:
				self.Ns = make_t_z_basis(L,zblock,kblock,a,self.N,self.m,self.basis)

			self.N = self.N[:self.Ns]
			self.m = self.m[:self.Ns]
			self.basis = self.basis[:self.Ns]
	
		elif type(kblock) is int:
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T"
			else: self.conserved = "T"
			self.Ns = int(_np.ceil(self.Ns*a*(1.1)/float(L_m)))

			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_t_basis(L,Nup,kblock,a,self.N,self.basis)
			else:
				self.Ns = make_t_basis(L,kblock,a,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.m = -_np.ones((self.Ns,),dtype=_np.int8)


		else: 
			# any other ideas for this?
			raise BasisError("if no symmetries used use base class")
		
	def Op(self,J,dtype,opstr,indx):
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')
		return op[self.conserved](self.N,self.m,self.basis,opstr,indx,J,self.L,dtype,**self.blocks)






