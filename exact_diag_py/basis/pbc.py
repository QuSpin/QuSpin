# python 2.7 modules
from numpy import int32 as _index_type
from numpy import array as _array
import numpy as _np
# local modules
from base import base, BasisError

from constructors import op_t,op_t_z

from constructors import make_t_basis,make_m_t_basis
from constructors import make_t_z_basis,make_m_t_z_basis






# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)


# this is a dictionary which given a set of symmetries links to a function which does the correction actino for that set of symmtries.
op={"T":op_t,
		"M & T":op_t,
		"T & Z":op_t_z,
		"M & T & Z":op_t_z}




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
		if a is None:
			a=1
			blocks["a"]=1

		if (type(zblock) is int) and (abs(zblock) != 1):
			raise ValueError("zblock must be +/- 1.")

		if (type(pblock) is int) and (abs(pblock) != 1):
			raise ValueError("pblock must be +/- 1.")

		if (type(pzblock) is int) and (abs(pzblock) != 1):
			raise ValueError("pzblock must be +/- 1.")

		if (type(Nup) is int) and (type(zblock) is int):
			if (L % 2) != 0:
				raise ValueError("spin inversion symmetry must be used with even number of sites.")
			if Nup != L/2:
				raise ValueError("spin inversion symmetry only reduces the 0 magnetization sector.")

		base.__init__(self,L,Nup) # this calls the initialization of the basis class which initializes the basis list given Nup and Mcon/symm
		self.blocks=blocks
		# if symmetry is needed, the reference states must be found.
		# This is done through via the fortran constructors.

		if (type(kblock) is int) and (type(zblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & Z"
			else: self.conserved = "T & Z"

#			self.N,self.m = make_t_z_basis(L,self.basis,zblock,kblock,a)
#			self.m = self.m[self.basis != -1]
#			self.N = self.N[self.basis != -1]
#			self.basis = self.basis[self.basis != -1]
#			self.Ns = len(self.basis)

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

#			self.N = make_t_basis(L,self.basis,kblock,a)
#			self.m = -_np.ones((self.Ns,),dtype=_np.int8)
#			self.m = self.m[self.basis != -1]
#			self.N = self.N[self.basis != -1]
#			self.basis = self.basis[self.basis != -1]
#			self.Ns = len(self.basis)	

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

		row=_array(xrange(self.Ns),dtype=_index_type)
		ME,col = op[self.conserved](self.N,self.m,self.basis,opstr,indx,self.L,dtype,**self.blocks)
		mask = col >= 0
		row = row[ mask ]
		col = col[ mask ]
		ME = ME[ mask ]
		col -= 1 #convert from fortran index to c index.
		ME*=J
		
		
		return ME,row,col
		






