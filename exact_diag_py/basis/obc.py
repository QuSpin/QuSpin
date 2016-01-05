# python 2.7 modules
from numpy import int32 as _index_type
from numpy import array as _array
import numpy as _np
# local modules
from base import base, BasisError

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
		pblock=blocks.get("pblock")
		zblock=blocks.get("zblock")
		pzblock=blocks.get("pzblock")
		self.blocks=blocks

		base.__init__(self,L,Nup) # this calls the initialization of the basis class which initializes the basis list given Nup and Mcon/symm

		if (type(pblock) is int) and (abs(pblock) != 1):
			raise BasisError("pblock must be either +/- 1")
		if (type(zblock) is int) and (abs(zblock) != 1):
			raise BasisError("zblock must be either +/- 1")
		if (type(pzblock) is int) and (abs(pzblock) != 1):
			raise BasisError("pzblock must be either +/- 1")

		if (type(zblock) is int) and (type(Nup) is int) and (Nup != L/2):
			raise BasisError("Spin inversion symmetry only works for Nup=L/2")


		# if symmetry is needed, the reference states must be found.
		# This is done through the CheckState function. Depending on
		# the symmetry, a different function must be used. Also if multiple
		# symmetries are used, the Checkstate functions be called
		# sequentially in order to check the state for all symmetries used.
		if (type(pblock) is int) and (type(zblock) is int):
			if self.conserved: self.conserved += " & P & Z"
			else: self.conserved += "P & Z"

			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_p_z_basis(L,Nup,pblock,zblock,self.N,self.basis)
			else:
				self.Ns = make_p_z_basis(L,pblock,zblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]



		elif type(pblock) is int:
			if self.conserved: self.conserved += " & P"
			else: self.conserved = "P"

			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_p_basis(L,Nup,pblock,self.N,self.basis)
			else:
				self.Ns = make_p_basis(L,pblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]



		elif type(zblock) is int:
			if self.conserved: self.conserved += " & Z"
			else: self.conserved += "Z"

			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_z_basis(L,Nup,self.N,self.basis)
			else:
				self.Ns = make_z_basis(L,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]
				
		elif type(pzblock) is int:
			if self.conserved: self.conserved += " & PZ"
			else: self.conserved += "PZ"

			if (type(Nup) is int) and (Nup != L/2):
				raise BasisError("Spin inversion symmetry only works for Nup=L/2")

			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = make_m_pz_basis(L,Nup,pzblock,self.N,self.basis)
			else:
				self.Ns = make_pz_basis(L,pzblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]


		else: 
			# any other ideas for this?
			raise BasisError("if no symmetries used use base class")		

	def Op(self,J,dtype,opstr,indx):
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')

		row=_array(xrange(self.Ns),dtype=_index_type)
		ME,col = op[self.conserved](self.N,self.basis,opstr,indx,self.L,dtype,**self.blocks)
		mask = col >= 0
		row = row[ mask ]
		col = col[ mask ]
		ME = ME[ mask ]
		col -= 1 #convert from fortran index to c index.
		ME*=J
		
		
		return ME,row,col
		






