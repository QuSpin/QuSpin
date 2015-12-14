# python 2.7 modules
from numpy import int32 as _index_type
from numpy import array as _array
# local modules
from base import base, BasisError

from constructors import RefState_M
from constructors import RefState_Z
from constructors import RefState_P
from constructors import RefState_PZ
from constructors import RefState_P_Z
from constructors import SpinOp

from constructors import make_z_basis
from constructors import make_p_basis
from constructors import make_pz_basis
from constructors import make_p_z_basis





# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)


# this is a dictionary which given a set of symmetries links to a function which does the correction actino for that set of symmtries.
RefState={"M":RefState_M,
					"Z":RefState_Z,
					"M & Z":RefState_Z,
					"P":RefState_P,
					"M & P":RefState_P,
					"PZ":RefState_PZ,
					"M & PZ":RefState_PZ,
					"P & Z":RefState_P_Z,
					"M & P & Z":RefState_P_Z}




class obc(base):
	def __init__(self,L,Nup=None,pblock=None,zblock=None,pzblock=None):
		# This function in the constructor of the class:
		#		L: length of the chain
		#		Nup: number of up spins if restricting magnetization sector. 
		#		pblock: the number associated with parity quantum number of the block
		#		zblock: the number associated with spin inversion quantum number of the block
		#		pzblock: the number associated with parity + spin inversion quantum number of the block

		#	Note: the PZ block assumes the Hamiltonian is invariant under the total transformation PZ, 
		#				but not each transformation separately.

		base.__init__(self,L,Nup) # this calls the initialization of the basis class which initializes the basis list given Nup and Mcon/symm
		self.kblock=None
		self.a=1

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

			if self.conserved: self.conserved += " & P & Z"
			else: self.conserved += "P & Z"

			self.pblock = pblock
			self.zblock = zblock
			self.pzblock = 0
			self.N=make_p_z_basis(L,self.basis,pblock,zblock)

		elif type(pblock) is int:
			if abs(pblock) != 1:
				raise BasisError("pblock must be either +/- 1")

			if self.conserved: self.conserved += " & P"
			else: self.conserved = "P"

			self.pblock = pblock
			self.zblock = 0
			self.pzblock = 0
			self.N=make_p_basis(L,self.basis,pblock)

		elif type(zblock) is int:
			if abs(zblock) != 1:
				raise BasisError("zblock must be either +/- 1")

			if self.conserved: self.conserved += " & Z"
			else: self.conserved += "Z"

			self.pblock = 0
			self.zblock = zblock
			self.pzblock = 0
			self.N=make_z_basis(L,self.basis)

		elif type(pzblock) is int:
			if abs(pzblock) != 1:
				raise BasisError("pzblock must be either +/- 1")

			if self.conserved: self.conserved += " & PZ"
			else: self.conserved += "PZ"

			self.pblock = 0
			self.zblock = 0
			self.pzblock = pzblock
			self.N=make_pz_basis(L,self.basis,pzblock)
		else: 
			raise BasisError("if no symmetries are used use Basis class")

		self.basis=self.basis[self.basis != -1]
		self.N=self.N[self.N != -1]
		self.Ns=len(self.basis)	


	def Op(self,J,dtype,opstr,indx):
		row=_array(xrange(self.Ns),dtype=_index_type)

		ME,col=SpinOp(self.basis,opstr,indx,dtype)
		RefState[self.conserved](self.basis,col,self.L,self.N,ME,self.pblock,self.zblock,self.pzblock)

		# remove any states that give matrix elements which are no in the basis.
		mask=col>=0
		ME=ME[mask]
		col=col[mask]
		row=row[mask]

		col-=1 # fortran routines by default start at index 1 while here we start at 0.
		ME*=J

		return ME,row,col
		






