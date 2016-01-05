# python 2.7 modules
from numpy import int32 as _index_type
from numpy import array as _array
import numpy as _np
# local modules
from base import base, BasisError

from constructors import op_t

from constructors import make_t_basis






# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)


# this is a dictionary which given a set of symmetries links to a function which does the correction actino for that set of symmtries.
# this is a dictionary which given a set of symmetries links to a function which does the correction actino for that set of symmtries.
op={"T":op_t,
		"M & T":op_t}




class pbc(base):
	def __init__(self,L,**blocks):
		# This function in the constructor of the class:
		#		L: length of the chain
		#		Nup: number of up spins if restricting magnetization sector. 
		#		kblock: number which represents the momentum block 
		Nup=blocks.get("Nup")
		kblock=blocks.get("kblock")
		a=blocks.get("a")
		if a is None:
			a=1
			blocks["a"]=1


		base.__init__(self,L,Nup) # this calls the initialization of the basis class which initializes the basis list given Nup and Mcon/symm
		self.blocks=blocks
		# if symmetry is needed, the reference states must be found.
		# This is done through via the fortran constructors.
		if type(kblock) is int:
			if kblock < 0 or kblock >= L: raise BasisError("0 <= kblock < "+str(L))
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T"
			else: self.conserved = "T"
			self.N=make_t_basis(L,self.basis,kblock,a)
		else: 
			# any other ideas for this?
			raise BasisError("if no symmetries used use base class")

		self.basis=self.basis[self.basis != -1]
		self.N=self.N[self.N != -1]
		self.Ns=len(self.basis)	
		

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
		






