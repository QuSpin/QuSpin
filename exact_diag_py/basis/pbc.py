# python 2.7 modules
from numpy import int32 as _index_type
from numpy import array as _array
import numpy as _np
# local modules
from base import base, BasisError

from constructors import RefState_M
from constructors import RefState_T
from constructors import SpinOp

from constructors import make_t_basis






# References:
# [1]: A. W. Sandvik, AIP Conf. Proc. 1297, 135 (2010)


# this is a dictionary which given a set of symmetries links to a function which does the correction actino for that set of symmtries.
# this is a dictionary which given a set of symmetries links to a function which does the correction actino for that set of symmtries.
RefState={"M":RefState_M,
					"T":RefState_T,
					"M & T":RefState_T}




class pbc(base):
	def __init__(self,L,Nup=None,a=1,kblock=None):
		# This function in the constructor of the class:
		#		L: length of the chain
		#		Nup: number of up spins if restricting magnetization sector. 
		#		kblock: number which represents the momentum block 

		base.__init__(self,L,Nup) # this calls the initialization of the basis class which initializes the basis list given Nup and Mcon/symm
		self.blocks={}
		# if symmetry is needed, the reference states must be found.
		# This is done through via the fortran constructors.
		if type(kblock) is int:
			if kblock < 0 or kblock >= L: raise BasisError("0<= kblock < "+str(L))
			self.a=a
			self.blocks["kblock"]=kblock
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T"
			else: self.conserved = "T"
			self.N=make_t_basis(L,self.basis,kblock,a)
		else: 
			self.N=_np.ones((Ns,),dtype=_np.int8)

		self.basis=self.basis[self.basis != -1]
		self.N=self.N[self.N != -1]
		self.Ns=len(self.basis)	


	def Op(self,J,dtype,opstr,indx):
		row=_array(xrange(self.Ns),dtype=_index_type)

		ME,col=SpinOp(self.basis,opstr,indx,dtype)
		RefState[self.conserved](self.basis,col,self.L,self.N,ME,a=self.a,**self.blocks)

		# remove any states that give matrix elements which are no in the basis.
		mask=col>=0
		ME=ME[mask]
		col=col[mask]
		row=row[mask]

		col-=1 # fortran routines by default start at index 1 while here we start at 0.
		ME*=J

		return ME,row,col
		






