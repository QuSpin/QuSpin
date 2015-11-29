from Basis import BasisError, Basis
from PBC_Basis import PeriodicBasis1D
from OBC_Basis import OpenBasis1D
import BitOps

class Basis1D:
	def __init__(self,Length,**basis_params):

		# if arguement is not passed, then the get function returns None which should be handled by the lower basis classes.
		Nup=basis_params.get("Nup")
		kblock=basis_params.get("kblock")
		zblock=basis_params.get("zblock")
		pblock=basis_params.get("pblock")
		pzblock=basis_params.get("pzblock")
		a=basis_params.get("a")

		if a == None: a=1

		# testing blocks for basis
		if (type(kblock) is int):
			if (type(zblock) is int) or (type(pblock) is int) or (type(pzblock) is int):
				raise BasisError("Translation, spin inversion, and parity symmetries are not implimented at this time.")
			else:
				self.B=PeriodicBasis1D(Length,Nup=Nup,kblock=kblock,a=a)
		elif (type(zblock) is int) or (type(pblock) is int) or (type(pzblock) is int):
			self.B=OpenBasis1D(Length,Nup=Nup,zblock=zblock,pblock=pblock,pzblock=pzblock)
		else:
			self.B=Basis(Length,Nup=Nup)
		
		self.Nup=Nup
		self.kblock=kblock
		self.a=a
		self.zblock=zblock
		self.pblock=pblock
		self.pzblock=pzblock
		self.Ns=self.B.Ns

	def Op(self,J,dtype,opstr,indx):
		return self.B.Op(J,dtype,opstr,indx)

	def __call__(self,J,dtype,opstr,indx):
		return self.B.Op(J,dtype,opstr,indx)





