from base import base as _base
from obc import obc as _obc
from pbc import pbc as _pbc


class basis1d:
	def __init__(self,Length,**basis_params):

		# if arguement is not passed, then the get function returns None which should be handled by the lower basis classes.
		Nup=basis_params.get("Nup")
		kblock=basis_params.get("kblock")
		pblock=basis_params.get("pblock")
		zblock=basis_params.get("zblock")
		pzblock=basis_params.get("pzblock")

		# testing blocks for basis
		if (type(kblock) is int):
			if (type(pblock) is int) or (type(pzblock) is int):
				raise NotImplementedError
			self.B=_pbc(Length,**basis_params)
		elif (type(zblock) is int) or (type(pblock) is int) or (type(pzblock) is int):
			self.B=_obc(Length,**basis_params)
		else:
			self.B=_base(Length,Nup=Nup)
		
		self.Ns=self.B.Ns

	def Op(self,J,dtype,opstr,indx):
		return self.B.Op(J,dtype,opstr,indx)
