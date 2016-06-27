import numpy as _np
from scipy import sparse as _sp
from ..base import basis




class photon(basis):
	def __init__(self,Nho,n_ph=0):
		if (type(Nho) is not int):
			raise ValueError("expecting integer for Nho")
		if (type(n_ph) is not int):
			raise ValueError("expecting integer for n_ph")

		self.n_ph = n_ph
		self.Nho = Nho
		self._Ns = Nho+1
		self.col_dtype = _np.min_scalar_type(-self._Ns)
		self.basis_dtype = _np.min_scalar_type(self._Ns)
		self.basis = _np.arange(self._Ns,dtype=self.basis_dtype)



	def get_vec(self,v0,sparse=True):
		if self._Ns <= 0:
			return _np.array([])
		if v0.ndim == 1:
			if v0.shape[0] != self._Ns:
				raise ValueError("v0 has incompatible dimensions with basis")
			v0 = v0.reshape((-1,1))
		elif v0.ndim == 2:
			if v0.shape[0] != self._Ns:
				raise ValueError("v0 has incompatible dimensions with basis")
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")

		if sparse:
			return _sp.csr_matrix(v0)
		else:
			return v0


	def Op(self,dtype,J,opstr,*args):

		row = _np.array(self.basis,dtype=self.col_dtype)
		col = _np.array(self.basis,dtype=self.col_dtype)
		ME = _np.ones((self._Ns,),dtype=dtype)
		for o in opstr[::-1]:
			if o == "I":
				continue
			elif o == "+":
				col += 1
				ME *= _np.sqrt(self.n_ph + dtype(_np.abs(col)))
			elif o == "-":
				ME *= _np.sqrt(self.n_ph + dtype(_np.abs(col)))
				col -= 1
			elif o == "n":
				ME *= dtype(_np.abs(col))
			else:
				raise Exception("operator symbol {0} not recognized".format(o))
		mask = ( col < 0)
		mask += (col > (self._Ns))
		ME[mask] *= 0 
		ME *= J

		return ME,row,col		



			
