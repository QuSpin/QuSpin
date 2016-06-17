import numpy as _np
from scipy import sparse as _sp
from ..base import basis



class ho_basis(basis):
	def __init__(self,Np):
		if (type(Np) is not int):
			raise ValueError("expecting integer for Np")

		self.Np = Np
		self.Ns = Np+1
		self.dtype = _np.min_scalar_type(-self.Ns)
		self.basis = _np.arange(self.Ns,dtype=self.dtype)



	def get_vec(self,v0,sparse=True):
		if self.Ns <= 0:
			return _np.array([])
		if v0.ndim == 1:
			if v0.shape[0] != self.Ns:
				raise ValueError("v0 has incompatible dimensions with basis")
			v0 = v0.reshape((-1,1))
		elif v0.ndim == 2:
			if v0.shape[0] != self.Ns:
				raise ValueError("v0 has incompatible dimensions with basis")
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")

		if sparse:
			return _sp.csr_matrix(v0)
		else:
			return v0


	def Op(self,dtype,J,opstr,*args):

		row = _np.array(self.basis)
		col = _np.array(self.basis)
		ME = _np.ones((self.Ns,),dtype=dtype)
		for o in opstr[::-1]:
			if o == "I":
				continue
			elif o == "+":
				col += 1
				ME *= _np.sqrt(dtype(_np.abs(col)))
			elif o == "-":
				ME *= _np.sqrt(dtype(_np.abs(col)))
				col -= 1
			else:
				raise Exception("operator symbol {0} not recognized".format(o))

		mask = ( col < 0)
		mask += (col > (self.Ns))
		ME[mask] *= 0 
		ME *= J

		return ME,row,col		



			
