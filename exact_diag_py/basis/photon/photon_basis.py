import numpy as _np
from scipy import sparse as _sp
from ..base import basis
from ..base import tensor


class photon_basis(tensor):
	def __init__(self,basis_constructor,*constructor_args,**blocks)
		Ntot = blocks.get("Ntot")
		Hph = blocks.get("Hph")

		if type(Ntot) is None:
			self._pcon = False
			other_basis = basis_constructor(*constructor_args,**blocks)
			ph_basis = ho_basis(Nph)
			tensor.__init__(other_basis,ph_basis)
			self._blocks = blocks

		else:
			pass
			# ... do particle conserving constructor




	def get_vec(self,v0,sparse=True):
		if not self._pcon:
			return tensor.get_vec(self,v0,sparse=sparse)
		else:
			raise NotImplementedError("get_vec not implimented for particle conservation symm.")
			# ... impliment get_vec for particle conservation here



	def get_proj(self,dtype):
		if not self._pcon:
			return tensor.get_proj(self,dtype)	
		else:
			raise NotImplementedError("get_proj not implimented for particle conservation symm.")
			# ... impliment get_proj for particle conservation here
				













# helper class which calcualates ho matrix elements
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


	def get_proj(self,dtype):
		return _sp.identity(self.Ns,dtype=dtype)


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



			
