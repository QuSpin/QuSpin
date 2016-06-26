import numpy as _np
from scipy import sparse as _sp
from ..base import basis,MAXPRINT
from ..base import tensor


class photon_basis(tensor):
	def __init__(self,basis_constructor,*constructor_args,**blocks):
		Ntot = blocks.get("Ntot")
		Nph = blocks.get("Nph")
		if Ntot is None:
			self._pcon = False
			if Nph is None: raise TypeError("If Ntot not specified, Nph must specify the number of photon states.")
			if type(Nph) is not int: raise TypeError("Nph must be integer")
			b1 = basis_constructor(*constructor_args,**blocks)
			b2 = ho_basis(Nph)
			tensor.__init__(self,b1,b2)
			self._blocks = blocks

		else:
			if type(Ntot) is not int: raise TypeError("Ntot must be integer")
			del blocks["Ntot"]
			self._pcon=True
			self._b1 = basis_constructor(*constructor_args,_Np=Ntot,**blocks)
			if isinstance(self._b1,tensor): raise TypeError("Can only create photon basis with non-tensor type basis")
			self._b2 = ho_basis(Ntot)
			self._n = Ntot - self._b1._Np
			self._blocks = blocks
			self._Ns = self._b1._Ns
			self._operators = self._b1._operators +"\n"+ self._b2._operators
			



	def Op(self,opstr,indx,J,dtype,pauli):
		if self._Ns <= 0:
			return [],[],[]

		if not self._pcon:
			return tensor.Op(self,opstr,indx,J,dtype,pauli)
		else:
			# read off spin and photon operators
			n=opstr.count("|")
			if n > 1: 
				raise ValueError("only one '|' charactor allowed")
			i = opstr.index("|")
			indx1 = indx[:i]
			indx2 = indx[i:]
			

			opstr1,opstr2=opstr.split("|")

			# calculates matrix elements of spin and photon basis

			ME_ph,row_ph,col_ph =  self._b2.Op(opstr2,indx2,J,dtype,pauli)
			ME, row, col  =	self._b1.Op(opstr1,indx1,J,dtype,pauli)

			# calculate total matrix element
			ME *= ME_ph[self._n[row]]

			mask = ME != dtype(0.0)
			row = row[mask]
			col = col[mask]
			ME = ME[mask]

			del ME_ph, row_ph, col_ph

			return ME, row, col	

	def _get__str__(self):
		if not self._pcon:
			return tensor._get__str__(self)
		else:
			n_digits = int(_np.ceil(_np.log10(self._Ns)))
			str_list_1 = self._b1._get__str__()
			temp = "\t{0:"+str(n_digits)+"d}  "
			str_list=[]
			for b1 in str_list_1:
				b1 = b1.split()
				s1 = b1[1]
				i1 = int(b1[0])
				s2 = "|{0}>".format(self._n[i1])
				str_list.append((temp.format(i1))+"\t"+s1+s2)

			if self._Ns > MAXPRINT:
				half = MAXPRINT//2
				str_list_1 = str_list[:half]
				str_list_2 = str_list[-half:]

				str_list = str_list_1
				str_list.extend(str_list_2)	

			return str_list	





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

		self._Np = Np
		self._Ns = Np+1
		self._dtype = _np.min_scalar_type(-self.Ns)
		self._basis = _np.arange(self.Ns,dtype=_np.min_scalar_type(self.Ns))
		self._operators = ("availible operators for ho_basis:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tn: number operator")



	def get_vec(self,v0,sparse=True):
		if self._Ns <= 0:
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



	def _get__str__(self):
		n_digits = int(_np.ceil(_np.log10(self._Ns)))
		temp = "\t{0:"+str(n_digits)+"d}  "+"|{0}>"

		if self._Ns > MAXPRINT:
			half = MAXPRINT // 2
			str_list = [temp.format(i,b) for i,b in zip(xrange(half),self._basis[:half])]
			str_list.extend([temp.format(i,b) for i,b in zip(xrange(self._Ns-half,self._Ns,1),self._basis[-half:])])
		else:
			str_list = [temp.format(i,b) for i,b in enumerate(self._basis)]

		return str_list

		


	def get_proj(self,dtype):
		return _sp.identity(self.Ns,dtype=dtype)


	def Op(self,opstr,indx,J,dtype,*args):

		row = _np.array(self._basis)
		col = _np.array(self._basis)
		ME = _np.ones((self._Ns,),dtype=dtype)
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')


		for o in opstr[::-1]:
			if o == "I":
				continue
			elif o == "n":
				ME *= dtype(_np.abs(col))
			elif o == "+":
				col += 1
				ME *= _np.sqrt(dtype(_np.abs(col)))
			elif o == "-":
				ME *= _np.sqrt(dtype(_np.abs(col)))
				col -= 1
			else:
				raise Exception("operator symbol {0} not recognized".format(o))

		mask = ( col < 0)
		mask += (col > (self._Ns))
		ME[mask] *= 0 
		ME *= J

		return ME,row,col		



			
