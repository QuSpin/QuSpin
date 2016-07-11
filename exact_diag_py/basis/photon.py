from .base import basis,MAXPRINT
from .tensor import tensor_basis

import numpy as _np
from scipy import sparse as _sp

import warnings


class photon_basis(tensor_basis):
	def __init__(self,basis_constructor,*constructor_args,**blocks):
		Ntot = blocks.get("Ntot")
		Nph = blocks.get("Nph")
		self.Nph = Nph
		self.Ntot = Ntot
		if Ntot is None:
			self._pcon = False
			if Nph is None: raise TypeError("If Ntot not specified, Nph must specify the number of photon states.")
			del blocks["Nph"]
			if type(Nph) is not int: raise TypeError("Nph must be integer")
			b1 = basis_constructor(*constructor_args,**blocks)
			b2 = ho_basis(Nph)
			tensor_basis.__init__(self,b1,b2)
			self._blocks = blocks
		else:
			if type(Ntot) is not int: raise TypeError("Ntot must be integer")
			del blocks["Ntot"]
			self._pcon=True
			self._b1 = basis_constructor(*constructor_args,_Np=Ntot,**blocks)
			if isinstance(self._b1,tensor_basis): raise TypeError("Can only create photon basis with non-tensor type basis")
			if not isinstance(self._b1,basis): raise TypeError("Can only create photon basis with basis type")
			self._b2 = ho_basis(Ntot)
			self._n = Ntot - self._b1._Np
			self._blocks = blocks
			self._Ns = self._b1._Ns
			self._operators = self._b1._operators +"\n"+ self._b2._operators
			



	def Op(self,opstr,indx,J,dtype,pauli):
		if self._Ns <= 0:
			return [],[],[]

		if not self._pcon:
			return tensor_basis.Op(self,opstr,indx,J,dtype,pauli)
		else:
			# read off spin and photon operators
			n=opstr.count("|")
			if n > 1: 
				raise ValueError("only one '|' charactor allowed")

			if len(opstr)-1 != len(indx):
				raise ValueError("not enough indices for opstr in: {0}, {1}".format(opstr,indx))

			i = opstr.index("|")
			indx1 = indx[:i]
			indx2 = indx[i:]			

			opstr1,opstr2=opstr.split("|")

			# calculates matrix elements of spin and photon basis

			ME_ph,row_ph,col_ph =  self._b2.Op(opstr2,indx2,1.0,dtype,pauli)
			ME, row, col  =	self._b1.Op(opstr1,indx1,J,dtype,pauli)

			# calculate total matrix element
			ME *= ME_ph[self._n[row]]

			mask = ME != dtype(0.0)
			row = row[mask]
			col = col[mask]
			ME = ME[mask]

			del ME_ph, row_ph, col_ph

			return ME, row, col	


	def check_hermitian(self,static_list,dynamic_list):
		# assumes static and dynamic lists are ordered


		# static list
		if static_list:

			static_expand = []
			static_expand_hc = []
			for opstr, bonds in static_list:
				# calculate conjugate opstr
				opstr_hc = opstr.replace('-','!')
				opstr_hc = opstr_hc.replace('+','-')
				opstr_hc = opstr_hc.replace('!','+')
				for bond in bonds:
					static_expand.append( (opstr,bond[0], tuple(bond[1:])) )
					static_expand_hc.append( (opstr_hc, _np.conj(bond[0]),tuple(bond[1:]) ) )

			# calculate non-hermitian elements
			diff = set( tuple(static_expand) ) - set( tuple(static_expand_hc) )
			if len(diff)!=0:
				unique_opstrs = list(set( zip(*tuple(diff))[0]) )
				warnings.warn("The following static operator strings contain non-hermitian couplings: {}".format(unique_opstrs),UserWarning,stacklevel=4)
				user_input = raw_input("Display all {} non-hermitian couplings? (y or n) ".format(len(diff)) )
				if user_input == 'y':
					print "   (opstr, coupling, indices)"
					for i in xrange(len(diff)):
						print "{}. {}".format(i+1, list(diff)[i])
				raise TypeError("Hamiltonian not hermitian!")
			
			
		if dynamic_list:
			# define arbitrarily complicated weird-ass number
			t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )

			dynamic_expand = []
			dynamic_expand_hc = []
			for opstr, bonds, f, f_args in dynamic_list:
				# calculate conjugate opstr
				opstr_hc = opstr.replace('-','!')
				opstr_hc = opstr_hc.replace('+','-')
				opstr_hc = opstr_hc.replace('!','+')
				for bond in bonds:
					dynamic_expand.append( (opstr,bond[0]*f(t,*f_args), tuple(bond[1:])) )
					dynamic_expand_hc.append( (opstr_hc, _np.conj(bond[0]*f(t,*f_args)),tuple(bond[1:]) ) )

			# calculate non-hermitian elements
			diff = set( tuple(dynamic_expand) ) - set( tuple(dynamic_expand_hc) )
			if len(diff)!=0:
				unique_opstrs = list(set( zip(*tuple(diff))[0]) )
				warnings.warn("The following dynamic operator strings contain non-hermitian couplings: {}".format(unique_opstrs),UserWarning,stacklevel=4)
				user_input = raw_input("Display all {} non-hermitian couplings at time t = {}? (y or n) ".format( len(diff), _np.round(t,5)))
				if user_input == 'y':
					print "   (opstr, coupling(t), indices)"
					for i in xrange(len(diff)):
						print "{}. {}".format(i+1, list(diff)[i])
				raise TypeError("Hamiltonian not hermitian!")

		print "Hermiticity check passed!"

	def _get__str__(self):
		if not self._pcon:
			return tensor_basis._get__str__(self)
		else:
			n_digits = int(_np.ceil(_np.log10(self._Ns)))
			str_list_1 = self._b1._get__str__()
			temp = "\t{0:"+str(n_digits)+"d}.  "
			str_list=[]
			for b1 in str_list_1:
				b1 = b1.replace(".","")
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
			return tensor_basis.get_vec(self,v0,sparse=sparse)
		else:
#			raise NotImplementedError("get_vec not implimented for particle conservation symm.")
			# still needs testing...
			if v0.ndim == 1:
				if v0.shape[0] != self._Ns:
					raise ValueError("v0 has incompatible dimensions with basis")
				v0 = v0.reshape((-1,1))
				if sparse:
					return _conserved_get_vec(self,v0,sparse)
				else:
					return _conserved_get_vec(self,v0,sparse).reshape((-1,))

			elif v0.ndim == 2:
				if v0.shape[0] != self._Ns:
					raise ValueError("v0 has incompatible dimensions with basis")

				return _conserved_get_vec(self,v0,sparse)
			else:
				raise ValueError("excpecting v0 to have ndim at most 2")




	def get_proj(self,dtype):
		if not self._pcon:
			return tensor_basis.get_proj(self,dtype)	
		else:
			raise NotImplementedError("get_proj not implimented for particle conservation symm.")
			# still needs testing...
			return _conserved_get_proj(self,dtype)
				








def _conserved_get_vec(p_basis,v0,sparse):
	v0_mask = _np.zeros_like(v0)
	np_min = p_basis._n.min()
	np_max = p_basis._n.max()
	v_ph = _np.zeros((np_max+1,1),dtype=_np.int64)
	
	v_ph[np_min] = 1
	mask = p_basis._n == np_min
	v0_mask[mask] = v0[mask]

	v0_full = p_basis._b1.get_vec(v0_mask,sparse=sparse)

	if sparse:
		v0_full = _sp.kron(v0_full,v_ph,format="csr")
	else:
		v0_full = _np.kron(v0_full,v_ph)
		
	v_ph[np_min] = 0
	v0_mask[mask] = 0.0

	for np in xrange(np_min+1,np_max+1,1):
		v_ph[np] = 1
		mask = p_basis._n == np
		v0_mask[mask] = v0[mask]
		v0_full_1 = p_basis._b1.get_vec(v0_mask,sparse=sparse)

		if sparse:
			v0_full = v0_full + _sp.kron(v0_full_1,v_ph,format="csr")
			v0_full.sum_duplicates()
			v0_full.eliminate_zeros()
		else:
			v0_full += _np.kron(v0_full_1,v_ph)
		
		v_ph[np] = 0
		v0_mask[mask] = 0.0		



	return v0_full




def _conserved_get_proj(p_basis,dtype):
	np_min = p_basis._n.min()
	np_max = p_basis._n.max()
	v_ph = _np.zeros((np_max+1,1),dtype=_np.int32)

	proj_1 = p_basis._b1.get_proj(dtype).tocsc()
	proj_1_mask = _sp.csc_matrix(proj_1.shape,dtype=dtype)

	v_ph[np_min] = 1
	mask = p_basis._n == np_min
	proj_1_mask[:,mask] = proj_1[:,mask]

	proj_1_full = _sp.kron(proj_1_mask,v_ph,format="csr")

	proj_1_mask[:,mask]=0.0
	proj_1_mask.eliminate_zeros()
	v_ph[np_min] = 0


	for np in xrange(np_min+1,np_max+1,1):
		v_ph[np] = 1
		mask = p_basis._n == np
		proj_1_mask[:,mask] = proj_1[:,mask]

		proj_1_full = proj_1_full + _sp.kron(proj_1_mask,v_ph,format="csr")

		proj_1_mask[:,mask]=0.0
		proj_1_mask.eliminate_zeros()
		v_ph[np] = 0		














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

		row = _np.array(self._basis,dtype=self._dtype)
		col = _np.array(self._basis,dtype=self._dtype)
		ME = _np.ones((self._Ns,),dtype=dtype)

		
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')
		if not _np.can_cast(J,_np.dtype(dtype)):
			raise TypeError("can't cast J to proper dtype")
		

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
		ME[mask] *= 0.0
		
		if J != 1.0: 
			ME *= J

		return ME,row,col		





			
