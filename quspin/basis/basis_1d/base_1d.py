from ..base import basis,MAXPRINT
from ..base import _lattice_partial_trace_pure,_lattice_reshape_pure
from ..base import _lattice_partial_trace_mixed,_lattice_reshape_mixed
from ..base import _lattice_partial_trace_sparse_pure,_lattice_reshape_sparse_pure
from . import _check_1d_symm as _check
import numpy as _np
import scipy.sparse as _sp
import scipy.linalg as _sla
import numpy.linalg as _npla
import scipy.sparse.linalg as _spla
from numpy import array,cos,sin,exp,pi
from numpy.linalg import norm,eigvalsh,svd
from scipy.sparse.linalg import eigsh
import warnings
from types import ModuleType


import warnings


# this is how we encode which fortran function to call when calculating 
# the action of operator string

_dtypes={"f":_np.float32,"d":_np.float64,"F":_np.complex64,"D":_np.complex128}


_basis_op_errors={1:"opstr character not recognized.",
				-1:"attemping to use real hamiltonian with complex matrix elements.",
				-2:"index of operator not between 0 <= index <= L-1"}



class OpstrError(Exception):
	# this class defines an exception which can be raised whenever there is some sort of error which we can
	# see will obviously break the code. 
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message


class bitops:
	def __init__(self,ops_module,**blocks):
		def try_add(func_str,block):
			try:
				self.__dict__[func_str] = ops_module.__dict__[func_str]
			except KeyError:
				if blocks.get(block) is not None:
					raise AttributeError("module {} missing implementation of {}.".format(module.__name__,func))

		try_add("py_fliplr","pblock")
		try_add("py_shift","kblock")
		try_add("py_flip_all","zblock")
		try_add("py_flip_sublat_A","zAblock")
		try_add("py_flip_sublat_B","zBblock")




class basis_1d(basis):
	def __init__(self,basis_module,ops_module,L,Np=None,_Np=None,pars=None,**blocks):

		if self.__class__.__name__ == "basis_1d":
			raise ValueError("This class is not intended"
							 " to be instantiated directly.")

		if type(Np) is int:
			self._check_pcon=True
			self._get_proj_pcon = True
			self._make_Np_block(basis_module,ops_module,L,Np=Np,pars=pars,**blocks)
	
		elif Np is None: # User hasn't specified Np,
			if _Np is not None: # check to see if photon_basis can create the particle sectors.

				if type(_Np) is not int:
					raise ValueError("Np must be integer")

				if _Np == -1: 
					self._make_Np_block(basis_module,ops_module,L,pars=pars,**blocks)
				elif _Np >= 0:
					if _Np+1 > L: _Np = L
					blocks["count_particles"] = True

					zblock = blocks.get("zblock")
					zAblock = blocks.get("zAblock")
					zBblock = blocks.get("zAblock")
				
					if (type(zblock) is int) or (type(zAblock) is int) or (type(zBblock) is int):
						raise ValueError("spin inversion symmetry not compatible with particle conserving photon_basis.")
					
					# loop over the first Np particle sectors (use the iterator initialization).
					Np = list(range(_Np+1))
					self.__init__(L,Np,**blocks)
				else:
					raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")

			else: # if _Np is None then assume user wants to not specify Magnetization sector
				self._check_pcon = False
				self._get_proj_pcon = False
				self._make_Np_block(basis_module,ops_module,L,pars=pars,**blocks)


		else: # try to interate over Np 
			try:
				Nup_iter = iter(Np)
			except TypeError:
				raise TypeError("Np must be integer or iteratable object.")

			blocks["check_z_symm"] = False
			Np = next(Nup_iter)
			self._get_proj_pcon = False
			self._make_Np_block(basis_module,ops_module,L,Np=Np,pars=pars,**blocks)
			for Np in Nup_iter:
				temp_basis =self.__class__(L,Np,**blocks)
				self.append(temp_basis)	

	def _make_Np_block(self,basis_module,ops_module,L,Np=None,pars=None,**blocks):
		# getting arguments which are used in basis.
		kblock=blocks.get("kblock")
		zblock=blocks.get("zblock")
		zAblock=blocks.get("zAblock")
		zBblock=blocks.get("zBblock")
		pblock=blocks.get("pblock")
		pzblock=blocks.get("pzblock")
		a=blocks.get("a")

		count_particles = blocks.get("count_particles")
		if count_particles is None:
			count_particles=False

		check_z_symm = blocks.get("check_z_symm")
		if check_z_symm is None:
			check_z_symm=True


		if type(L) is not int:
			raise TypeError('L must be integer')

		if self.sps < 2:
			raise ValueError("invalid value for sps, sps >= 2.")


		if type(a) is not int:
			raise TypeError('a must be integer')

		# checking if a is compatible with L
		if(L%a != 0):
			raise ValueError('L must be interger multiple of lattice spacing a')

		# checking type, and value of blocks
		if Np is not None:
			if type(Np) is not int: raise TypeError('Nup/Nb/Nf must be integer')
			if Np < 0 or Np > L*(self.sps-1): raise ValueError("0 <= Number of particles <= %d" % (L*(self.sps-1)))

		if pblock is not None:
			if type(pblock) is not int: raise TypeError('pblock must be integer')
			if abs(pblock) != 1: raise ValueError("pblock must be +/- 1")

		if zblock is not None:
			if type(zblock) is not int: raise TypeError('zblock/cblock must be integer')
			if abs(zblock) != 1: raise ValueError("zblock/cblock must be +/- 1")

		if zAblock is not None:
			if type(zAblock) is not int: raise TypeError('zAblock/cAblock must be integer')
			if abs(zAblock) != 1: raise ValueError("zAblock/cAblock must be +/- 1")

		if zBblock is not None:
			if type(zBblock) is not int: raise TypeError('zBblock/cBblock must be integer')
			if abs(zBblock) != 1: raise ValueError("zBblock/cBblock must be +/- 1")

		if pzblock is not None:
			if type(pzblock) is not int: raise TypeError('pzblock/pcblock must be integer')
			if abs(pzblock) != 1: raise ValueError("pzblock/pcblock must be +/- 1")

		if kblock is not None and (a <= L):
			if type(kblock) is not int: raise TypeError('kblock must be integer')
			if a == L:
				warnings.warn("using momentum with L == a",stacklevel=5)
			kblock = kblock % (L//a)
			blocks["kblock"] = kblock
			self._k = 2*(_np.pi)*a*kblock/L

		self._L = L
		self._Ns = basis_module.get_Ns(L,Np,self.sps,**blocks) # estimate how many states in H-space to preallocate memory.
		self._basis_type = basis_module.get_basis_type(L,Np,self.sps,**blocks) # get the size of the integer representation needed for this basis (uint32,uint64,object)
		self._pars = _np.asarray(pars,dtype=self._basis_type)
		self._bitops = bitops(basis_module,**blocks)

		if type(Np) is int:
			self._conserved = "N"
			self._Ns_pcon = basis_module.get_Ns(L,Np,self.sps,**{})
			self._Np = Np
			self._make_n_basis = basis_module.n_basis
		else:
			self._conserved = ""
			self._Ns_pcon = None


		# shout out if pblock and zA/zB blocks defined simultaneously
		if type(pblock) is int and ((type(zAblock) is int) or (type(zBblock) is int)):
			raise ValueError("zA and zB symmetries incompatible with parity symmetry")

		if check_z_symm:
			blocks["Np"] = Np
			# checking if spin inversion is compatible with Np and L
			if (type(Np) is int) and ((type(zblock) is int) or (type(pzblock) is int)):
				if (L*(self.sps-1) % 2) != 0:
					raise ValueError("spin inversion/particle-hole symmetry with particle/magnetization conservation must be used with chains with 0 magnetization sector or at half filling")
				if Np != L*(self.sps-1)//2:
					raise ValueError("spin inversion/particle-hole symmetry only reduces the 0 magnetization or half filled particle sector")

			if (type(Np) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
				raise ValueError("zA/cA and zB/cB symmetries incompatible with magnetisation/particle symmetry")

			# checking if ZA/ZB spin inversion is compatible with unit cell of translation symemtry
			if (type(kblock) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
				if a%2 != 0: # T and ZA (ZB) symemtries do NOT commute
					raise ValueError("unit cell size 'a' must be even")




		self._blocks_1d = blocks
		self._unique_me = True	
		
		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & T & P & Z"
			else: self._conserved = "T & P & Z"

			self._blocks_1d["pzblock"] = pblock*zblock
			self._unique_me = False

			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._op = ops_module.t_p_z_op

			if self._basis_type == _np.object:
				# if object is basis type then most likely this is for single particle stuff in which case the 
				# normalizations need to be large ~ 1000 or more which won't fit in int8/int16.
				self._N=_np.empty(self._basis.shape,dtype=_np.int32) 
				self._M=_np.empty(self._basis.shape,dtype=_np.int32)
			else:
				self._N=_np.empty(self._basis.shape,dtype=_np.int8) # normalisation*sigma
				self._M=_np.empty(self._basis.shape,dtype=_np.int16) # m = mp + (L+1)mz + (L+1)^2c; Anders' paper

			if (type(Np) is int):
				# arguments get overwritten by ops.-_basis 
				self._Ns = basis_module.n_t_p_z_basis(L,Np,pblock,zblock,kblock,a,self._pars,self._N,self._M,self._basis)
			else:
				self._Ns = basis_module.t_p_z_basis(L,pblock,zblock,kblock,a,self._pars,self._N,self._M,self._basis)

			# cut off extra memory for overestimated state number
			self._N.resize((self._Ns,))
			self._M.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(zAblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & T & ZA & ZB"
			else: self._conserved = "T & ZA & ZB"
			self._blocks_1d["zblock"] = zAblock*zBblock


			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zA_zB_op

			if self._basis_type == _np.object:
				self._N=_np.empty(self._basis.shape,dtype=_np.int32) 
				self._M=_np.empty(self._basis.shape,dtype=_np.int32)
			else:
				self._N=_np.empty(self._basis.shape,dtype=_np.int8)
				self._M=_np.empty(self._basis.shape,dtype=_np.int16)

			if (type(Np) is int):
				self._Ns = basis_module.n_t_zA_zB_basis(L,Np,zAblock,zBblock,kblock,a,self._pars,self._N,self._M,self._basis)
			else:
				self._Ns = basis_module.t_zA_zB_basis(L,zAblock,zBblock,kblock,a,self._pars,self._N,self._M,self._basis)

			self._N.resize((self._Ns,))
			self._M.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(pzblock) is int):
			if self._conserved: self._conserved += " & T & PZ"
			else: self._conserved = "T & PZ"
			self._unique_me = False

			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._op = ops_module.t_pz_op

			if self._basis_type == _np.object:
				self._N=_np.empty(self._basis.shape,dtype=_np.int32) 
				self._M=_np.empty(self._basis.shape,dtype=_np.int32)
			else:			
				self._N=_np.empty(self._basis.shape,dtype=_np.int8)
				self._M=_np.empty(self._basis.shape,dtype=_np.int8) #mpz

			if (type(Np) is int):
				self._Ns = basis_module.n_t_pz_basis(L,Np,pzblock,kblock,a,self._pars,self._N,self._M,self._basis)
			else:
				self._Ns = basis_module.t_pz_basis(L,pzblock,kblock,a,self._pars,self._N,self._M,self._basis)

			self._N.resize((self._Ns,))
			self._M.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(pblock) is int):
			if self._conserved: self._conserved += " & T & P"
			else: self._conserved = "T & P"
			self._unique_me = False

			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._op = ops_module.t_p_op

			if self._basis_type == _np.object:
				self._N=_np.empty(self._basis.shape,dtype=_np.int32) 
				self._M=_np.empty(self._basis.shape,dtype=_np.int32)
			else:			
				self._N=_np.empty(self._basis.shape,dtype=_np.int8)
				self._M=_np.empty(self._basis.shape,dtype=_np.int8)

			if (type(Np) is int):
				self._Ns = basis_module.n_t_p_basis(L,Np,pblock,kblock,a,self._pars,self._N,self._M,self._basis)
			else:
				self._Ns = basis_module.t_p_basis(L,pblock,kblock,a,self._pars,self._N,self._M,self._basis)


			self._N.resize((self._Ns,))
			self._M.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & T & Z"
			else: self._conserved = "T & Z"
			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._op = ops_module.t_z_op

			if self._basis_type == _np.object:
				self._N=_np.empty(self._basis.shape,dtype=_np.int32) 
				self._M=_np.empty(self._basis.shape,dtype=_np.int32)
			else:
				self._N=_np.empty(self._basis.shape,dtype=_np.int8)
				self._M=_np.empty(self._basis.shape,dtype=_np.int8)

			if (type(Np) is int):
				self._Ns = basis_module.n_t_z_basis(L,Np,zblock,kblock,a,self._pars,self._N,self._M,self._basis)
			else:
				self._Ns = basis_module.t_z_basis(L,zblock,kblock,a,self._pars,self._N,self._M,self._basis)

			self._N.resize((self._Ns,))
			self._M.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]


		elif (type(kblock) is int) and (type(zAblock) is int):
			if self._conserved: self._conserved += " & T & ZA"
			else: self._conserved = "T & ZA"
			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zA_op

			if self._basis_type == _np.object:
				self._N=_np.empty(self._basis.shape,dtype=_np.int32) 
				self._M=_np.empty(self._basis.shape,dtype=_np.int32)
			else:			
				self._N=_np.empty(self._basis.shape,dtype=_np.int8)
				self._M=_np.empty(self._basis.shape,dtype=_np.int8)

			if (type(Np) is int):
				self._Ns = basis_module.n_t_zA_basis(L,Np,zAblock,kblock,a,self._pars,self._N,self._M,self._basis)
			else:
				self._Ns = basis_module.t_zA_basis(L,zAblock,kblock,a,self._pars,self._N,self._M,self._basis)

			self._N.resize((self._Ns,))
			self._M.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(kblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & T & ZB"
			else: self._conserved = "T & ZB"
			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._op = ops_module.t_zB_op

			if self._basis_type == _np.object:
				self._N=_np.empty(self._basis.shape,dtype=_np.int32) 
				self._M=_np.empty(self._basis.shape,dtype=_np.int32)
			else:			
				self._N=_np.empty(self._basis.shape,dtype=_np.int8)
				self._M=_np.empty(self._basis.shape,dtype=_np.int8)

			if (type(Np) is int):
				self._Ns = basis_module.n_t_zB_basis(L,Np,zBblock,kblock,a,self._pars,self._N,self._M,self._basis)
			else:
				self._Ns = basis_module.t_zB_basis(L,zBblock,kblock,a,self._pars,self._N,self._M,self._basis)

			self._N.resize((self._Ns,))
			self._M.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._M,self._basis,self._L,self._pars]

		elif (type(pblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & P & Z"
			else: self._conserved += "P & Z"
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			self._op = ops_module.p_z_op
			
			if (type(Np) is int):
				self._Ns = basis_module.n_p_z_basis(L,Np,pblock,zblock,self._pars,self._N,self._basis)
			else:
				self._Ns = basis_module.p_z_basis(L,pblock,zblock,self._pars,self._N,self._basis)
			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L,self._pars]


		elif (type(zAblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & ZA & ZB"
			else: self._conserved += "ZA & ZB"

			self._op = ops_module.zA_zB_op

			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			if (type(Np) is int):
				self._Ns = basis_module.n_zA_zB_basis(L,Np,zAblock,zBblock,self._pars,self._N,self._basis)
			else:
				self._Ns = basis_module.zA_zB_basis(L,zAblock,zBblock,self._pars,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L,self._pars]



		elif type(pblock) is int:
			if self._conserved: self._conserved += " & P"
			else: self._conserved = "P"
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			self._op = ops_module.p_op

			if (type(Np) is int):
				self._Ns = basis_module.n_p_basis(L,Np,pblock,self._pars,self._N,self._basis)
			else:
				self._Ns = basis_module.p_basis(L,pblock,self._pars,self._N,self._basis)
				
			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L,self._pars]

		elif type(zblock) is int:
			if self._conserved: self._conserved += " & Z"
			else: self._conserved += "Z"
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			self._op = ops_module.z_op

			if (type(Np) is int):
				self._Ns = basis_module.n_z_basis(L,Np,zblock,self._pars,self._N,self._basis)
			else:
				self._Ns = basis_module.z_basis(L,zblock,self._pars,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L,self._pars]

			

		elif type(zAblock) is int:
			if self._conserved: self._conserved += " & ZA"
			else: self._conserved += "ZA"
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			self._op = ops_module.zA_op

			if (type(Np) is int):
				self._Ns = basis_module.n_zA_basis(L,Np,zAblock,self._pars,self._N,self._basis)
			else:
				self._Ns = basis_module.zA_basis(L,zAblock,self._pars,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L,self._pars]


		elif type(zBblock) is int:
			if self._conserved: self._conserved += " & ZB"
			else: self._conserved += "ZB"
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			self._op = ops_module.zB_op

			if (type(Np) is int):
				self._Ns = basis_module.n_zB_basis(L,Np,zBblock,self._pars,self._N,self._basis)
			else:
				self._Ns = basis_module.zB_basis(L,zBblock,self._pars,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L,self._pars]
				
		elif type(pzblock) is int:
			if self._conserved: self._conserved += " & PZ"
			else: self._conserved += "PZ"
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			self._op = ops_module.pz_op

			if (type(Np) is int):
				self._Ns = basis_module.n_pz_basis(L,Np,pzblock,self._pars,self._N,self._basis)
			else:
				self._Ns = basis_module.pz_basis(L,pzblock,self._pars,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L,self._pars]
	
		elif type(kblock) is int:
			if self._conserved: self._conserved += " & T"
			else: self._conserved = "T"
			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._op = ops_module.t_op
			
			if self._basis_type == _np.object:
				self._N=_np.empty(self._basis.shape,dtype=_np.int32) 
			else:			
				self._N=_np.empty(self._basis.shape,dtype=_np.int8)

			if (type(Np) is int):
				self._Ns = basis_module.n_t_basis(L,Np,kblock,a,self._pars,self._N,self._basis)
			else:
				self._Ns = basis_module.t_basis(L,kblock,a,self._pars,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L,self._pars]

		else: 
			if type(Np) is int:
				self._op = ops_module.n_op
				self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
				basis_module.n_basis(L,Np,self._Ns,self._pars,self._basis)
			else:
				self._op = ops_module.op
				self._basis = _np.arange(0,self._Ns,1,dtype=self._basis_type)
			self._op_args=[self._basis,self._pars]

		if count_particles: self._Np_list = _np.full(self._basis.shape,Np,dtype=_np.int8)

	def append(self,other):
		if not isinstance(other,self.__class__):
			raise TypeError("can only append basis objects of the same type")
		if self._L != other._L:
			raise ValueError("appending incompatible system sizes with")
		if self._blocks_1d != other._blocks_1d:
			raise ValueError("appending incompatible blocks")
		

		Ns = self._Ns + other._Ns

		if self._conserved == "" or self._conserved == "N":
			self._op_args=[self._pars]
		else:
			self._op_args=[self._L,self._pars]


		self._basis.resize((Ns,),refcheck=False)
		self._basis[self._Ns:] = other._basis[:]
		arg = _np.argsort(self._basis)
		self._basis = self._basis[arg].copy()

		self._op_args.insert(0,self._basis)

		if hasattr(self,"_Np_list"):
			self._Np_list.resize((Ns,),refcheck=False)
			self._Np_list[self._Ns:] = other._Np_list[:]
			self._Np_list = self._Np_list[arg].copy()

		if hasattr(self,"_M"):
			self._M.resize((Ns,),refcheck=False)
			self._M[self._Ns:] = other._M[:]
			self._M = self._M[arg].copy()
			self._op_args.insert(0,self._M)	

		if hasattr(self,"_N"):
			self._N.resize((Ns,),refcheck=False)
			self._N[self._Ns:] = other._N[:]
			self._N = self._N[arg].copy()
			self._op_args.insert(0,self._N)

		self._Ns = Ns

	@property
	def blocks(self):
		return self._blocks

	@property
	def L(self):
		return self._L

	@property
	def N(self):
		return self._L

	@property
	def sps(self):
		return self._sps

	@property
	def conserved(self):
		return self._conserved

	@property
	def description(self):
		blocks = ""
		lat_space = "lattice spacing: a = {a}".format(**self._blocks)

		for symm in self._blocks:
			if symm != "a":
				blocks += symm+" = {"+symm+"}, "

		blocks = blocks.format(**self._blocks)

		if len(self.conserved) == 0:
			symm = "no symmetry"
		elif len(self.conserved) == 1:
			symm = "symmetry"
		else:
			symm = "symmetries"

		string = """1d basis for chain of L = {0} containing {5} states \n\t{1}: {2} \n\tquantum numbers: {4} \n\t{3} \n\n""".format(self._L,symm,self._conserved,lat_space,blocks,self._Ns)
		string += self.operators
		return string 

	def __getitem__(self,key):
		return self._basis.__getitem__(key)

	def index(self,s):
		if type(s) is int:
			pass
		elif type(s) is str:
			s = long(s[::-1],self.sps)
		else:
			raise ValueError("s must be integer or state")

		indx = _np.argwhere(self._basis == s)

		if len(indx) != 0:
			return _np.squeeze(indx)
		else:
			raise ValueError("s must be representive state in basis. ")

	def __iter__(self):
		return self._basis.__iter__()

	def Op(self,opstr,indx,J,dtype):
		indx = _np.asarray(indx,dtype=_np.int32)

		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')

		if _np.any(indx >= self._L) or _np.any(indx < 0):
			raise ValueError('values in indx falls outside of system')

		extra_ops = set(opstr) - self._allowed_ops
		if extra_ops:
			raise ValueError("unrecognized charactors {} in operator string.".format(extra_ops))


		if self._Ns <= 0:
			return [],[],[]

		if self._unique_me:
			N_op = self.Ns
		else:
			N_op = 2*self.Ns

		col = _np.zeros(N_op,dtype=self._basis_type)
		row = _np.zeros(N_op,dtype=self._basis_type)
		ME = _np.zeros(N_op,dtype=dtype)

		error = self._op(row,col,ME,opstr,indx,J,*self._op_args,**self._blocks_1d)

		if error != 0: raise OpstrError(_basis_op_errors[error])

		mask = _np.logical_not(_np.isnan(ME))
		col = col[mask]
		row = row[mask]
		ME = ME[mask]

		return ME,row,col		

	def get_norms(self,dtype):
		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")

		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			c = _np.empty(self._M.shape,dtype=_np.int8)
			nn = array(c)
			mm = array(c)
			_np.floor_divide(self._M,(self._L+1)**2,out=c)
			_np.floor_divide(self._M,self._L+1,out=nn)
			_np.mod(nn,self._L+1,out=nn)
			_np.mod(self._M,self._L+1,out=mm)
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,4*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm *= _np.sign(self._N)
			norm /= self._N
			# c = 2
			mask = (c == 2)
			norm[mask] *= (1.0 + _np.sign(self._N[mask])*pblock*_np.cos(self._k*mm[mask]))
			# c = 3
			mask = (c == 3)
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*nn[mask]))	
			# c = 4
			mask = (c == 4)
			norm[mask] *= (1.0 + _np.sign(self._N[mask])*pzblock*_np.cos(self._k*mm[mask]))	
			# c = 5
			mask = (c == 5)
			norm[mask] *= (1.0 + _np.sign(self._N[mask])*pblock*_np.cos(self._k*mm[mask]))
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*nn[mask]))	
			del mask
		elif (type(kblock) is int) and (type(zAblock) is int) and (type(zBblock) is int):
			c = _np.empty(self._M.shape,dtype=_np.int8)
			mm = array(c)
			_np.floor_divide(self._M,(self._L+1),c)
			_np.mod(self._M,self._L+1,mm)
			norm = _np.full(self._basis.shape,4*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			# c = 2
			mask = (c == 2)
			norm[mask] *= (1.0 + zAblock*_np.cos(self._k*mm[mask]))
			# c = 3
			mask = (c == 3)
			norm[mask] *= (1.0 + zBblock*_np.cos(self._k*mm[mask]))	
			# c = 4
			mask = (c == 4)
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*mm[mask]))	
			del mask
		elif (type(kblock) is int) and (type(pblock) is int):
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm *= _np.sign(self._N)
			norm /= self._N
			# m >= 0 
			mask = (self._M >= 0)
			norm[mask] *= (1.0 + _np.sign(self._N[mask])*pblock*_np.cos(self._k*self._M[mask]))
			del mask
		elif (type(kblock) is int) and (type(pzblock) is int):
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm *= _np.sign(self._N)
			norm /= self._N
			# m >= 0 
			mask = (self._M >= 0)
			norm[mask] *= (1.0 + _np.sign(self._N[mask])*pzblock*_np.cos(self._k*self._M[mask]))
			del mask
		elif (type(kblock) is int) and (type(zblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			# m >= 0 
			mask = (self._M >= 0)
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*self._M[mask]))
			del mask
		elif (type(kblock) is int) and (type(zAblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			# m >= 0 
			mask = (self._M >= 0)
			norm[mask] *= (1.0 + zAblock*_np.cos(self._k*self._M[mask]))
			del mask
		elif (type(kblock) is int) and (type(zBblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			# m >= 0 
			mask = (self._M >= 0)
			norm[mask] *= (1.0 + zBblock*_np.cos(self._k*self._M[mask]))
			del mask
		elif (type(pblock) is int) and (type(zblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zAblock) is int) and (type(zBblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(pblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(pzblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zAblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(zBblock) is int):
			norm = array(self._N,dtype=dtype)
		elif (type(kblock) is int):
			norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm /= self._N
		else:
			norm = _np.ones(self._basis.shape,dtype=dtype)

		_np.sqrt(norm,norm)

		return norm

	def get_vec(self,v0,sparse=True):

		if not hasattr(v0,"shape"):
			v0 = _np.asanyarray(v0)

		squeeze = False
		
		if v0.ndim == 1:
			shape = (self._sps**self._L,1)
			v0 = v0.reshape((-1,1))
			squeeze = True
		elif v0.ndim == 2:
			shape = (self._sps**self._L,v0.shape[1])
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")

		if self._Ns <= 0:
			if sparse:
				return _sp.csr_matrix(([],([],[])),shape=(0,0))
			else:
				return _np.zeros((0,0),dtype=v0.dtype)

		if v0.shape[0] != self._Ns:
			raise ValueError("v0 shape {0} not compatible with Ns={1}".format(v0.shape,self._Ns))


		if _sp.issparse(v0): # current work around for sparse states.
			return self.get_proj(v0.dtype).dot(v0)

		norms = self.get_norms(v0.dtype)

		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")


		if (type(kblock) is int) and ((type(pblock) is int) or (type(pzblock) is int)):
			mask = (self._N < 0)
			ind_neg, = _np.nonzero(mask)
			mask = (self._N > 0)
			ind_pos, = _np.nonzero(mask)
			del mask
			def C(r,k,c,norms,dtype,ind_neg,ind_pos):
				c[ind_pos] = cos(dtype(k*r))
				c[ind_neg] = -sin(dtype(k*r))
				_np.true_divide(c,norms,c)
		else:
			ind_pos = _np.fromiter(range(v0.shape[0]),count=v0.shape[0],dtype=_np.int32)
			ind_neg = array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == _np.pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(dtype(1.0j*k*r))
				_np.true_divide(c,norms,c)

		if sparse:
			return _get_vec_sparse(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d)
		else:
			if squeeze:
				return  _np.squeeze(_get_vec_dense(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d))
			else:
				return _get_vec_dense(self._bitops,self._pars,v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks_1d)

	def get_proj(self,dtype,pcon=False):
		norms = self.get_norms(dtype)

		a = self._blocks_1d.get("a")
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		pzblock = self._blocks_1d.get("pzblock")

		

		if pcon and self._get_proj_pcon:
			basis_pcon = _np.ones(self._Ns_pcon,dtype=self._basis_type)
			self._make_n_basis(self.L,self._Np,self._Ns_pcon,self._pars,basis_pcon)
			shape = (self._Ns_pcon,self._Ns)
		elif pcon and not self._get_proj_pcon:
			raise TypeError("pcon=True only works for basis of a single particle number sector.")
		else:
			shape = (self.sps**self.L,self._Ns)
			basis_pcon = None

		if self._Ns <= 0:
			return _sp.csr_matrix(([],([],[])),shape=shape)


		if (type(kblock) is int) and ((type(pblock) is int) or (type(pzblock) is int)):
			mask = (self._N < 0)
			ind_neg, = _np.nonzero(mask)
			mask = (self._N > 0)
			ind_pos, = _np.nonzero(mask)
			del mask
			def C(r,k,c,norms,dtype,ind_neg,ind_pos):
				c[ind_pos] = cos(dtype(k*r))
				c[ind_neg] = -sin(dtype(k*r))
				_np.true_divide(c,norms,c)
		else:
			if (type(kblock) is int):
				if ((2*kblock*a) % self._L != 0) and not _np.iscomplexobj(dtype(1.0)):
					raise TypeError("symmetries give complex vector, requested dtype is not complex")

			ind_pos = _np.arange(0,self._Ns,1)
			ind_neg = array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == _np.pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(dtype(1.0j*k*r))
				_np.true_divide(c,norms,c)





		return _get_proj_sparse(self._bitops,self._pars,self._basis,basis_pcon,norms,ind_neg,ind_pos,dtype,shape,C,self._L,**self._blocks_1d)

	def partial_trace(self,state,sub_sys_A=None,return_rdm="A",enforce_pure=False,sparse=False):
		if sub_sys_A is None:
			sub_sys_A = tuple(range(self.L//2))
		elif len(sub_sys_A)==self.L:
			raise ValueError("Size of subsystem must be strictly smaller than total system size L!")


		L_A = len(sub_sys_A)
		L_B = self.L - L_A

		if sub_sys_A is None:
			sub_sys_A = tuple(range(self.L//2))

		sub_sys_A = tuple(sub_sys_A)

		if any(not _np.issubdtype(type(s),_np.integer) for s in sub_sys_A):
			raise ValueError("sub_sys_A must iterable of integers with values in {0,...,L-1}!")

		if any(s < 0 or s > self.L for s in sub_sys_A):
			raise ValueError("sub_sys_A must iterable of integers with values in {0,...,L-1}")

		if return_rdm not in set(["A","B","both"]):
			raise ValueError("return_rdm must be: 'A','B','both' or None")

		sps = self.sps
		L = self.L

		if not hasattr(state,"shape"):
			state = _np.asanyarray(state)
			state = state.squeeze() # avoids artificial higher-dim reps of ndarray


		if state.shape[0] != self.Ns:
			raise ValueError("state shape {0} not compatible with Ns={1}".format(state.shape,self._Ns))

		if _sp.issparse(state) or sparse:
			state=self.get_vec(state,sparse=True).T
			
			if state.shape[0] == 1:
				# sparse_pure partial trace
				rdm_A,rdm_B = _lattice_partial_trace_sparse_pure(state,sub_sys_A,L,sps,return_rdm=return_rdm)
			else:
				if state.shape[0]!=state.shape[1] or enforce_pure:
					# vectorize sparse_pure partial trace 
					state = state.tocsr()
					try:
						state_gen = (_lattice_partial_trace_sparse_pure(state.getrow(i),sub_sys_A,L,sps,return_rdm=return_rdm) for i in xrange(state.shape[0]))
					except NameError:
						state_gen = (_lattice_partial_trace_sparse_pure(state.getrow(i),sub_sys_A,L,sps,return_rdm=return_rdm) for i in range(state.shape[0]))

					left,right = zip(*state_gen)

					rdm_A,rdm_B = _np.stack(left),_np.stack(right)

					if any(rdm is None for rdm in rdm_A):
						rdm_A = None

					if any(rdm is None for rdm in rdm_B):
						rdm_B = None
				else: 
					raise ValueError("Expecting a dense array for mixed states.")

		else:
			if state.ndim==1:
				# calculate full H-space representation of state
				state=self.get_vec(state,sparse=False)
				rdm_A,rdm_B = _lattice_partial_trace_pure(state.T,sub_sys_A,L,sps,return_rdm=return_rdm)

			elif state.ndim==2: 
				if state.shape[0]!=state.shape[1] or enforce_pure:
					# calculate full H-space representation of state
					state=self.get_vec(state,sparse=False)
					rdm_A,rdm_B = _lattice_partial_trace_pure(state.T,sub_sys_A,L,sps,return_rdm=return_rdm)

				else: 
					proj = self.get_proj(_dtypes[state.dtype.char])
					proj_state = proj*state*proj.H

					shape0 = proj_state.shape
					proj_state = proj_state.reshape((1,)+shape0)					

					rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,L,sps,return_rdm=return_rdm)

			elif state.ndim==3: #3D DM 
				proj = self.get_proj(_dtypes[state.dtype.char])
				state = state.transpose((2,0,1))
				
				Ns_full = proj.shape[0]
				n_states = state.shape[0]
				
				gen = (proj*s*proj.H for s in state[:])

				proj_state = _np.zeros((n_states,Ns_full,Ns_full),dtype=_dtypes[state.dtype.char])
				
				for i,s in enumerate(gen):
					proj_state[i,...] += s[...]

				rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,L,sps,return_rdm=return_rdm)
			else:
				raise ValueError("state must have ndim < 4")

		if return_rdm == "A":
			return rdm_A
		elif return_rdm == "B":
			return rdm_B
		else:
			return rdm_A,rdm_B

	def _p_pure(self,state,sub_sys_A,return_rdm=None):
		
		# calculate full H-space representation of state
		state=self.get_vec(state,sparse=False)
		# put states in rows
		state=state.T
		# reshape state according to sub_sys_A
		v=_lattice_reshape_pure(state,sub_sys_A,self._L,self._sps)
		
		rdm_A=None
		rdm_B=None

		# perform SVD	
		if return_rdm is None:
			lmbda = svd(v, compute_uv=False) 
		else:
			U, lmbda, V = svd(v, full_matrices=False)
			if return_rdm=='A':
				rdm_A = _np.einsum('...ij,...j,...kj->...ik',U,lmbda**2,U.conj() )
			elif return_rdm=='B':
				rdm_B = _np.einsum('...ji,...j,...jk->...ik',V.conj(),lmbda**2,V )
			elif return_rdm=='both':
				rdm_A = _np.einsum('...ij,...j,...kj->...ik',U,lmbda**2,U.conj() )
				rdm_B = _np.einsum('...ji,...j,...jk->...ik',V.conj(),lmbda**2,V )


		return lmbda**2 + _np.finfo(lmbda.dtype).eps, rdm_A, rdm_B

	def _p_pure_sparse(self,state,sub_sys_A,return_rdm=None,sparse_diag=True,maxiter=None):

		"""
		# THE FOLLOWING LINES HAVE BEEN DEPRECATED

		if svds: # patchy sparse svd

			# calculate full H-space representation of state
			state=self.get_vec(state.T,sparse=True).T
			# reshape state according to sub_sys_A
			v=_lattice_reshape_sparse_pure(state,sub_sys_A,self._L,self._sps)
			#print(v.todense())
			n=min(v.shape)

			# perform SVD	
			if return_rdm is None:
				lmbda_SM = _spla.svds(v, k=n//2, which='SM',return_singular_vectors=False)
				lmbda_LM = _spla.svds(v, k=n//2+n%2, which='LM',return_singular_vectors=False)
				#_, lmbda_dense, _ = _npla.svd(v.todense(),full_matrices=False)
				# concatenate lower and upper part
				lmbda=_np.concatenate((lmbda_LM,lmbda_SM),axis=0)
				lmbda.sort()
				return lmbda[::-1]**2 + _np.finfo(lmbda.dtype).eps
			else:
				
				if return_rdm=='A':
					U_SM, lmbda_SM, V_SM = _spla.svds(v, k=n//2, which='SM',return_singular_vectors='u')
					U_LM, lmbda_LM, V_LM = _spla.svds(v, k=n//2+n%2, which='LM',return_singular_vectors='u')
					#ua,lmbdas,va = _npla.svd(v.todense())

					# concatenate lower and upper part
					lmbda=_np.concatenate((lmbda_LM,lmbda_SM),axis=0)
					arg = _np.argsort(lmbda)
					lmbda = lmbda[arg]**2

					U=_np.concatenate((U_LM,U_SM),axis=1)
					U=U[...,arg]
					#V=_np.concatenate((V_LM,V_SM[...,::-1,:]),axis=0)

					# check and orthogonalise VF in degenerate subspaces
					if _np.any( _np.diff(lmbda) < 1E3*_np.finfo(lmbda.dtype).eps):
						U,_ = _sla.qr(U, overwrite_a=True)
						#V,_ = _sla.qr(V.T, overwrite_a=True)
						#V = V.T
	
					# calculate reduced DM
					rdm_A = _np.einsum('...ij,...j,...kj->...ik',U,lmbda,U.conj() )

					return lmbda[::-1] + _np.finfo(lmbda.dtype).eps, rdm_A

				elif return_rdm=='B':
					U_SM, lmbda_SM, V_SM = _spla.svds(v, k=n//2, which='SM',return_singular_vectors='vh')
					U_LM, lmbda_LM, V_LM = _spla.svds(v, k=n//2+n%2, which='LM',return_singular_vectors='vh')
					#ua,lmbdas,va = _npla.svd(v.todense())

					# concatenate lower and upper part
					lmbda=_np.concatenate((lmbda_LM,lmbda_SM),axis=0)
					#U=_np.concatenate((U_LM,U_SM[...,::-1]),axis=1)
					V=_np.concatenate((V_LM,V_SM),axis=0)

					arg = _np.argsort(lmbda)
					lmbda = lmbda[arg]**2
					V = V[...,arg,:]

					

					# check and orthogonalise VF in degenerate subspaces
					if _np.any( _np.diff(lmbda) < 1E3*_np.finfo(lmbda.dtype).eps):
						#U,_ = _sla.qr(U, overwrite_a=True)
						V,_ = _sla.qr(V.T, overwrite_a=True)
						V = V.T
					
					# calculate reduced DM
					rdm_B = _np.einsum('...ji,...j,...jk->...ik',V.conj(),lmbda,V )

					return lmbda[::-1] + _np.finfo(lmbda.dtype).eps, rdm_B

				elif return_rdm=='both':
					U_SM, lmbda_SM, V_SM = _spla.svds(v, k=n//2, which='SM',return_singular_vectors=True)
					U_LM, lmbda_LM, V_LM = _spla.svds(v, k=n//2+n%2, which='LM',return_singular_vectors=True)

					# concatenate lower and upper part
					lmbda=_np.concatenate((lmbda_LM,lmbda_SM),axis=0)
					U=_np.concatenate((U_LM,U_SM),axis=1)
					V=_np.concatenate((V_LM,V_SM),axis=0)
					arg = _np.argsort(lmbda)
					lmbda = lmbda[arg]**2
					V = V[...,arg,:]
					U = U[...,arg]
					# check and orthogonalise VF in degenerate subspaces
					if _np.any( _np.diff(lmbda) < 1E3*_np.finfo(lmbda.dtype).eps):
						U,_ = _sla.qr(U, overwrite_a=True)
						V,_ = _sla.qr(V.T, overwrite_a=True)
						V = V.T
					# calculate reduced DM
					rdm_A = _np.einsum('...ij,...j,...kj->...ik',U,lmbda,U.conj() )
					rdm_B = _np.einsum('...ji,...j,...jk->...ik',V.conj(),lmbda,V )

					return lmbda[::-1] + _np.finfo(lmbda.dtype).eps, rdm_A, rdm_B
		"""

		partial_trace_args = dict(sub_sys_A=sub_sys_A,sparse=True,enforce_pure=True)

		L_A=len(sub_sys_A)
		L_B=self.L-L_A

		rdm_A=None
		rdm_B=None

		if return_rdm is None:
			if L_A <= L_B:
				partial_trace_args["return_rdm"] = "A"
				rdm = self.partial_trace(state,**partial_trace_args)
			else:
				partial_trace_args["return_rdm"] = "B"
				rdm = self.partial_trace(state,**partial_trace_args)

		elif return_rdm=='A' and L_A <= L_B:
			partial_trace_args["return_rdm"] = "A"
			rdm_A = self.partial_trace(state,**partial_trace_args)
			rdm = rdm_A

		elif return_rdm=='B' and L_B <= L_A:
			partial_trace_args["return_rdm"] = "B"
			rdm_B = self.partial_trace(state,**partial_trace_args)
			rdm = rdm_B

		else:
			partial_trace_args["return_rdm"] = "both"
			rdm_A,rdm_B = self.partial_trace(state,**partial_trace_args)

			if L_A < L_B:
				rdm = rdm_A
			else:
				rdm = rdm_B

		if sparse_diag and rdm.shape[0] > 16:

			def get_p_patchy(rdm):
				n = rdm.shape[0]
				p_LM = eigsh(rdm,k=n//2+n%2,which="LM",maxiter=maxiter,return_eigenvectors=False) # get upper half
				p_SM = eigsh(rdm,k=n//2,which="SM",maxiter=maxiter,return_eigenvectors=False) # get lower half
				p = _np.concatenate((p_LM[::-1],p_SM)) + _np.finfo(p_LM.dtype).eps
				return p

			if _sp.issparse(rdm):
				p = get_p_patchy(rdm)
			else:
				p_gen = (get_p_patchy(dm) for dm in rdm[:])
				p = _np.stack(p_gen)

		else:
			if _sp.issparse(rdm):
				p = eigvalsh(rdm.todense())[::-1] + _np.finfo(rdm.dtype).eps
			else:
				p_gen = (eigvalsh(dm.todense())[::-1] + _np.finfo(dm.dtype).eps for dm in rdm[:])
				p = _np.stack(p_gen)

		return p,rdm_A,rdm_B
	
	def _p_mixed(self,state,sub_sys_A,return_rdm=None):
		"""
		This function calculates the eigenvalues of the reduced density matrix.
		It will first calculate the partial trace of the full density matrix and
		then diagonalizes it to get the eigenvalues. It will automatically choose
		the subsystem with the smaller hilbert space to do the diagonalization in order
		to reduce the calculation time but will only return the desired reduced density
		matrix. 
		"""
		L = self.L
		sps = self.sps

		L_A = len(sub_sys_A)
		L_B = L - L_A

		proj = self.get_proj(_dtypes[state.dtype.char])
		state = state.transpose((2,0,1))

		Ns_full = proj.shape[0]
		n_states = state.shape[0]
		
		gen = (proj*s*proj.H for s in state[:])

		proj_state = _np.zeros((n_states,Ns_full,Ns_full),dtype=_dtypes[state.dtype.char])
		
		for i,s in enumerate(gen):
			proj_state[i,...] += s[...]	

		rdm_A,p_A=None,None
		rdm_B,p_B=None,None
		
		if return_rdm=='both':
			rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,L,sps,return_rdm="both")
			
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

		elif return_rdm=='A':
			rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,L,sps,return_rdm="A")
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			
		elif return_rdm=='B':
			rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,L,sps,return_rdm="B")
			p_B = eigvalsh(rdm_B) + _np.finfo(rdm_B.dtype).eps

		else:
			rdm_A,rdm_B = _lattice_partial_trace_mixed(proj_state,sub_sys_A,L,sps,return_rdm="A")
			p_A = eigvalsh(rdm_A) + _np.finfo(rdm_A.dtype).eps
			
			
		return p_A, p_B, rdm_A, rdm_B

	def ent_entropy(self,state,sub_sys_A=None,densities=True,subsys_ordering=True,return_rdm=None,enforce_pure=False,return_rdm_EVs=False,sparse=False,alpha=1.0,sparse_diag=True,maxiter=None):
		"""
		This function calculates the entanglement entropy of subsystem A and the corresponding reduced 
		density matrix.

		RETURNS: dictionary with keys:

		'Sent': entanglement entropy of subystem A.
		'Sent_B': (optional) entanglement entropy of subystem B.
		'rdm_A': (optional) reduced density matrix of subsystem A
		'rdm_B': (optional) reduced density matrix of subsystem B
		'p_A': (optional) eigenvalues of reduced density matrix of subsystem A
		'p_B': (optional) eigenvalues of reduced density matrix of subsystem B

		--- arguments ---

		state: (required) the state of the quantum system. Can be a:

				-- pure state (default) [numpy array of shape (Ns,)].

				-- density matrix [numpy array of shape (Ns,Ns)].

				-- collection of states containing the states in the columns of state

		sub_sys_A: (optional) tuple or list to define the sites contained in subsystem A 
						[by python convention the first site of the chain is labelled j=0]. 
						Default is tuple(range(L//2)).

		densities: (optional) if set to 'True', the entanglement _entropy is normalised by the size of the
					subsystem [i.e., by the length of 'sub_sys_A']. Detault is 'True'.


		subsys_ordering: (optional) if set to 'True', 'sub_sys_A' is being ordered. Default is 'True'.


		return_rdm: (optional) flag to return the reduced density matrix. Default is 'None'.

				-- 'A': str, returns reduced DM of subsystem A

				-- 'B': str, returns reduced DM of subsystem B

				-- 'both': str, returns reduced DM of both subsystems A and B

		return_rdm_EVs: (optional) boolean to return eigenvalues of reduced DM. If `return_rdm` is specified,
						the eigenvalues of the corresponding DM are returned. If `return_rdm` is NOT specified, 
						the spectrum of `rdm_A` is terurned. Default is `False`.

		enforce_pure: (optional) boolean to determine if 'state' is a collection of pure states or
						a density matrix

		sparse: (optional) flag to enable usage of sparse linear algebra algorithms.

		alpha: (optional) Renyi alpha parameter. Default is '1.0'.

		"""
		if sub_sys_A is None:
			sub_sys_A = list(range(self.L//2))
		else:
			sub_sys_A = list(sub_sys_A)
	
		if len(sub_sys_A)>=self.L:
			raise ValueError("Size of subsystem must be strictly smaller than total system size L!")

		L_A = len(sub_sys_A)
		L_B = self.L - L_A

		if any(not _np.issubdtype(type(s),_np.integer) for s in sub_sys_A):
			raise ValueError("sub_sys_A must iterable of integers with values in {0,...,L-1}!")

		if any(s < 0 or s > self.L for s in sub_sys_A):
			raise ValueError("sub_sys_A must iterable of integers with values in {0,...,L-1}")

		if return_rdm not in set(["A","B","both",None]):
			raise ValueError("return_rdm must be: 'A','B','both' or None")

		if subsys_ordering:
			sub_sys_A = sorted(sub_sys_A)

		sps = self.sps
		L = self.L

		if not hasattr(state,"shape"):
			state = _np.asanyarray(state)
			state = state.squeeze() # avoids artificial higher-dim reps of ndarray


		if state.shape[0] != self.Ns:
			raise ValueError("state shape {0} not compatible with Ns={1}".format(state.shape,self._Ns))

		

		pure=True # set pure state parameter to True
		if _sp.issparse(state) or sparse:
			if state.ndim == 1:
				state = state.reshape((-1,1))

			sparse=True # set sparse flag to True
			if state.shape[1] == 1:
				p, rdm_A, rdm_B = self._p_pure_sparse(state,sub_sys_A,return_rdm=return_rdm,sparse_diag=sparse_diag,maxiter=maxiter)
			else:
				if state.shape[0]!=state.shape[1] or enforce_pure:
					p, rdm_A, rdm_B = self._p_pure_sparse(state,sub_sys_A,return_rdm=return_rdm)
				else: 
					raise ValueError("Expecting a dense array for mixed states.")
					
		else:
			if state.ndim==1:
				state = state.reshape((-1,1))
				p, rdm_A, rdm_B = self._p_pure(state,sub_sys_A,return_rdm=return_rdm)
			
			elif state.ndim==2: 
				if state.shape[0]!=state.shape[1] or enforce_pure:
					p, rdm_A, rdm_B = self._p_pure(state,sub_sys_A,return_rdm=return_rdm)
				else: # 2D mixed
					pure=False
					"""
					# check if DM's are positive definite
					try:
						_np.linalg.cholesky(state)
					except:
						raise ValueError("LinAlgError: (collection of) DM(s) not positive definite")
					# check oif trace of DM is unity
					if _np.any( abs(_np.trace(state) - 1.0 > 1E3*_np.finfo(state.dtype).eps)  ):
						raise ValueError("Expecting eigenvalues of DM to sum to unity!")
					"""
					shape0 = state.shape
					state = state.reshape(shape0+(1,))
					p_A, p_B, rdm_A, rdm_B = self._p_mixed(state,sub_sys_A,return_rdm=return_rdm)
				
			elif state.ndim==3: #3D DM 
				pure=False

				"""
				# check if DM's are positive definite
				try:
					_np.linalg.cholesky(state)
				except:
					raise ValueError("LinAlgError: (collection of) DM(s) not positive definite")

				# check oif trace of DM is unity
				if _np.any( abs(_np.trace(state, axis1=1,axis2=2) - 1.0 > 1E3*_np.finfo(state.dtype).eps)  ):
					raise ValueError("Expecting eigenvalues of DM to sum to unity!")
				"""
				p_A, p_B, rdm_A, rdm_B = self._p_mixed(state,sub_sys_A,return_rdm=return_rdm)

			else:
				raise ValueError("state must have ndim < 4")

		

		if pure:
			p_A, p_B = p, p


		Sent, Sent_B = None, None
		if alpha == 1.0:
			if p_A is not None:
				Sent = - _np.nansum((p_A * _np.log(p_A)),axis=-1)
				if densities: Sent /= L_A
			if p_B is not None:
				Sent_B = - _np.nansum((p_B * _np.log(p_B)),axis=-1)
				if densities: Sent_B /= L_B
		elif alpha >= 0.0:
			if p_A is not None:
				Sent = _np.log(_np.nansum(_np.power(p_A,alpha),axis=-1))/(1.0-alpha)
				if densities: Sent /= L_A
			if p_B is not None:
				Sent_B = _np.log(_np.nansum(_np.power(p_B,alpha),axis=-1))/(1.0-alpha)
				if densities: Sent_B /= L_B
		else:
			raise ValueError("alpha >= 0")
			


		# initiate variables
		variables = ["Sent"]
		if return_rdm_EVs:
			variables.append("p_A")

		if return_rdm == "A":
			variables.append("rdm_A")
			
		elif return_rdm == "B":
			variables.extend(["Sent_B","rdm_B"])
			if return_rdm_EVs:
				variables.append("p_B")
			
		elif return_rdm == "both":
			variables.extend(["rdm_A","Sent_B","rdm_B"])
			if return_rdm_EVs:
				variables.extend(["p_A","p_B"])
	
		# store variables to dictionar
		return_dict = {}
		for i in variables:
			if locals()[i] is not None:
				if sparse and 'rdm' in i:
					return_dict[i] = locals()[i] # don't squeeze sparse matrix
				else:
					return_dict[i] = _np.squeeze( locals()[i] )

		return return_dict

	def _check_symm(self,static,dynamic,basis=None):
		kblock = self._blocks_1d.get("kblock")
		pblock = self._blocks_1d.get("pblock")
		zblock = self._blocks_1d.get("zblock")
		pzblock = self._blocks_1d.get("pzblock")
		zAblock = self._blocks_1d.get("zAblock")
		zBblock = self._blocks_1d.get("zBblock")
		a = self._blocks_1d.get("a")
		L = self.L

		if basis is None:
			basis = self
		
		basis_sort_opstr = basis._sort_opstr
		static_list,dynamic_list = basis.get_local_lists(static,dynamic)

		static_blocks = {}
		dynamic_blocks = {}
		if kblock is not None:
			missingops = _check.check_T(basis_sort_opstr,static_list,L,a)
			if missingops:	static_blocks["T symm"] = (tuple(missingops),)

			missingops = _check.check_T(basis_sort_opstr,dynamic_list,L,a)
			if missingops:	dynamic_blocks["T symm"] = (tuple(missingops),)

		if pblock is not None:
			missingops = _check.check_P(basis_sort_opstr,static_list,L)
			if missingops:	static_blocks["P symm"] = (tuple(missingops),)

			missingops = _check.check_P(basis_sort_opstr,dynamic_list,L)
			if missingops:	dynamic_blocks["P symm"] = (tuple(missingops),)

		if zblock is not None:
			oddops,missingops = _check.check_Z(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["Z/C symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_Z(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["Z/C symm"] = (tuple(oddops),tuple(missingops))

		if zAblock is not None:
			oddops,missingops = _check.check_ZA(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["ZA/CA symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_ZA(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["ZA/CA symm"] = (tuple(oddops),tuple(missingops))

		if zBblock is not None:
			oddops,missingops = _check.check_ZB(basis_sort_opstr,static_list)
			if missingops or oddops:
				static_blocks["ZB/CB symm"] = (tuple(oddops),tuple(missingops))

			oddops,missingops = _check.check_ZB(basis_sort_opstr,dynamic_list)
			if missingops or oddops:
				dynamic_blocks["ZB/CB symm"] = (tuple(oddops),tuple(missingops))

		if pzblock is not None:
			missingops = _check.check_PZ(basis_sort_opstr,static_list,L)
			if missingops:	static_blocks["PZ/PC symm"] = (tuple(missingops),)

			missingops = _check.check_PZ(basis_sort_opstr,dynamic_list,L)
			if missingops:	dynamic_blocks["PZ/PC symm"] = (tuple(missingops),)

		return static_blocks,dynamic_blocks


def _get_vec_dense(ops,pars,v0,basis_in,norms,ind_neg,ind_pos,shape,C,L,**blocks):
	dtype=_dtypes[v0.dtype.char]

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	zAblock = blocks.get("zAblock")
	zBblock = blocks.get("zBblock")
	pzblock = blocks.get("pzblock")


	c = _np.zeros(basis_in.shape,dtype=v0.dtype)	
	v = _np.zeros(shape,dtype=v0.dtype)

	bits=" ".join(["{"+str(i)+":0"+str(L)+"b}" for i in range(len(basis_in))])

	if type(kblock) is int:
		k = 2*_np.pi*kblock*a/L
	else:
		k = 0.0
		a = L

	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)	
		vc = (v0.T*c).T
		v[basis_in[ind_pos]] += vc[ind_pos]
		v[basis_in[ind_neg]] += vc[ind_neg]

		if type(zAblock) is int:
			ops.py_flip_sublat_A(basis_in,L,pars)
			v[basis_in[ind_pos]] += vc[ind_pos]*zAblock
			v[basis_in[ind_neg]] += vc[ind_neg]*zAblock
			ops.py_flip_sublat_A(basis_in,L,pars)
		
		if type(zBblock) is int:
			ops.py_flip_sublat_B(basis_in,L,pars)
			v[basis_in[ind_pos]] += vc[ind_pos]*zBblock
			v[basis_in[ind_neg]] += vc[ind_neg]*zBblock
			ops.py_flip_sublat_B(basis_in,L,pars)
		
		if type(zblock) is int:
			ops.py_flip_all(basis_in,L,pars)
			v[basis_in[ind_pos]] += vc[ind_pos]*zblock
			v[basis_in[ind_neg]] += vc[ind_neg]*zblock
			ops.py_flip_all(basis_in,L,pars)

		if type(pblock) is int:
			ops.py_fliplr(basis_in,L,pars)
			v[basis_in[ind_pos]] += vc[ind_pos]*pblock
			v[basis_in[ind_neg]] += vc[ind_neg]*pblock
			ops.py_fliplr(basis_in,L,pars)

		if type(pzblock) is int:
			ops.py_fliplr(basis_in,L,pars)
			ops.py_flip_all(basis_in,L,pars)
			v[basis_in[ind_pos]] += vc[ind_pos]*pzblock
			v[basis_in[ind_neg]] += vc[ind_neg]*pzblock
			ops.py_fliplr(basis_in,L,pars)
			ops.py_flip_all(basis_in,L,pars)
		
		ops.py_shift(basis_in,a,L,pars)
	
	return v


def _get_vec_sparse(ops,pars,v0,basis_in,norms,ind_neg,ind_pos,shape,C,L,**blocks):
	dtype=_dtypes[v0.dtype.char]

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	zAblock = blocks.get("zAblock")
	zBblock = blocks.get("zBblock")
	pzblock = blocks.get("pzblock")

	m = shape[1]


	
	if ind_neg.shape[0] == 0:
		row_neg = array([],dtype=_np.int64)
		col_neg = array([],dtype=_np.int64)
	else:
		col_neg = _np.arange(0,m,1)
		row_neg = _np.kron(ind_neg,_np.ones_like(col_neg))
		col_neg = _np.kron(_np.ones_like(ind_neg),col_neg)

	if ind_pos.shape[0] == 0:
		row_pos = array([],dtype=_np.int64)
		col_pos = array([],dtype=_np.int64)
	else:
		col_pos = _np.arange(0,m,1)
		row_pos = _np.kron(ind_pos,_np.ones_like(col_pos))
		col_pos = _np.kron(_np.ones_like(ind_pos),col_pos)



	if type(kblock) is int:
		k = 2*_np.pi*kblock*a/L
	else:
		k = 0.0
		a = L

	c = _np.zeros(basis_in.shape,dtype=v0.dtype)	
	v = _sp.csr_matrix(shape,dtype=v0.dtype)



	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)

		vc = (v0.T*c).T
		data_pos = vc[ind_pos].flatten()
		data_neg = vc[ind_neg].flatten()
		v = v + _sp.csr_matrix((data_pos,(basis_in[row_pos],col_pos)),shape,dtype=v.dtype)
		v = v + _sp.csr_matrix((data_neg,(basis_in[row_neg],col_neg)),shape,dtype=v.dtype)

		if type(zAblock) is int:
			ops.py_flip_sublat_A(basis_in,L,pars)
			data_pos *= zAblock
			data_neg *= zAblock
			v = v + _sp.csr_matrix((data_pos,(basis_in[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg,(basis_in[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zAblock
			data_neg *= zAblock
			ops.py_flip_sublat_A(basis_in,L,pars)

		if type(zBblock) is int:
			ops.py_flip_sublat_B(basis_in,L,pars)
			data_pos *= zBblock
			data_neg *= zBblock
			v = v + _sp.csr_matrix((data_pos,(basis_in[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg,(basis_in[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zBblock
			data_neg *= zBblock
			ops.py_flip_sublat_B(basis_in,L,pars)

		if type(zblock) is int:
			ops.py_flip_all(basis_in,L,pars)
			data_pos *= zblock
			data_neg *= zblock
			v = v + _sp.csr_matrix((data_pos,(basis_in[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg,(basis_in[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zblock
			data_neg *= zblock
			ops.py_flip_all(basis_in,L,pars)

		if type(pblock) is int:
			ops.py_fliplr(basis_in,L,pars)
			data_pos *= pblock
			data_neg *= pblock
			v = v + _sp.csr_matrix((data_pos,(basis_in[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg,(basis_in[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= pblock
			data_neg *= pblock
			ops.py_fliplr(basis_in,L,pars)

		if type(pzblock) is int:
			ops.py_flip_all(basis_in,L,pars)
			ops.py_fliplr(basis_in,L,pars)
			data_pos *= pzblock
			data_neg *= pzblock
			v = v + _sp.csr_matrix((data_pos,(basis_in[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sp.csr_matrix((data_neg,(basis_in[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= pzblock
			data_neg *= pzblock
			ops.py_fliplr(basis_in,L,pars)
			ops.py_flip_all(basis_in,L,pars)

		v.sum_duplicates()
		v.eliminate_zeros()
		ops.py_shift(basis_in,a,L,pars)

	return v


def _get_proj_sparse(ops,pars,basis_in,basis_pcon,norms,ind_neg,ind_pos,dtype,shape,C,L,**blocks):

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	zAblock = blocks.get("zAblock")
	zBblock = blocks.get("zBblock")
	pzblock = blocks.get("pzblock")


	if type(kblock) is int:
		k = 2*_np.pi*kblock*a/L
	else:
		k = 0.0
		a = L

	c = _np.zeros(basis_in.shape,dtype=dtype)	
	v = _sp.csr_matrix(shape,dtype=dtype)

	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)
		data_pos = c[ind_pos]
		data_neg = c[ind_neg]
		if basis_pcon is not None:
			index = _np.searchsorted(basis_pcon,basis_in[ind_pos])
		else:
			index = basis_in[ind_pos]
		v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)

		if basis_pcon is not None:
			index = _np.searchsorted(basis_pcon,basis_in[ind_neg])
		else:
			index = basis_in[ind_neg]
		v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)

		if type(zAblock) is int:
			ops.py_flip_sublat_A(basis_in,L,pars)
			data_pos *= zAblock
			data_neg *= zAblock
			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_pos])
			else:
				index = basis_in[ind_pos]
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)

			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_neg])
			else:
				index = basis_in[ind_neg]
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= zAblock
			data_neg *= zAblock
			ops.py_flip_sublat_A(basis_in,L,pars)

		if type(zBblock) is int:
			ops.py_flip_sublat_B(basis_in,L,pars)
			data_pos *= zBblock
			data_neg *= zBblock
			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_pos])
			else:
				index = basis_in[ind_pos]
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)

			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_neg])
			else:
				index = basis_in[ind_neg]
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= zBblock
			data_neg *= zBblock
			ops.py_flip_sublat_B(basis_in,L,pars)

		if type(zblock) is int:
			ops.py_flip_all(basis_in,L,pars)
			data_pos *= zblock
			data_neg *= zblock
			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_pos])
			else:
				index = basis_in[ind_pos]
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)

			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_neg])
			else:
				index = basis_in[ind_neg]
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= zblock
			data_neg *= zblock
			ops.py_flip_all(basis_in,L,pars)

		if type(pblock) is int:
			ops.py_fliplr(basis_in,L,pars)
			data_pos *= pblock
			data_neg *= pblock
			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_pos])
			else:
				index = basis_in[ind_pos]

			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)

			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_neg])
			else:
				index = basis_in[ind_neg]
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= pblock
			data_neg *= pblock
			ops.py_fliplr(basis_in,L,pars)

		if type(pzblock) is int:
			ops.py_fliplr(basis_in,L,pars)
			ops.py_flip_all(basis_in,L,pars)
			data_pos *= pzblock
			data_neg *= pzblock
			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_pos])
			else:
				index = basis_in[ind_pos]
			v = v + _sp.csr_matrix((data_pos,(index,ind_pos)),shape,dtype=v.dtype)

			if basis_pcon is not None:
				index = _np.searchsorted(basis_pcon,basis_in[ind_neg])
			else:
				index = basis_in[ind_neg]
			v = v + _sp.csr_matrix((data_neg,(index,ind_neg)),shape,dtype=v.dtype)
			data_pos *= pzblock
			data_neg *= pzblock
			ops.py_fliplr(basis_in,L,pars)
			ops.py_flip_all(basis_in,L,pars)

		ops.py_shift(basis_in,a,L,pars)


	return v
