from ..base import basis,MAXPRINT
from . import _constructors as _cn
from ._1d_kblock_Ns import kblock_Ns
from . import _check_1d_symm as _check
import numpy as _np
from scipy.misc import comb
from numpy import array,asarray
from numpy import right_shift,left_shift,invert,bitwise_and,bitwise_or,bitwise_xor
from numpy import cos,sin,exp,pi
from numpy.linalg import norm

import scipy.sparse as _sm

import warnings


# this is how we encode which fortran function to call when calculating 
# the action of operator string

_dtypes={"f":_np.float32,"d":_np.float64,"F":_np.complex64,"D":_np.complex128}
if hasattr(_np,"float128"): _dtypes["g"]=_np.float128
if hasattr(_np,"complex256"): _dtypes["G"]=_np.complex256


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




op={"":_cn.spin_op,
	"M":_cn.spin_n_op,
	"Z":_cn.spin_z_op,
	"ZA":_cn.spin_zA_op,
	"ZB":_cn.spin_zB_op,
	"ZA & ZB":_cn.spin_zA_zB_op,
	"M & Z":_cn.spin_z_op,
	"M & ZA":_cn.spin_zA_op,
	"M & ZB":_cn.spin_zB_op,
	"M & ZA & ZB":_cn.spin_zA_zB_op,
	"P":_cn.spin_p_op,
	"M & P":_cn.spin_p_op,
	"PZ":_cn.spin_pz_op,
	"M & PZ":_cn.spin_pz_op,
	"P & Z":_cn.spin_p_z_op,
	"M & P & Z":_cn.spin_p_z_op,
	"T":_cn.spin_t_op,
	"M & T":_cn.spin_t_op,
	"T & Z":_cn.spin_t_z_op,
	"T & ZA":_cn.spin_t_zA_op,
	"T & ZB":_cn.spin_t_zB_op,
	"T & ZA & ZB":_cn.spin_t_zA_zB_op,
	"M & T & Z":_cn.spin_t_z_op,
	"M & T & ZA":_cn.spin_t_zA_op,
	"M & T & ZB":_cn.spin_t_zB_op,
	"M & T & ZA & ZB":_cn.spin_t_zA_zB_op,
	"T & P":_cn.spin_t_p_op,
	"M & T & P":_cn.spin_t_p_op,
	"T & PZ":_cn.spin_t_pz_op,
	"M & T & PZ":_cn.spin_t_pz_op,
	"T & P & Z":_cn.spin_t_p_z_op,
	"M & T & P & Z":_cn.spin_t_p_z_op
	}

class spin_basis_1d(basis):
	def __init__(self,L,Nup=None,_Np=None,**blocks):

		if blocks.get("a") is None: # by default a = 1
			a=1
			blocks["a"]=1

		if blocks.get("pauli") is None:
			blocks["pauli"] = True

		input_keys = set(blocks.keys())
		expected_keys = set(["kblock","zblock","zAblock","zBblock","pblock","pzblock","pauli","a","count_spins","check_z_symm","L"])
		if not input_keys <= expected_keys:
			wrong_keys = expected_keys - input_keys 
			temp = ", ".join(["{}" for key in wrong_keys])
			raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))
 

		if type(Nup) is int:
			self._check_pcon=True
			self._make_Nup_block(L,Nup=Nup,**blocks)
	
		elif Nup is None: # User hasn't specified Nup,
			if _Np is not None: # check to see if photon_basis can create the particle sectors.

				if type(_Np) is not int:
					raise ValueError("Np must be integer")

				if _Np == -1: 
					spin_basis_1d.__init__(self,L,Nup=None,_Np=None,**blocks)
				elif _Np >= 0:
					if _Np+1 > L: _Np = L
					blocks["count_spins"] = True

					zblock = blocks.get("zblock")
					zAblock = blocks.get("zAblock")
					zBblock = blocks.get("zAblock")
				
					if (type(zblock) is int) or (type(zAblock) is int) or (type(zBblock) is int):
						raise ValueError("spin inversion symmetry not compatible with particle conserving photon_basis.")
					
					# loop over the first Np particle sectors (use the iterator initialization).
					spin_basis_1d.__init__(self,L,Nup=range(_Np+1),_Np=None,**blocks)
				else:
					raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")

			else: # if _Np is None then assume user wants to not specify Magnetization sector
				self._check_pcon = False
				self._make_Nup_block(L,Nup=Nup,**blocks)


		else: # try to interate over Nup 
			try:
				Nup_iter = iter(Nup)
			except TypeError:
				raise TypeError("Nup must be integer or iteratable object.")

			blocks["check_z_symm"] = False
			Nup = next(Nup_iter)
#			print Nup
			spin_basis_1d.__init__(self,L,Nup=Nup,**blocks)

			for Nup in Nup_iter:
#				print Nup
				temp_basis = spin_basis_1d(L,Nup=Nup,**blocks)
				self.append(temp_basis)	
		


				


	def _make_Nup_block(self,L,Nup=None,**blocks):
		# getting arguments which are used in basis.
		kwkeys = set(blocks.keys())
		kblock=blocks.get("kblock")
		zblock=blocks.get("zblock")
		zAblock=blocks.get("zAblock")
		zBblock=blocks.get("zBblock")
		pblock=blocks.get("pblock")
		pzblock=blocks.get("pzblock")
		a=blocks.get("a")


		count_spins = blocks.get("count_spins")
		if count_spins is None:
			count_spins=False

		check_z_symm = blocks.get("check_z_symm")
		if check_z_symm is None:
			check_z_symm=True


		if type(L) is not int:
			raise TypeError('L must be integer')

		if L <= 32: 
			self._basis_type = _np.uint32
		elif L <= 64:
			self._basis_type = _np.uint64
		else:
			self._basis_type = _np.object

		if type(a) is not int:
			raise TypeError('a must be integer')

		# checking if a is compatible with L
		if(L%a != 0):
			raise ValueError('L must be interger multiple of lattice spacing a')



		self._L=L
		if type(Nup) is int:
			self._conserved="M"
			self._Ns=comb(L,Nup,exact=True)
		else:
			self._conserved=""
			self._Ns=2**L



		# checking type, and value of blocks
		if Nup is not None:
			if type(Nup) is not int: raise TypeError('Nup must be integer')
			if Nup < 0 or Nup > L: raise ValueError("0 <= Nup <= %d" % L)

		if pblock is not None:
			if type(pblock) is not int: raise TypeError('pblock must be integer')
			if abs(pblock) != 1: raise ValueError("pblock must be +/- 1")

		if zblock is not None:
			if type(zblock) is not int: raise TypeError('zblock must be integer')
			if abs(zblock) != 1: raise ValueError("zblock must be +/- 1")

		if zAblock is not None:
			if type(zAblock) is not int: raise TypeError('zAblock must be integer')
			if abs(zAblock) != 1: raise ValueError("zAblock must be +/- 1")

		if zBblock is not None:
			if type(zBblock) is not int: raise TypeError('zBblock must be integer')
			if abs(zBblock) != 1: raise ValueError("zBblock must be +/- 1")

		if pzblock is not None:
			if type(pzblock) is not int: raise TypeError('pzblock must be integer')
			if abs(pzblock) != 1: raise ValueError("pzblock must be +/- 1")

		if kblock is not None and (a < L):
			if type(kblock) is not int: raise TypeError('kblock must be integer')
			kblock = kblock % (L//a)
			blocks["kblock"] = kblock
#			Nup_tup = Nup
#			if Nup is not None:
#				if Nup > L//2: Nup_tup = L - Nup
			 
				

#			if kblock > L//(2*a): kblock_tup = L//a - kblock
#			else: kblock_tup = kblock
#			self._Ns = kblock_Ns.get((L,a,Nup_tup,kblock_tup))
#			if self._Ns is None:
#				self._Ns = 1
		if type(Nup) is int:
			self._Ns = comb(L,Nup,exact=True)
		else:
			self._Ns = (1 << L)


		if type(kblock) is int:
			self._Ns = _cn.kblock_Ns_estimate(self._Ns,L,a)



		# shout out if pblock and zA/zB blocks defined simultaneously
		if type(pblock) is int and ((type(zAblock) is int) or (type(zBblock) is int)):
			raise ValueError("zA and zB symmetries incompatible with parity symmetry")

		if check_z_symm:
			blocks["Nup"] = Nup
			# checking if spin inversion is compatible with Nup and L
			if (type(Nup) is int) and ((type(zblock) is int) or (type(pzblock) is int)):
				if (L % 2) != 0:
					raise ValueError("spin inversion symmetry with magnetization conservation must be used with even number of sites")
				if Nup != L//2:
					raise ValueError("spin inversion symmetry only reduces the 0 magnetization sector")

			if (type(Nup) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
				raise ValueError("zA and zB symmetries incompatible with magnetisation symmetry")

			# checking if ZA/ZB spin inversion is compatible with unit cell of translation symemtry
			if (type(kblock) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
				if a%2 != 0: # T and ZA (ZB) symemtries do NOT commute
					raise ValueError("unit cell size 'a' must be even")




		self._blocks=blocks
		self._operators = ("availible operators for spin_basis_1d:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tx: x pauli/spin operator"+
							"\n\ty: y pauli/spin operator"+
							"\n\tz: z pauli/spin operator")

		# allocates memory for number of basis states
		frac = 1.0

		self._unique_me = True	
		
		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & P & Z"
			else: self._conserved = "T & P & Z"
			self._blocks["pzblock"] = pblock*zblock
			self._unique_me = False

			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8) # normalisation*sigma
			self._m=_np.empty(self._basis.shape,dtype=_np.int16) #m = mp + (L+1)mz + (L+1)^2c; Anders' paper
			if (type(Nup) is int):
				# arguments get overwritten by _cn.spin_...  
				self._Ns = _cn.spin_n_t_p_z_basis(L,Nup,pblock,zblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.spin_t_p_z_basis(L,pblock,zblock,kblock,a,self._N,self._m,self._basis)

			# cut off extra memory for overestimated state number
			self._N.resize((self._Ns,))
			self._m.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(zAblock) is int) and (type(zBblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & ZA & ZB"
			else: self._conserved = "T & ZA & ZB"
			self._blocks["zblock"] = zAblock*zBblock


			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int16)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_t_zA_zB_basis(L,Nup,zAblock,zBblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.spin_t_zA_zB_basis(L,zAblock,zBblock,kblock,a,self._N,self._m,self._basis)

			self._N.resize((self._Ns,))
			self._m.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(pzblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & PZ"
			else: self._conserved = "T & PZ"
			self._unique_me = False

			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8) #mpz
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_t_pz_basis(L,Nup,pzblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.spin_t_pz_basis(L,pzblock,kblock,a,self._N,self._m,self._basis)

			self._N.resize((self._Ns,))
			self._m.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(pblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & P"
			else: self._conserved = "T & P"
			self._unique_me = False

			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_t_p_basis(L,Nup,pblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.spin_t_p_basis(L,pblock,kblock,a,self._N,self._m,self._basis)


			self._N.resize((self._Ns,))
			self._m.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(zblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & Z"
			else: self._conserved = "T & Z"


			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_t_z_basis(L,Nup,zblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.spin_t_z_basis(L,zblock,kblock,a,self._N,self._m,self._basis)

			self._N.resize((self._Ns,))
			self._m.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._m,self._basis,self._L]


		elif (type(kblock) is int) and (type(zAblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & ZA"
			else: self._conserved = "T & ZA"


			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_t_zA_basis(L,Nup,zAblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.spin_t_zA_basis(L,zAblock,kblock,a,self._N,self._m,self._basis)

			self._N.resize((self._Ns,))
			self._m.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(zBblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & ZB"
			else: self._conserved = "T & ZB"


			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_t_zB_basis(L,Nup,zBblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.spin_t_zB_basis(L,zBblock,kblock,a,self._N,self._m,self._basis)

			self._N.resize((self._Ns,))
			self._m.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(pblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & P & Z"
			else: self._conserved += "P & Z"
			self._blocks["pzblock"] = pblock*zblock

			
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_p_z_basis(L,Nup,pblock,zblock,self._N,self._basis)
			else:
				self._Ns = _cn.spin_p_z_basis(L,pblock,zblock,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L]


		elif (type(zAblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & ZA & ZB"
			else: self._conserved += "ZA & ZB"
			self._blocks["zblock"] = zAblock*zBblock

			
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_zA_zB_basis(L,Nup,self._basis)
			else:
				self._Ns = _cn.spin_zA_zB_basis(L,self._basis)

			self._basis.resize((self._Ns,))
			self._op_args=[self._basis,self._L]



		elif type(pblock) is int:
			if self._conserved: self._conserved += " & P"
			else: self._conserved = "P"
			
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_p_basis(L,Nup,pblock,self._N,self._basis)
			else:
				self._Ns = _cn.spin_p_basis(L,pblock,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L]



		elif type(zblock) is int:
			if self._conserved: self._conserved += " & Z"
			else: self._conserved += "Z"

			
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_z_basis(L,Nup,self._basis)
			else:
				self._Ns = _cn.spin_z_basis(L,self._basis)

			self._basis.resize((self._Ns,))
			self._op_args=[self._basis,self._L]

		elif type(zAblock) is int:
			if self._conserved: self._conserved += " & ZA"
			else: self._conserved += "ZA"

			
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_zA_basis(L,Nup,self._basis)
			else:
				self._Ns = _cn.spin_zA_basis(L,self._basis)

			self._basis.resize((self._Ns,))
			self._op_args=[self._basis,self._L]


		elif type(zBblock) is int:
			if self._conserved: self._conserved += " & ZB"
			else: self._conserved += "ZB"
			
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_zB_basis(L,Nup,self._basis)
			else:
				self._Ns = _cn.spin_zB_basis(L,self._basis)

			self._basis.resize((self._Ns,))
			self._op_args=[self._basis,self._L]
				
		elif type(pzblock) is int:
			if self._conserved: self._conserved += " & PZ"
			else: self._conserved += "PZ"
			
			self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_pz_basis(L,Nup,pzblock,self._N,self._basis)
			else:
				self._Ns = _cn.spin_pz_basis(L,pzblock,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L]
	
		elif type(kblock) is int:
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T"
			else: self._conserved = "T"
			
			self._basis=_np.empty((self._Ns,),dtype=self._basis_type)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.spin_n_t_basis(L,Nup,kblock,a,self._N,self._basis)
			else:
				self._Ns = _cn.spin_t_basis(L,kblock,a,self._N,self._basis)

			self._N.resize((self._Ns,))
			self._basis.resize((self._Ns,))
			self._op_args=[self._N,self._basis,self._L]

		else: 
			if type(Nup) is int:
				self._basis = _np.empty((self._Ns,),dtype=self._basis_type)
				_cn.spin_n_basis(L,Nup,self._Ns,self._basis)
			else:
				self._basis = _np.arange(0,self._Ns,1,dtype=self._basis_type)
			self._op_args=[self._basis]

		if count_spins: self._Np = _np.full_like(self._basis,Nup,dtype=_np.int8)



	def append(self,other):
		if not isinstance(other,spin_basis_1d):
			raise TypeError("can only append spin_basis_1d object to another")
		if self._L != other._L:
			raise ValueError("spin_basis_1d appending incompatible system sizes with: {0} and {1}".format(self._L,other._L))
		if self._blocks != other._blocks:
			raise ValueError("spin_basis_1d appending incompatible blocks: {0} and {1}".format(self._blocks,other._blocks))
		

		Ns = self._Ns + other._Ns

		if self._conserved == "" or self._conserved == "M":
			self._op_args=[]
		else:
			self._op_args=[self._L]


		self._basis.resize((Ns,),refcheck=False)
		self._basis[self._Ns:] = other._basis[:]
		arg = _np.argsort(self._basis)
		self._basis = self._basis[arg]

		self._op_args.insert(0,self._basis)

		if hasattr(self,"_Np"):
			self._Np.resize((Ns,),refcheck=False)
			self._Np[self._Ns:] = other._Np[:]
			self._Np = self._Np[arg]

		if hasattr(self,"_m"):
			self._m.resize((Ns,),refcheck=False)
			self._m[self._Ns:] = other._m[:]
			self._m = self._m[arg]
			self._op_args.insert(0,self._m)	

		if hasattr(self,"_N"):
			self._N.resize((Ns,),refcheck=False)
			self._N[self._Ns:] = other._N[:]
			self._N = self._N[arg]
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
	def description(self):
		blocks = ""
		lat_space = "lattice spacing: a = {a}".format(**self._blocks)

		for symm in self._blocks:
			if symm != "a":
				blocks += symm+" = {"+symm+"}, "

		blocks = blocks.format(**self._blocks)

		if len(self._conserved) == 0:
			symm = "no symmetry"
		elif len(self._conserved) == 1:
			symm = "symmetry"
		else:
			symm = "symmetries"

		string = """1d spin 1/2 basis for chain of L = {0} containing {5} states \n\t{1}: {2} \n\tquantum numbers: {4} \n\t{3} \n\n""".format(self._L,symm,self._conserved,lat_space,blocks,self._Ns)
		string += self.operators
		return string 


	def __type__(self):
		return "<type 'qspin.basis.spin_basis_1d'>"


	def __getitem__(self,key):
		return self._basis.__getitem__(key)

	def index(self,s):
		return _np.searchsorted(self._basis,s)

	def __iter__(self):
		return self._basis.__iter__()


	def Op(self,opstr,indx,J,dtype):
		indx = _np.asarray(indx,dtype=_np.int32)

		if opstr.count("|") > 0:
			raise ValueError("charactor '|' not allowed in opstr: {0}".format(opstr)) 
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')
		if _np.any(indx >= self._L) or _np.any(indx < 0):
			raise ValueError('values in indx falls outside of system')

		if self._Ns <= 0:
			return [],[],[]

		pauli = self._blocks['pauli']

		N_op = _cn.op_array_size[self._conserved]*self.Ns
		col = _np.zeros(N_op,dtype=self._basis_type)
		row = _np.zeros(N_op,dtype=self._basis_type)
		ME = _np.zeros(N_op,dtype=dtype)

		error = op[self._conserved](row,col,ME,opstr,indx,J,*self._op_args,**self._blocks)

		if error != 0: raise OpstrError(_basis_op_errors[error])

		mask = _np.logical_not(_np.isnan(ME))
		col = col[mask]
		row = row[mask]
		ME = ME[mask]

		if not pauli:
			Nop = len(opstr.replace("I",""))
			ME /= (1 << Nop)

		return ME,row,col		



	def get_norms(self,dtype):
		a = self._blocks.get("a")
		kblock = self._blocks.get("kblock")
		pblock = self._blocks.get("pblock")
		zblock = self._blocks.get("zblock")
		zAblock = self._blocks.get("zAblock")
		zBblock = self._blocks.get("zBblock")
		pzblock = self._blocks.get("pzblock")


		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			c = _np.empty(self._m.shape,dtype=_np.int8)
			nn = _np.array(c)
			mm = _np.array(c)
			_np.floor_divide(self._m,(self._L+1)**2,out=c)
			_np.floor_divide(self._m,self._L+1,out=nn)
			_np.mod(nn,self._L+1,out=nn)
			_np.mod(self._m,self._L+1,out=mm)
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
			c = _np.empty(self._m.shape,dtype=_np.int8)
			mm = _np.array(c)
			_np.floor_divide(self._m,(self._L+1),c)
			_np.mod(self._m,self._L+1,mm)
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
			mask = (self._m >= 0)
			norm[mask] *= (1.0 + _np.sign(self._N[mask])*pblock*_np.cos(self._k*self._m[mask]))
			del mask
		elif (type(kblock) is int) and (type(pzblock) is int):
			if _np.abs(_np.sin(self._k)) < 1.0/self._L:
				norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm *= _np.sign(self._N)
			norm /= self._N
			# m >= 0 
			mask = (self._m >= 0)
			norm[mask] *= (1.0 + _np.sign(self._N[mask])*pzblock*_np.cos(self._k*self._m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			# m >= 0 
			mask = (self._m >= 0)
			norm[mask] *= (1.0 + zblock*_np.cos(self._k*self._m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zAblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			# m >= 0 
			mask = (self._m >= 0)
			norm[mask] *= (1.0 + zAblock*_np.cos(self._k*self._m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zBblock) is int):
			norm = _np.full(self._basis.shape,2*(self._L/a)**2,dtype=dtype)
			norm /= self._N
			# m >= 0 
			mask = (self._m >= 0)
			norm[mask] *= (1.0 + zBblock*_np.cos(self._k*self._m[mask]))
			del mask
		elif (type(pblock) is int) and (type(zblock) is int):
			norm = _np.array(self._N,dtype=dtype)
		elif (type(zAblock) is int) and (type(zBblock) is int):
			norm = _np.full(self._basis.shape,4.0,dtype=dtype)
		elif (type(pblock) is int):
			norm = _np.array(self._N,dtype=dtype)
		elif (type(pzblock) is int):
			norm = _np.array(self._N,dtype=dtype)
		elif (type(zblock) is int):
			norm = _np.full(self._basis.shape,2.0,dtype=dtype)
		elif (type(zAblock) is int):
			norm = _np.full(self._basis.shape,2.0,dtype=dtype)
		elif (type(zBblock) is int):
			norm = _np.full(self._basis.shape,2.0,dtype=dtype)
		elif (type(kblock) is int):
			norm = _np.full(self._basis.shape,(self._L/a)**2,dtype=dtype)
			norm /= self._N
		else:
			norm = _np.ones(self._basis.shape,dtype=dtype)
	
		_np.sqrt(norm,norm)

		return norm




	def get_vec(self,v0,sparse=True):
		if _sm.issparse(v0):
			raise TypeError("expecting v0 to be dense array")

		if not hasattr(v0,"shape"):
			v0 = _np.asanyarray(v0)

		if self._Ns <= 0:
			return _np.array([])
		if v0.ndim == 1:
			ravel=True
			shape = (2**self._L,1)
			v0 = v0.reshape((-1,1))
		elif v0.ndim == 2:
			ravel=False
			shape = (2**self._L,v0.shape[1])
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")

		if v0.shape[0] != self._Ns:
			raise ValueError("v0 shape {0} not compatible with Ns={1}".format(v0.shape,self._Ns))


		norms = self.get_norms(v0.dtype)

		a = self._blocks.get("a")
		kblock = self._blocks.get("kblock")
		pblock = self._blocks.get("pblock")
		zblock = self._blocks.get("zblock")
		zAblock = self._blocks.get("zAblock")
		zBblock = self._blocks.get("zBblock")
		pzblock = self._blocks.get("pzblock")


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
			ind_neg = _np.array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == _np.pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(dtype(1.0j*k*r))
				_np.true_divide(c,norms,c)

		if sparse:
			return _get_vec_sparse(v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks)
		else:
			if ravel:
				return  _get_vec_dense(v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks).ravel()
			else:
				return  _get_vec_dense(v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks)


	def get_proj(self,dtype):
		if self._Ns <= 0:
			return _np.array([])

		norms = self.get_norms(dtype)

		a = self._blocks.get("a")
		kblock = self._blocks.get("kblock")
		pblock = self._blocks.get("pblock")
		zblock = self._blocks.get("zblock")
		zAblock = self._blocks.get("zAblock")
		zBblock = self._blocks.get("zBblock")
		pzblock = self._blocks.get("pzblock")

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
			ind_neg = _np.array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == _np.pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(dtype(1.0j*k*r))
				_np.true_divide(c,norms,c)

		return _get_proj_sparse(self._basis,norms,ind_neg,ind_pos,dtype,C,self._L,**self._blocks)






	# functions called in base class:


	def _get__str__(self):
		n_digits = int(_np.ceil(_np.log10(self._Ns)))
		temp1 = "\t{0:"+str(n_digits)+"d}.  "
		temp2 = ">{0:0"+str(self._L)+"b}|"

		if self._Ns > MAXPRINT:
			half = MAXPRINT // 2
			str_list = [(temp1.format(i))+(temp2.format(b))[::-1] for i,b in zip(range(half),self._basis[:half])]
			str_list.extend([(temp1.format(i))+(temp2.format(b))[::-1] for i,b in zip(range(self._Ns-half,self._Ns,1),self._basis[-half:])])
		else:
			str_list = [(temp1.format(i))+(temp2.format(b))[::-1] for i,b in enumerate(self._basis)]

		return tuple(str_list)



	def _sort_opstr(self,op):
		if op[0].count("|") > 0:
			raise ValueError("'|' character found in op: {0},{1}".format(op[0],op[1]))
		if len(op[0]) != len(op[1]):
			raise ValueError("number of operators in opstr: {0} not equal to length of indx {1}".format(op[0],op[1]))

		op = list(op)
		zipstr = list(zip(op[0],op[1]))
		if zipstr:
			zipstr.sort(key = lambda x:x[1])
			op1,op2 = zip(*zipstr)
			op[0] = "".join(op1)
			op[1] = tuple(op2)
		return tuple(op)

	def _non_zero(self,op):
		opstr = _np.array(list(op[0]))
		indx = _np.array(op[1])
		if _np.any(indx):
			indx_p = indx[opstr == "+"].tolist()
			p = not any(indx_p.count(x) > 1 for x in indx_p)
			indx_p = indx[opstr == "-"].tolist()
			m = not any(indx_p.count(x) > 1 for x in indx_p)
			return (p and m)
		else:
			return True
		


	def _hc_opstr(self,op):
		op = list(op)
		# take h.c. + <--> - , reverse operator order , and conjugate coupling
		op[0] = list(op[0].replace("+","%").replace("-","+").replace("%","-"))
		op[0].reverse()
		op[0] = "".join(op[0])
		op[1] = list(op[1])
		op[1].reverse()
		op[1] = tuple(op[1])
		op[2] = op[2].conjugate()
		return self._sort_opstr(op) # return the sorted op.


	def _expand_opstr(self,op,num):
		opstr = str(op[0])
		indx = list(op[1])
		J = op[2]
 
		if len(opstr) <= 1:
			if opstr == "x":
				op1 = list(op)
				op1[0] = op1[0].replace("x","+")
				op1[2] *= 0.5
				op1.append(num)

				op2 = list(op)
				op2[0] = op2[0].replace("x","-")
				op2[2] *= 0.5
				op2.append(num)

				return (tuple(op1),tuple(op2))
			elif opstr == "y":
				op1 = list(op)
				op1[0] = op1[0].replace("y","+")
				op1[2] *= -0.5j
				op1.append(num)

				op2 = list(op)
				op2[0] = op2[0].replace("y","-")
				op2[2] *= 0.5j
				op2.append(num)

				return (tuple(op1),tuple(op2))
			else:
				op = list(op)
				op.append(num)
				return [tuple(op)]	
		else:
	 
			i = len(opstr)//2
			op1 = list(op)
			op1[0] = opstr[:i]
			op1[1] = tuple(indx[:i])
			op1[2] = complex(J)
			op1 = tuple(op1)

			op2 = list(op)
			op2[0] = opstr[i:]
			op2[1] = tuple(indx[i:])
			op2[2] = complex(1)
			op2 = tuple(op2)

			l1 = self._expand_opstr(op1,num)
			l2 = self._expand_opstr(op2,num)

			l = []
			for op1 in l1:
				for op2 in l2:
					op = list(op1)
					op[0] += op2[0]
					op[1] += op2[1]
					op[2] *= op2[2]
					l.append(tuple(op))

			return tuple(l)



	def _check_symm(self,static,dynamic,basis=None):
		kblock = self._blocks.get("kblock")
		pblock = self._blocks.get("pblock")
		zblock = self._blocks.get("zblock")
		pzblock = self._blocks.get("pzblock")
		zAblock = self._blocks.get("zAblock")
		zBblock = self._blocks.get("zBblock")
		a = self._blocks.get("a")
		L = self.L

		if basis is None:
			basis = self

		static_list,dynamic_list = basis.get_lists(static,dynamic)

		static_blocks = {}
		dynamic_blocks = {}
		if kblock is not None:
			missing_ops = _check.check_T(basis,static_list,L,a)
			if missing_ops:	static_blocks["T symm"] = (tuple(missing_ops),)

			missing_ops = _check.check_T(basis,dynamic_list,L,a)
			if missing_ops:	dynamic_blocks["T symm"] = (tuple(missing_ops),)

		if pblock is not None:
			missing_ops = _check.check_P(basis,static_list,L)
			if missing_ops:	static_blocks["P symm"] = (tuple(missing_ops),)

			missing_ops = _check.check_P(basis,dynamic_list,L)
			if missing_ops:	dynamic_blocks["P symm"] = (tuple(missing_ops),)

		if zblock is not None:
			odd_ops,missing_ops = _check.check_Z(basis,static_list)
			if missing_ops or odd_ops:
				static_blocks["Z symm"] = (tuple(odd_ops),tuple(missing_ops))

			odd_ops,missing_ops = _check.check_Z(basis,dynamic_list)
			if missing_ops or odd_ops:
				dynamic_blocks["Z symm"] = (tuple(odd_ops),tuple(missing_ops))

		if zAblock is not None:
			odd_ops,missing_ops = _check.check_ZA(basis,static_list)
			if missing_ops or odd_ops:
				static_blocks["ZA symm"] = (tuple(odd_ops),tuple(missing_ops))

			odd_ops,missing_ops = _check.check_ZA(basis,dynamic_list)
			if missing_ops or odd_ops:
				dynamic_blocks["ZA symm"] = (tuple(odd_ops),tuple(missing_ops))

		if zBblock is not None:
			odd_ops,missing_ops = _check.check_ZB(basis,static_list)
			if missing_ops or odd_ops:
				static_blocks["ZB symm"] = (tuple(odd_ops),tuple(missing_ops))

			odd_ops,missing_ops = _check.check_ZB(basis,dynamic_list)
			if missing_ops or odd_ops:
				dynamic_blocks["ZB symm"] = (tuple(odd_ops),tuple(missing_ops))

		if pzblock is not None:
			missing_ops = _check.check_PZ(basis,static_list,L)
			if missing_ops:	static_blocks["PZ symm"] = (tuple(missing_ops),)

			missing_ops = _check.check_PZ(basis,dynamic_list,L)
			if missing_ops:	dynamic_blocks["PZ symm"] = (tuple(missing_ops),)

		return static_blocks,dynamic_blocks








def _get_vec_dense(v0,basis,norms,ind_neg,ind_pos,shape,C,L,**blocks):
	dtype=_dtypes[v0.dtype.char]

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	zAblock = blocks.get("zAblock")
	zBblock = blocks.get("zBblock")
	pzblock = blocks.get("pzblock")


	c = _np.zeros(basis.shape,dtype=v0.dtype)	
	v = _np.zeros(shape,dtype=v0.dtype)

	bits=" ".join(["{"+str(i)+":0"+str(L)+"b}" for i in range(len(basis))])

	if type(kblock) is int:
		k = 2*_np.pi*kblock*a/L
	else:
		k = 0.0
		a = L

	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)	
		vc = (v0.T*c).T
		v[basis[ind_pos]] += vc[ind_pos]
		v[basis[ind_neg]] += vc[ind_neg]

		if type(zAblock) is int:
			flip_sublat_A(basis,L)
			v[basis[ind_pos]] += vc[ind_pos]*zAblock
			v[basis[ind_neg]] += vc[ind_neg]*zAblock
			flip_sublat_A(basis,L)
		
		if type(zBblock) is int:
			flip_sublat_B(basis,L)
			v[basis[ind_pos]] += vc[ind_pos]*zBblock
			v[basis[ind_neg]] += vc[ind_neg]*zBblock
			flip_sublat_B(basis,L)
		
		if type(zblock) is int:
			flipall(basis,L)
			v[basis[ind_pos]] += vc[ind_pos]*zblock
			v[basis[ind_neg]] += vc[ind_neg]*zblock
			flipall(basis,L)

		if type(pblock) is int:
			fliplr(basis,L)
			v[basis[ind_pos]] += vc[ind_pos]*pblock
			v[basis[ind_neg]] += vc[ind_neg]*pblock
			fliplr(basis,L)

		if type(pzblock) is int:
			fliplr(basis,L)
			flipall(basis,L)
			v[basis[ind_pos]] += vc[ind_pos]*pzblock
			v[basis[ind_neg]] += vc[ind_neg]*pzblock
			fliplr(basis,L)
			flipall(basis,L)
		
		shiftc(basis,-a,L)
	
	return v





def _get_vec_sparse(v0,basis,norms,ind_neg,ind_pos,shape,C,L,**blocks):
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
		row_neg = _np.array([],dtype=_np.int64)
		col_neg = _np.array([],dtype=_np.int64)
	else:
		col_neg = _np.arange(0,m,1)
		row_neg = _np.kron(ind_neg,_np.ones_like(col_neg))
		col_neg = _np.kron(_np.ones_like(ind_neg),col_neg)

	if ind_pos.shape[0] == 0:
		row_pos = _np.array([],dtype=_np.int64)
		col_pos = _np.array([],dtype=_np.int64)
	else:
		col_pos = _np.arange(0,m,1)
		row_pos = _np.kron(ind_pos,_np.ones_like(col_pos))
		col_pos = _np.kron(_np.ones_like(ind_pos),col_pos)



	if type(kblock) is int:
		k = 2*_np.pi*kblock*a/L
	else:
		k = 0.0
		a = L

	c = _np.zeros(basis.shape,dtype=v0.dtype)	
	v = _sm.csr_matrix(shape,dtype=v0.dtype)



	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)

		vc = (v0.T*c).T
		data_pos = vc[ind_pos].flatten()
		data_neg = vc[ind_neg].flatten()
		v = v + _sm.csr_matrix((data_pos,(basis[row_pos],col_pos)),shape,dtype=v.dtype)
		v = v + _sm.csr_matrix((data_neg,(basis[row_neg],col_neg)),shape,dtype=v.dtype)

		if type(zAblock) is int:
			flip_sublat_A(basis,L)
			data_pos *= zAblock
			data_neg *= zAblock
			v = v + _sm.csr_matrix((data_pos,(basis[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zAblock
			data_neg *= zAblock
			flip_sublat_A(basis,L)

		if type(zBblock) is int:
			flip_sublat_B(basis,L)
			data_pos *= zBblock
			data_neg *= zBblock
			v = v + _sm.csr_matrix((data_pos,(basis[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zBblock
			data_neg *= zBblock
			flip_sublat_B(basis,L)

		if type(zblock) is int:
			flipall(basis,L)
			data_pos *= zblock
			data_neg *= zblock
			v = v + _sm.csr_matrix((data_pos,(basis[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= zblock
			data_neg *= zblock
			flipall(basis,L)

		if type(pblock) is int:
			fliplr(basis,L)
			data_pos *= pblock
			data_neg *= pblock
			v = v + _sm.csr_matrix((data_pos,(basis[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= pblock
			data_neg *= pblock
			fliplr(basis,L)

		if type(pzblock) is int:
			fliplr(basis,L)
			flipall(basis,L)
			data_pos *= pzblock
			data_neg *= pzblock
			v = v + _sm.csr_matrix((data_pos,(basis[row_pos],col_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[row_neg],col_neg)),shape,dtype=v.dtype)
			data_pos *= pzblock
			data_neg *= pzblock
			fliplr(basis,L)
			flipall(basis,L)

		v.sum_duplicates()
		v.eliminate_zeros()
		shiftc(basis,-a,L)

	return v




def _get_proj_sparse(basis,norms,ind_neg,ind_pos,dtype,C,L,**blocks):

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

	shape = (2**L,basis.shape[0])

	c = _np.zeros(basis.shape,dtype=dtype)	
	v = _sm.csr_matrix(shape,dtype=dtype)


	for r in range(0,L//a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)
		data_pos = c[ind_pos]
		data_neg = c[ind_neg]
		v = v + _sm.csr_matrix((data_pos,(basis[ind_pos],ind_pos)),shape,dtype=v.dtype)
		v = v + _sm.csr_matrix((data_neg,(basis[ind_neg],ind_neg)),shape,dtype=v.dtype)

		if type(zAblock) is int:
			flip_sublat_A(basis,L)
			data_pos *= zAblock
			data_neg *= zAblock
			v = v + _sm.csr_matrix((data_pos,(basis[ind_pos],ind_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[ind_neg],ind_neg)),shape,dtype=v.dtype)
			data_pos *= zAblock
			data_neg *= zAblock
			flip_sublat_A(basis,L)

		if type(zBblock) is int:
			flip_sublat_B(basis,L)
			data_pos *= zBblock
			data_neg *= zBblock
			v = v + _sm.csr_matrix((data_pos,(basis[ind_pos],ind_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[ind_neg],ind_neg)),shape,dtype=v.dtype)
			data_pos *= zBblock
			data_neg *= zBblock
			flip_sublat_B(basis,L)

		if type(zblock) is int:
			flipall(basis,L)
			data_pos *= zblock
			data_neg *= zblock
			v = v + _sm.csr_matrix((data_pos,(basis[ind_pos],ind_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[ind_neg],ind_neg)),shape,dtype=v.dtype)
			data_pos *= zblock
			data_neg *= zblock
			flipall(basis,L)

		if type(pblock) is int:
			fliplr(basis,L)
			data_pos *= pblock
			data_neg *= pblock
			v = v + _sm.csr_matrix((data_pos,(basis[ind_pos],ind_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[ind_neg],ind_neg)),shape,dtype=v.dtype)
			data_pos *= pblock
			data_neg *= pblock
			fliplr(basis,L)

		if type(pzblock) is int:
			fliplr(basis,L)
			flipall(basis,L)
			data_pos *= pzblock
			data_neg *= pzblock
			v = v + _sm.csr_matrix((data_pos,(basis[ind_pos],ind_pos)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[ind_neg],ind_neg)),shape,dtype=v.dtype)
			data_pos *= pzblock
			data_neg *= pzblock
			fliplr(basis,L)
			flipall(basis,L)

		shiftc(basis,-a,L)


	return v





def fliplr(x,length):
	x1 = array(x)
	x[:] = 0
	for i in range(length):
		x2 = array(x1)
		x2 = right_shift(x2,i)
		bitwise_and(x2,1,out=x2)
		left_shift(x2,length-1-i,out=x2)
		x += x2


def flipall(x,length):
	mask = 2**length-1
	invert(x,out=x)
	bitwise_and(x,mask,out=x)


def flip_sublat_A(x,length):
	# flip all even bits: sublat A
	mask = sum(2**i for i in range(0,length,2))
	bitwise_xor(x,mask,out=x)
	
def flip_sublat_B(x,length):
	# flip all odd bits: sublat B
	mask = sum(2**i for i in range(1,length,2))
	bitwise_xor(x,mask,out=x)


def shiftc(x,shift,period):
	Imax=2**period-1

	bitwise_and(x,Imax,x)
	x1 = array(x)
	if shift < 0:	
		shift=abs(shift)
		shift = shift % period
		m_shift = period - shift

		left_shift(x,shift,out=x)
		bitwise_and(x,Imax,out=x)
		right_shift(x1,m_shift,out=x1)
		bitwise_or(x,x1,out=x)
	else:
		shift = shift % period
		m_shift = period - shift

		right_shift(x,shift,out=x)
		left_shift(x1,m_shift,out=x1)
		bitwise_and(x1,Imax,out=x1)
		bitwise_or(x,x1,out=x)

	del x1



