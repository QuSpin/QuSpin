from ..base import basis

import constructors as _cn
import numpy as _np
from numpy import array,asarray
from numpy import right_shift,left_shift,invert,bitwise_and,bitwise_or
from numpy import cos,sin,exp,pi
from numpy.linalg import norm

import scipy.sparse as _sm

# this is how we encode which fortran function to call when calculating 
# the action of operator string


_dtypes={"f":_np.float32,"d":_np.float64,"F":_np.complex64,"D":_np.complex128}
if hasattr(_np,"float128"): _dtypes["g"]=_np.float128
if hasattr(_np,"complex256"): _dtypes["G"]=_np.complex256



op={"":_cn.op,
		"M":_cn.op_m,
		"Z":_cn.op_z,
		"ZA":_cn.op_zA,
		"ZB":_cn.op_zB,
		"ZA & ZB":_cn.op_zA_zB,
		"M & Z":_cn.op_z,
		"M & ZA":_cn.op_zA,
		"M & ZB":_cn.op_zB,
		"M & ZA & ZB":_cn.op_zA_zB,
		"P":_cn.op_p,
		"M & P":_cn.op_p,
		"PZ":_cn.op_pz,
		"M & PZ":_cn.op_pz,
		"P & Z":_cn.op_p_z,
		"M & P & Z":_cn.op_p_z,
		"T":_cn.op_t,
		"M & T":_cn.op_t,
		"T & Z":_cn.op_t_z,
		"T & ZA":_cn.op_t_zA,
		"T & ZB":_cn.op_t_zB,
		"T & ZA & ZB":_cn.op_t_zA_zB,
		"M & T & Z":_cn.op_t_z,
		"M & T & ZA":_cn.op_t_zA,
		"M & T & ZB":_cn.op_t_zB,
		"M & T & ZA & ZB":_cn.op_t_zA_zB,
		"T & P":_cn.op_t_p,
		"M & T & P":_cn.op_t_p,
		"T & PZ":_cn.op_t_pz,
		"M & T & PZ":_cn.op_t_pz,
		"T & P & Z":_cn.op_t_p_z,
		"M & T & P & Z":_cn.op_t_p_z}

MAXPRINT = 50

class basis1d(basis):
	def __init__(self,L,**blocks):
		# getting arguements which are used in basis.
		Nup=blocks.get("Nup")
		kblock=blocks.get("kblock")
		zblock=blocks.get("zblock")
		zAblock=blocks.get("zAblock")
		zBblock=blocks.get("zBblock")
		pblock=blocks.get("pblock")
		pzblock=blocks.get("pzblock")
		a=blocks.get("a")
		self._blocks=blocks
		if a is None: # by default a = 1
			a=1
			blocks["a"]=1

		if type(L) is not int:
			raise TypeError('L must be integer')

		if L>32: raise NotImplementedError('basis can only be constructed for L<=32')


		# checking type, and value of blocks
		if Nup is not None:
			if type(Nup) is not int: raise TypeError('kblock must be integer')
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

		if kblock is not None:
			if type(kblock) is not int: raise TypeError('kblock must be integer')
			kblock = kblock % (L/a)
			blocks["kblock"] = kblock

		if type(a) is not int:
			raise TypeError('a must be integer')



		# checking if a is compatible with L
		if(L%a != 0):
			raise ValueError('L must be interger multiple of lattice spacing a')

		# checking if spin inversion is compatible with Nup and L
		if (type(Nup) is int) and ((type(zblock) is int) or (type(pzblock) is int)):
			if (L % 2) != 0:
				raise ValueError("spin inversion symmetry must be used with even number of sites")
			if Nup != L/2:
				raise ValueError("spin inversion symmetry only reduces the 0 magnetization sector")

		if (type(Nup) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
			raise ValueError("zA and zB incompatible with magnetisation symmetry")

		# checking if ZA/ZB spin inversion is compatible with unit cell of translation symemtry
		if (type(kblock) is int) and ((type(zAblock) is int) or (type(zBblock) is int)):
			if a%2 != 0: # T and ZA (ZB) symemtries do NOT commute
				raise ValueError("unit cell size 'a' must be even")



		self._L=L
		if type(Nup) is int:
			self._Nup=Nup
			self._conserved="M"
			self._Ns=ncr(L,Nup) 
		else:
			self._conserved=""
			self._Ns=2**L

		self._operators = ("availible operators for this basis:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tx: x pauli/spin operator"+
							"\n\ty: y pauli/spin operator"+
							"\n\tz: z pauli/spin operator")

		# allocates memory for number of basis states
		frac = 1.0
		if(L >= 10): frac = 0.6

		if L > 1: L_m = L-1
		else: L_m = 1

		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & P & Z"
			else: self._conserved = "T & P & Z"
			self._blocks["pzblock"] = pblock*zblock

			self._Ns = int(_np.ceil(self._Ns*a*(0.65)/float(L_m))) # estimate fraction of basis needed for sector.

			self._basis=_np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8) # normalisation*sigma
			self._m=_np.empty(self._basis.shape,dtype=_np.int16) #m = mp + (L+1)mz + (L+1)^2c; Anders' paper
			if (type(Nup) is int):
				# arguments get overwritten by _cn.make_...  
				self._Ns = _cn.make_m_t_p_z_basis(L,Nup,pblock,zblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.make_t_p_z_basis(L,pblock,zblock,kblock,a,self._N,self._m,self._basis)
			# cut off extra memory for overestimated state number
			self._N = self._N[:self._Ns]
			self._m = self._m[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(zAblock) is int) and (type(zBblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & ZA & ZB"
			else: self._conserved = "T & ZA & ZB"
			self._blocks["zblock"] = zAblock*zBblock
			
			self._Ns = int(_np.ceil(self._Ns*a*(0.65)/float(L_m))) # estimate fraction of basis needed for sector.

			self._basis=_np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int16)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_t_zA_zB_basis(L,Nup,zAblock,zBblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.make_t_zA_zB_basis(L,zAblock,zBblock,kblock,a,self._N,self._m,self._basis)

			self._N = self._N[:self._Ns]
			self._m = self._m[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(pzblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & PZ"
			else: self._conserved = "T & PZ"
			self._Ns = int(_np.ceil(self._Ns*a*(1.2)/float(L_m))) # estimate fraction of basis needed for sector.

			self._basis=_np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8) #mpz
			if (type(Nup) is int):
				self._Ns = _cn.make_m_t_pz_basis(L,Nup,pzblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.make_t_pz_basis(L,pzblock,kblock,a,self._N,self._m,self._basis)

			self._N = self._N[:self._Ns]
			self._m = self._m[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(pblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & P"
			else: self._conserved = "T & P"
			self._Ns = int(_np.ceil(self._Ns*a*(2)/float(L_m))) # estimate fraction of basis needed for sector.


			self._basis=_np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_t_p_basis(L,Nup,pblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.make_t_p_basis(L,pblock,kblock,a,self._N,self._m,self._basis)

			self._N = self._N[:self._Ns]
			self._m = self._m[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(zblock) is int):
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & Z"
			else: self._conserved = "T & Z"
			self._Ns = int(_np.ceil((frac*self._Ns*a)/float(L_m))) # estimate fraction of basis needed for sector.

			self._basis=_np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_t_z_basis(L,Nup,zblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.make_t_z_basis(L,zblock,kblock,a,self._N,self._m,self._basis)

			self._N = self._N[:self._Ns]
			self._m = self._m[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._m,self._basis,self._L]


		elif (type(kblock) is int) and (type(zAblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & ZA"
			else: self._conserved = "T & ZA"
			self._Ns = int(_np.ceil((frac*self._Ns*a)/float(L_m))) # estimate fraction of basis needed for sector.

			self._basis=_np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_t_zA_basis(L,Nup,zAblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.make_t_zA_basis(L,zAblock,kblock,a,self._N,self._m,self._basis)

			self._N = self._N[:self._Ns]
			self._m = self._m[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(kblock) is int) and (type(zBblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T & ZB"
			else: self._conserved = "T & ZB"
			self._Ns = int(_np.ceil((frac*self._Ns*a)/float(L_m))) # estimate fraction of basis needed for sector.

			self._basis=_np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			self._m=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_t_zB_basis(L,Nup,zBblock,kblock,a,self._N,self._m,self._basis)
			else:
				self._Ns = _cn.make_t_zB_basis(L,zBblock,kblock,a,self._N,self._m,self._basis)

			self._N = self._N[:self._Ns]
			self._m = self._m[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._m,self._basis,self._L]

		elif (type(pblock) is int) and (type(zblock) is int):
			if self._conserved: self._conserved += " & P & Z"
			else: self._conserved += "P & Z"
			self._Ns = int(_np.ceil(self._Ns*0.5*frac)) # estimate fraction of basis needed for sector.
			self._blocks["pzblock"] = pblock*zblock
			
			self._basis = _np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_p_z_basis(L,Nup,pblock,zblock,self._N,self._basis)
			else:
				self._Ns = _cn.make_p_z_basis(L,pblock,zblock,self._N,self._basis)

			self._N = self._N[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._basis,self._L]


		elif (type(zAblock) is int) and (type(zBblock) is int):
			if self._conserved: self._conserved += " & ZA & ZB"
			else: self._conserved += "ZA & ZB"
			self._Ns = int(_np.ceil(self._Ns*0.5*frac)) # estimate fraction of basis needed for sector.
			self._blocks["zblock"] = zAblock*zBblock
			
			self._basis = _np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_zA_zB_basis(L,Nup,self._basis)
			else:
				self._Ns = _cn.make_zA_zB_basis(L,self._basis)

			self._N = self._N[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._basis,self._L]



		elif type(pblock) is int:
			if self._conserved: self._conserved += " & P"
			else: self._conserved = "P"
			self._Ns = int(_np.ceil(self._Ns*frac)) # estimate fraction of basis needed for sector.
			
			self._basis = _np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_p_basis(L,Nup,pblock,self._N,self._basis)
			else:
				self._Ns = _cn.make_p_basis(L,pblock,self._N,self._basis)

			self._N = self._N[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._basis,self._L]



		elif type(zblock) is int:
			if self._conserved: self._conserved += " & Z"
			else: self._conserved += "Z"
			self._Ns = int(_np.ceil(self._Ns*frac)) # estimate fraction of basis needed for sector.

			
			self._basis = _np.empty((self._Ns,),dtype=_np.uint32)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_z_basis(L,Nup,self._basis)
			else:
				self._Ns = _cn.make_z_basis(L,self._basis)

			self._basis = self._basis[:self._Ns]
			self._op_args=[self._basis,self._L]

		elif type(zAblock) is int:
			if self._conserved: self._conserved += " & ZA"
			else: self._conserved += "ZA"
			self._Ns = int(_np.ceil(self._Ns*frac)) # estimate fraction of basis needed for sector.

			
			self._basis = _np.empty((self._Ns,),dtype=_np.uint32)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_zA_basis(L,Nup,self._basis)
			else:
				self._Ns = _cn.make_zA_basis(L,self._basis)

			self._basis = self._basis[:self._Ns]
			self._op_args=[self._basis,self._L]


		elif type(zBblock) is int:
			if self._conserved: self._conserved += " & ZB"
			else: self._conserved += "ZB"
			self._Ns = int(_np.ceil(self._Ns*frac)) # estimate fraction of basis needed for sector.

			
			self._basis = _np.empty((self._Ns,),dtype=_np.uint32)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_zB_basis(L,Nup,self._basis)
			else:
				self._Ns = _cn.make_zB_basis(L,self._basis)

			self._basis = self._basis[:self._Ns]
			self._op_args=[self._basis,self._L]
				
		elif type(pzblock) is int:
			if self._conserved: self._conserved += " & PZ"
			else: self._conserved += "PZ"
			self._Ns = int(_np.ceil(self._Ns*frac)) # estimate fraction of basis needed for sector.
			
			self._basis = _np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty((self._Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_pz_basis(L,Nup,pzblock,self._N,self._basis)
			else:
				self._Ns = _cn.make_pz_basis(L,pzblock,self._N,self._basis)

			self._N = self._N[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._basis,self._L]
	
		elif type(kblock) is int:
			self._k=2*(_np.pi)*a*kblock/L
			if self._conserved: self._conserved += " & T"
			else: self._conserved = "T"
			self._Ns = int(_np.ceil(self._Ns*a*(1.1)/float(L_m))) # estimate fraction of basis needed for sector.

			self._basis=_np.empty((self._Ns,),dtype=_np.uint32)
			self._N=_np.empty(self._basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self._Ns = _cn.make_m_t_basis(L,Nup,kblock,a,self._N,self._basis)
			else:
				self._Ns = _cn.make_t_basis(L,kblock,a,self._N,self._basis)

			self._N = self._N[:self._Ns]
			self._basis = self._basis[:self._Ns]
			self._op_args=[self._N,self._basis,self._L]

		else: 
			if type(Nup) is int:
				self._basis = _cn.make_m_basis(L,Nup,self._Ns)
			else:
				self._basis = _np.arange(0,2**L,1,dtype=_np.uint32)
			self._op_args=[self._basis]




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



	def __str__(self):
		n_digits = int(_np.ceil(_np.log10(self._Ns)))
		temp = "\t{0:"+str(n_digits)+"d}  "+"|{1:0"+str(self._L)+"b}>"
		string = "reference states: \n"
		if self._Ns > MAXPRINT:
			half = MAXPRINT // 2
			t = temp.format(0,0)
			t = t.replace("0"," ").replace("|"," ").replace(">"," ")
			t = "\n"+t[:self._L/2]+":\n"
			string += "\n".join([temp.format(i,b) for i,b in zip(xrange(half),self._basis[:half])])
			string += t
			string += "\n".join([temp.format(i,b) for i,b in zip(xrange(self._Ns-half,self._Ns,1),self._basis[-half:])])
		else:
			string += "\n".join([temp.format(i,b) for i,b in enumerate(self._basis)])

		return string 







	def Op(self,opstr,indx,J,dtype,pauli):
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')
		if not _np.can_cast(J,_np.dtype(dtype)):
			raise TypeError("can't cast coupling to proper dtype")

		if self._Ns <= 0:
			return [],[],[]

		return op[self._conserved](opstr,indx,J,dtype,pauli,*self._op_args,**self._blocks)		




	def get_norms(self,dtype):
		a = self._blocks.get("a")
		kblock = self._blocks.get("kblock")
		pblock = self._blocks.get("pblock")
		zblock = self._blocks.get("zblock")
		pzblock = self._blocks.get("pzblock")


		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			c = _np.empty(self._m.shape,dtype=_np.int8)
			nn = _np.array(c)
			mm = _np.array(c)
			_np.divide(self._m,(self._L+1)**2,c)
			_np.divide(self._m,self._L+1,nn)
			_np.mod(nn,self._L+1,nn)
			_np.mod(self._m,self._L+1,mm)
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
		elif (type(pblock) is int) and (type(zblock) is int):
			norm = _np.array(self._N,dtype=dtype)
		elif (type(pblock) is int):
			norm = _np.array(self._N,dtype=dtype)
		elif (type(pzblock) is int):
			norm = _np.array(self._N,dtype=dtype)
		elif (type(zblock) is int):
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
			shape = (2**self._L,1)
			v0 = v0.reshape((-1,1))
		elif v0.ndim == 2:
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
				_np.divide(c,norms,c)
		else:
			ind_pos = _np.fromiter(xrange(v0.shape[0]),count=v0.shape[0],dtype=_np.int32)
			ind_neg = _np.array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == _np.pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(-dtype(1.0j*k*r))
				_np.divide(c,norms,c)

		if sparse:
			return _get_vec_sparse(v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks)
		else:
			return _get_vec_dense(v0,self._basis,norms,ind_neg,ind_pos,shape,C,self._L,**self._blocks)










	def get_proj(self,dtype):
		if self._Ns <= 0:
			return _np.array([])

		norms = self.get_norms(dtype)

		a = self._blocks.get("a")
		kblock = self._blocks.get("kblock")
		pblock = self._blocks.get("pblock")
		zblock = self._blocks.get("zblock")
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
				_np.divide(c,norms,c)
		else:
			if (type(kblock) is int):
				if ((2*kblock*a) % L != 0) and _np.iscomplexobj(dtype(1.0)):
					raise TypeError("symmetries give complex vector, requested dtype is not complex")

			ind_pos = _np.arange(0,self._Ns,1)
			ind_neg = _np.array([],dtype=_np.int32)
			def C(r,k,c,norms,dtype,*args):
				if k == 0.0:
					c[:] = 1.0
				elif k == _np.pi:
					c[:] = (-1.0)**r
				else:
					c[:] = exp(-dtype(1.0j*k*r))
				_np.divide(c,norms,c)

		return _get_proj_sparse(self._basis,norms,ind_neg,ind_pos,dtype,C,self._L,**self._blocks)





















def _get_vec_dense(v0,basis,norms,ind_neg,ind_pos,shape,C,L,**blocks):
	dtype=_dtypes[v0.dtype.char]

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	pzblock = blocks.get("pzblock")

	c = _np.zeros(basis.shape,dtype=v0.dtype)	
	v = _np.zeros(shape,dtype=v0.dtype)

	bits=" ".join(["{"+str(i)+":0"+str(L)+"b}" for i in xrange(len(basis))])

	if type(kblock) is int:
		k = 2*_np.pi*kblock*a/L
	else:
		k = 0.0
		a = L

	for r in xrange(0,L/a):
#		print bits.format(*basis)
		C(r,k,c,norms,dtype,ind_neg,ind_pos)	
		vc = (v0.T*c).T
		v[basis[ind_pos]] += vc[ind_pos]
		v[basis[ind_neg]] += vc[ind_neg]
		
		
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
		
#	v /= _np.linalg.norm(v,axis=0)
	return v






def _get_vec_sparse(v0,basis,norms,ind_neg,ind_pos,shape,C,L,**blocks):
	dtype=_dtypes[v0.dtype.char]

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	pzblock = blocks.get("pzblock")

	m = shape[1]

	n = ind_neg.shape[0]
	row_neg = _np.broadcast_to(ind_neg,(m,n)).T.ravel()
	col_neg = _np.arange(0,m,1)
	col_neg = _np.broadcast_to(col_neg,(n,m)).ravel()

	n = ind_pos.shape[0]
	row_pos = _np.broadcast_to(ind_pos,(m,n)).T.ravel()
	col_pos = _np.arange(0,m,1)
	col_pos = _np.broadcast_to(col_pos,(n,m)).ravel()



	if type(kblock) is int:
		k = 2*_np.pi*kblock*a/L
	else:
		k = 0.0
		a = L

	c = _np.zeros(basis.shape,dtype=v0.dtype)	
	v = _sm.csr_matrix(shape,dtype=v0.dtype)



	for r in xrange(0,L/a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)

		vc = (v0.T*c).T
		data_pos = vc[ind_pos].flatten()
		data_neg = vc[ind_neg].flatten()
		v = v + _sm.csr_matrix((data_pos,(basis[row_pos],col_pos)),shape,dtype=v.dtype)
		v = v + _sm.csr_matrix((data_neg,(basis[row_neg],col_neg)),shape,dtype=v.dtype)

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

		shiftc(basis,-a,L)

		


#	av = v.multiply(v.conj())
#	norm = av.sum(axis=0)
#	del av
#	_np.sqrt(norm,out=norm)
#	_np.divide(1.0,norm,out=norm)
#	norm = _sm.csr_matrix(norm)
#	v = v.multiply(norm)

	return v













def _get_proj_sparse(basis,norms,ind_neg,ind_pos,dtype,C,L,**blocks):

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	pzblock = blocks.get("pzblock")


	if type(kblock) is int:
		k = 2*_np.pi*kblock*a/L
	else:
		k = 0.0
		a = L

	shape = (2**L,basis.shape[0])

	c = _np.zeros(basis.shape,dtype=dtype)	
	v = _sm.csr_matrix(shape,dtype=dtype)



	for r in xrange(0,L/a):
		C(r,k,c,norms,dtype,ind_neg,ind_pos)
		data_pos = c[ind_pos]
		data_neg = c[ind_neg]
		v = v + _sm.csr_matrix((data_pos,(basis[ind_pos],ind_pos)),shape,dtype=v.dtype)
		v = v + _sm.csr_matrix((data_neg,(basis[ind_neg],ind_neg)),shape,dtype=v.dtype)

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
			v = v + _sm.csr_matrix((data_pos,(basis[row_pos],col)),shape,dtype=v.dtype)
			v = v + _sm.csr_matrix((data_neg,(basis[row_neg],col)),shape,dtype=v.dtype)
			data_pos *= pzblock
			data_neg *= pzblock
			fliplr(basis,L)
			flipall(basis,L)

		shiftc(basis,-a,L)

		


#	av = v.multiply(v.conj())
#	norm = av.sum(axis=0)
#	del av
#	_np.sqrt(norm,out=norm)
#	_np.divide(1.0,norm,out=norm)
#	norm = _sm.csr_matrix(norm)
#	v = v.multiply(norm)

	return v





















def ncr(n, r):
	import operator as _op
# this function calculates n choose r used to find the total number of basis states when the magnetization is conserved.
	r = min(r, n-r)
	if r == 0: return 1
	numer = reduce(_op.mul, xrange(n, n-r, -1))
	denom = reduce(_op.mul, xrange(1, r+1))
	return numer//denom




def fliplr(x,length):
	x1 = array(x)
	x[:] = 0
	for i in xrange(length):
		x2 = array(x1)
		x2 = right_shift(x2,i)
		bitwise_and(x2,1,out=x2)
		left_shift(x2,length-1-i,out=x2)
		x += x2


def flipall(x,length):
	mask = 2**length-1
	invert(x,out=x)
	bitwise_and(x,mask,out=x)
	



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























