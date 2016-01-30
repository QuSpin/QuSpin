import constructors as _cn
import numpy as _np

# this is how we encode which fortran function to call when calculating 
# the action of operator string
op={"":_cn.op,
		"M":_cn.op_m,
		"Z":_cn.op_z,
		"M & Z":_cn.op_z,
		"P":_cn.op_p,
		"M & P":_cn.op_p,
		"PZ":_cn.op_pz,
		"M & PZ":_cn.op_pz,
		"P & Z":_cn.op_p_z,
		"M & P & Z":_cn.op_p_z,
		"T":_cn.op_t,
		"M & T":_cn.op_t,
		"T & Z":_cn.op_t_z,
		"M & T & Z":_cn.op_t_z,
		"T & P":_cn.op_t_p,
		"M & T & P":_cn.op_t_p,
		"T & PZ":_cn.op_t_pz,
		"M & T & PZ":_cn.op_t_pz,
		"T & P & Z":_cn.op_t_p_z,
		"M & T & P & Z":_cn.op_t_p_z}




class BasisError(Exception):
	# this class defines an exception which can be raised whenever there is some sort of error which we can
	# see will obviously break the code. 
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message


def ncr(n, r):
	import operator as _op
# this function calculates n choose r used to find the total number of basis states when the magnetization is conserved.
	r = min(r, n-r)
	if r == 0: return 1
	numer = reduce(_op.mul, xrange(n, n-r, -1))
	denom = reduce(_op.mul, xrange(1, r+1))
	return numer//denom


class basis1d:
	def __init__(self,L,**blocks):
		# getting arguements which are used in basis.
		Nup=blocks.get("Nup")
		kblock=blocks.get("kblock")
		zblock=blocks.get("zblock")
		pblock=blocks.get("pblock")
		pzblock=blocks.get("pzblock")
		a=blocks.get("a")
		self.blocks=blocks
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

		if pzblock is not None:
			if type(pzblock) is not int: raise TypeError('pzblock must be integer')
			if abs(pzblock) != 1: raise ValueError("pzblock must be +/- 1")

		if kblock is not None:
			if type(kblock) is not int: raise TypeError('kblock must be integer')

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



		self.L=L
		if type(Nup) is int:
			self.Nup=Nup
			self.conserved="M"
			self.Ns=ncr(L,Nup) 
		else:
			self.conserved=""
			self.Ns=2**L

		if(L >= 10): frac = 0.6
		else: frac = 0.7

		if L > 1: L_m = L-1
		else: L_m = 1

		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & P & Z"
			else: self.conserved = "T & P & Z"
			self.Ns = int(_np.ceil(self.Ns*a*(0.55)/float(L_m))) # estimate fraction of basis needed for sector.

			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			self.m=_np.empty(self.basis.shape,dtype=_np.int16)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_t_p_z_basis(L,Nup,pblock,zblock,kblock,a,self.N,self.m,self.basis)
			else:
				self.Ns = _cn.make_t_p_z_basis(L,pblock,zblock,kblock,a,self.N,self.m,self.basis)

			self.N = self.N[:self.Ns]
			self.m = self.m[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.op_args=[self.N,self.m,self.basis,self.L]

		elif (type(kblock) is int) and (type(pzblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & PZ"
			else: self.conserved = "T & PZ"
			self.Ns = int(_np.ceil(self.Ns*a*(1.1)/float(L_m))) # estimate fraction of basis needed for sector.

			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			self.m=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_t_pz_basis(L,Nup,pzblock,kblock,a,self.N,self.m,self.basis)
			else:
				self.Ns = _cn.make_t_pz_basis(L,pzblock,kblock,a,self.N,self.m,self.basis)

			self.N = self.N[:self.Ns]
			self.m = self.m[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.op_args=[self.N,self.m,self.basis,self.L]

		elif (type(kblock) is int) and (type(pblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & P"
			else: self.conserved = "T & P"
			self.Ns = int(_np.ceil(self.Ns*a*(1.1)/float(L_m))) # estimate fraction of basis needed for sector.


			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			self.m=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_t_p_basis(L,Nup,pblock,kblock,a,self.N,self.m,self.basis)
			else:
				self.Ns = _cn.make_t_p_basis(L,pblock,kblock,a,self.N,self.m,self.basis)

			self.N = self.N[:self.Ns]
			self.m = self.m[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.op_args=[self.N,self.m,self.basis,self.L]

		elif (type(kblock) is int) and (type(zblock) is int):
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T & Z"
			else: self.conserved = "T & Z"
			self.Ns = int(_np.ceil((frac*self.Ns*a)/float(L_m))) # estimate fraction of basis needed for sector.

			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			self.m=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_t_z_basis(L,Nup,zblock,kblock,a,self.N,self.m,self.basis)
			else:
				self.Ns = _cn.make_t_z_basis(L,zblock,kblock,a,self.N,self.m,self.basis)

			self.N = self.N[:self.Ns]
			self.m = self.m[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.op_args=[self.N,self.m,self.basis,self.L]

		elif (type(pblock) is int) and (type(zblock) is int):
			if self.conserved: self.conserved += " & P & Z"
			else: self.conserved += "P & Z"
			self.Ns = int(_np.ceil(self.Ns*0.5*frac)) # estimate fraction of basis needed for sector.
			
			self.basis = _np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty((self.Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_p_z_basis(L,Nup,pblock,zblock,self.N,self.basis)
			else:
				self.Ns = _cn.make_p_z_basis(L,pblock,zblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.op_args=[self.N,self.basis,self.L]



		elif type(pblock) is int:
			if self.conserved: self.conserved += " & P"
			else: self.conserved = "P"
			self.Ns = int(_np.ceil(self.Ns*frac)) # estimate fraction of basis needed for sector.
			
			self.basis = _np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty((self.Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_p_basis(L,Nup,pblock,self.N,self.basis)
			else:
				self.Ns = _cn.make_p_basis(L,pblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.op_args=[self.N,self.basis,self.L]



		elif type(zblock) is int:
			if self.conserved: self.conserved += " & Z"
			else: self.conserved += "Z"
			self.Ns = int(_np.ceil(self.Ns*frac)) # estimate fraction of basis needed for sector.

			
			self.basis = _np.empty((self.Ns,),dtype=_np.int32)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_z_basis(L,Nup,self.basis)
			else:
				self.Ns = _cn.make_z_basis(L,self.basis)

			self.basis = self.basis[:self.Ns]
			self.op_args=[self.basis,self.L]
				
		elif type(pzblock) is int:
			if self.conserved: self.conserved += " & PZ"
			else: self.conserved += "PZ"
			self.Ns = int(_np.ceil(self.Ns*frac)) # estimate fraction of basis needed for sector.
			
			self.basis = _np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty((self.Ns,),dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_pz_basis(L,Nup,pzblock,self.N,self.basis)
			else:
				self.Ns = _cn.make_pz_basis(L,pzblock,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.op_args=[self.N,self.basis,self.L]
	
		elif type(kblock) is int:
			self.k=2*(_np.pi)*a*kblock/L
			if self.conserved: self.conserved += " & T"
			else: self.conserved = "T"
			self.Ns = int(_np.ceil(self.Ns*a*(1.1)/float(L_m))) # estimate fraction of basis needed for sector.

			self.basis=_np.empty((self.Ns,),dtype=_np.int32)
			self.N=_np.empty(self.basis.shape,dtype=_np.int8)
			if (type(Nup) is int):
				self.Ns = _cn.make_m_t_basis(L,Nup,kblock,a,self.N,self.basis)
			else:
				self.Ns = _cn.make_t_basis(L,kblock,a,self.N,self.basis)

			self.N = self.N[:self.Ns]
			self.basis = self.basis[:self.Ns]
			self.op_args=[self.N,self.basis,self.L]

		else: 
			if type(Nup) is int:
				s0=sum([2**i for i in xrange(0,Nup)])
				self.basis=_cn.make_m_basis(s0,self.Ns)
			else:
				self.basis=_np.array(xrange(self.Ns),dtype=_np.int32)
			self.op_args=[self.basis]



	def Op(self,opstr,indx,J,dtype,pauli):
		if len(opstr) != len(indx):
			raise ValueError('length of opstr does not match length of indx')
		if not _np.can_cast(J,_np.dtype(dtype)):
			raise TypeError("can't cast coupling to proper dtype")

		return  op[self.conserved](opstr,indx,J,dtype,pauli,*self.op_args,**self.blocks)		





















