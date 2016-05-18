import constructors as _cn
import numpy as _np
from numpy import array,asarray
from numpy import right_shift,left_shift,invert,bitwise_and,bitwise_or
from numpy import cos,sin,exp,pi
from numpy.linalg import norm

import scipy.sparse as _sm

# this is how we encode which fortran function to call when calculating 
# the action of operator string
dtypes={"f":_np.float32,
				"F":_np.complex64,
				"d":_np.float64,
				"D":_np.complex128}

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
			self.blocks["pzblock"] = pblock*zblock

			self.Ns = int(_np.ceil(self.Ns*a*(0.65)/float(L_m))) # estimate fraction of basis needed for sector.

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
			self.blocks["pzblock"] = pblock*zblock
			
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

		return op[self.conserved](opstr,indx,J,dtype,pauli,*self.op_args,**self.blocks)		




	def get_norms(self,dtype):
		a = self.blocks.get("a")
		kblock = self.blocks.get("kblock")
		pblock = self.blocks.get("pblock")
		zblock = self.blocks.get("zblock")
		pzblock = self.blocks.get("pzblock")


		if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
			c = _np.empty(self.m.shape,dtype=_np.int8)
			nn = _np.array(c)
			mm = _np.array(c)
			_np.divide(self.m,(self.L+1)**2,c)
			_np.divide(self.m,self.L+1,nn)
			_np.mod(nn,self.L+1,nn)
			_np.mod(self.m,self.L+1,mm)
			if _np.abs(_np.sin(self.k)) < 1.0/self.L:
				norm = _np.full(self.basis.shape,4*(self.L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self.basis.shape,2*(self.L/a)**2,dtype=dtype)
			norm *= _np.sign(self.N)
			norm /= self.N
			# c = 2
			mask = (c == 2)
			norm[mask] *= (1.0 + _np.sign(self.N[mask])*pblock*_np.cos(self.k*mm[mask]))
			# c = 3
			mask = (c == 3)
			norm[mask] *= (1.0 + zblock*_np.cos(self.k*nn[mask]))	
			# c = 4
			mask = (c == 4)
			norm[mask] *= (1.0 + _np.sign(self.N[mask])*pzblock*_np.cos(self.k*mm[mask]))	
			# c = 5
			mask = (c == 5)
			norm[mask] *= (1.0 + _np.sign(self.N[mask])*pblock*_np.cos(self.k*mm[mask]))
			norm[mask] *= (1.0 + zblock*_np.cos(self.k*nn[mask]))	
			del mask
		elif (type(kblock) is int) and (type(pblock) is int):
			if _np.abs(_np.sin(self.k)) < 1.0/self.L:
				norm = _np.full(self.basis.shape,2*(self.L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self.basis.shape,(self.L/a)**2,dtype=dtype)
			norm *= _np.sign(self.N)
			norm /= self.N
			# m >= 0 
			mask = (self.m >= 0)
			norm[mask] *= (1.0 + _np.sign(self.N[mask])*pblock*_np.cos(self.k*self.m[mask]))
			del mask
		elif (type(kblock) is int) and (type(pzblock) is int):
			if _np.abs(_np.sin(self.k)) < 1.0/self.L:
				norm = _np.full(self.basis.shape,2*(self.L/a)**2,dtype=dtype)
			else:
				norm = _np.full(self.basis.shape,(self.L/a)**2,dtype=dtype)
			norm *= _np.sign(self.N)
			norm /= self.N
			# m >= 0 
			mask = (self.m >= 0)
			norm[mask] *= (1.0 + _np.sign(self.N[mask])*pzblock*_np.cos(self.k*self.m[mask]))
			del mask
		elif (type(kblock) is int) and (type(zblock) is int):
			norm = _np.full(self.basis.shape,2*(self.L/a)**2,dtype=dtype)
			norm /= self.N
			# m >= 0 
			mask = (self.m >= 0)
			norm[mask] *= (1.0 + zblock*_np.cos(self.k*self.m[mask]))
			del mask
		elif (type(pblock) is int) and (type(zblock) is int):
			norm = _np.array(self.N,dtype=dtype)
		elif (type(pblock) is int):
			norm = _np.array(self.N,dtype=dtype)
		elif (type(pzblock) is int):
			norm = _np.array(self.N,dtype=dtype)
		elif (type(zblock) is int):
			norm = _np.full(self.basis.shape,2.0,dtype=dtype)
		elif (type(kblock) is int):
			norm = _np.full(self.basis.shape,(self.L/a)**2,dtype=dtype)
			norm /= self.N
		else:
			norm = _np.ones(self.basis.shape,dtype=dtype)
	
		_np.sqrt(norm,norm)

		return norm




	def get_vec(self,v0,sparse=True):
		if self.Ns <= 0:
			return _np.array([])
		if v0.ndim == 1:
			shape = (2**self.L,1)
			v0 = v0.reshape((-1,1))
		elif v0.ndim == 2:
			shape = (2**self.L,v0.shape[1])
		else:
			raise ValueError("excpecting v0 to have ndim at most 2")


		norms = self.get_norms(v0.dtype)

		a = self.blocks.get("a")
		kblock = self.blocks.get("kblock")
		pblock = self.blocks.get("pblock")
		zblock = self.blocks.get("zblock")
		pzblock = self.blocks.get("pzblock")


		if (type(kblock) is int) and ((type(pblock) is int) or (type(pzblock) is int)):
			mask = (self.N < 0)
			ind_neg, = _np.nonzero(mask)
			mask = (self.N > 0)
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
			return _get_vec_sparse(v0,self.basis,norms,ind_neg,ind_pos,shape,C,self.L,**self.blocks)
		else:
			return _get_vec_dense(v0,self.basis,norms,ind_neg,ind_pos,shape,C,self.L,**self.blocks)












def _get_vec_dense(v0,basis,norms,ind_neg,ind_pos,shape,C,L,**blocks):
	dtype=dtypes[v0.dtype.char]

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
	dtype=dtypes[v0.dtype.char]

	a = blocks.get("a")
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	pzblock = blocks.get("pzblock")

	m = shape[1]

	n = ind_neg.shape[0]
	row_neg = _np.broadcast_to(ind_neg,(m,n)).T.flatten()
	col_neg = _np.fromiter(xrange(m),count=m,dtype=_np.int32)
	col_neg = _np.broadcast_to(col_neg,(n,m)).flatten()

	n = ind_pos.shape[0]
	row_pos = _np.broadcast_to(ind_pos,(m,n)).T.flatten()
	col_pos = _np.fromiter(xrange(m),count=m,dtype=_np.int32)
	col_pos = _np.broadcast_to(col_pos,(n,m)).flatten()



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
		data_pos = (v0.T*c).T[ind_pos].flatten()
		data_neg = (v0.T*c).T[ind_neg].flatten()
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























