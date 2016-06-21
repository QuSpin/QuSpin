from ..base import basis
from ..basis1d import constructors as _cn

from photon import photon
from ..basis1d import basis1d


import numpy as _np
from numpy import array,asarray
from numpy import right_shift,left_shift,invert,bitwise_and,bitwise_or
from numpy import cos,sin,exp,pi
from numpy.linalg import norm

import scipy.sparse as _sm
from scipy.special import hyp2f1

# this is how we encode which fortran function to call when calculating 
# the action of operator string


_dtypes={"f":_np.float32,"d":_np.float64,"F":_np.complex64,"D":_np.complex128}
if hasattr(_np,"float128"): _dtypes["g"]=_np.float128
if hasattr(_np,"complex256"): _dtypes["G"]=_np.complex256



op={"M":_cn.op_m, "M & P":_cn.op_p, "M & T":_cn.op_t, "M & T & P":_cn.op_t_p}

MAXPRINT = 50

class spin_photon(basis):
	def __init__(self,L,Ntot,n_ph=0,**blocks):
		# getting arguments which are used in basis.
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

		# handled by underlying class
		#if L>32: raise NotImplementedError('basis can only be constructed for L<=32')

		# maybe allow for this to happen, shouldn't change the size of the basis overall.
		#if Ntot>L: raise TypeError('basis can only be constructed for Ntot<=L') # otherwise, shift photon mode energies
 
		# checking type, and value of blocks 
		if Nup is not None:
			raise TypeError('Hamiltonian does not feature magnetisation symmetry')

		if (zblock is not None) or (zAblock is not None) or (zBblock is not None) or (pzblock is not None):
			raise TypeError('Hamiltonian does not feature spin inversion symmetry of any kind')

		# have this handled by basis1d class
		"""
		if pblock is not None:
			if type(pblock) is not int: raise TypeError('pblock must be integer')
			if abs(pblock) != 1: raise ValueError("pblock must be +/- 1")


		if kblock is not None:
			if type(kblock) is not int: raise TypeError('kblock must be integer')
			kblock = kblock % (L/a)
			blocks["kblock"] = kblock

		if type(a) is not int:
			raise TypeError('a must be integer')



		# checking if a is compatible with L
		if(L%a != 0):
			raise ValueError('L must be interger multiple of lattice spacing a')
		"""



		self._L=L
		self._Ntot = Ntot
		self.n_ph = n_ph
		#self._conserved="Ntot"
		# number of states n particle-conserving case: sum_{i=0}^Ntot ncr(L,i)
		def num_states(L,Ntot):
			return int( 2**L - ncr(L,Ntot+1)*hyp2f1(1, Ntot+1-L, 1+Ntot+1, -1) )
		self._Ns = num_states(L,Ntot)

		
		self._operators = ("availible operators for this basis:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tx: x pauli/spin operator"+
							"\n\ty: y pauli/spin operator"+
							"\n\tz: z pauli/spin operator"+
							"\n\tn: photon number operator")


		spin_basis = basis1d(L,Nup=0,**blocks)
		# preallocate memory for class objects

		#  dtype = spin_basis._basis.dtype # use same dtype as the basis1d class.
		#  self._basis = _np.empty((self._Ns,),dtype=dtype)

		self._basis = _np.empty((self._Ns,),dtype=_np.uint32)
		self._basis[0] = spin_basis._basis

		# dtype = _np.min_scalar_type(n_ph) #check the smalleest type which will fit this integer inside it
		# self._n = _np.empty((self._Ns,),dtype=dtype) #vector with photon occupations 

		self._n = _np.empty((self._Ns,),dtype=_np.uint32) #vector with photon occupations
		self._n[0] = self._Ntot 
		if hasattr(spin_basis,"_N"):
			# dtype = spin_basis._N.dtype
			# self._N = _np.empty((self._Ns,),dtype=dtype)

			self._N = _np.empty((self._Ns,),dtype=_np.uint8) 
			self._N[0] = spin_basis._N
		if hasattr(spin_basis,"_m"):
			# dtype = spin_basis._m.dtype
			# self._N = _np.empty((self._Ns,),dtype=dtype)

			self._m = _np.empty((self._Ns,),dtype=_np.uint8)
			self._m[0] = spin_basis._m

		# build the total particle-conserving spin_photon basis
		Nup_max = min(self._Ntot+1,self._L+1) #stop at Nup=L if N_tot > L
		for Nup in xrange(1,Nup_max,1):

			basis_tmp = basis1d(L,Nup=Nup,**blocks)
			
			self._basis[num_states(L,Nup-1):num_states(L,Nup)] = basis_tmp._basis
			self._n[num_states(L,Nup-1):num_states(L,Nup)] = _np.full(basis_tmp.Ns, self._Ntot-Nup, dtype =_np.int8, order='C')
			if hasattr(self, "_N"):
				self._N[num_states(L,Nup-1):num_states(L,Nup)] = basis_tmp._N
			if hasattr(self,"_m"):
				self._n[num_states(L,Nup-1):num_states(L,Nup)] = basis_tmp._n

		self._conserved = basis_tmp._conserved
		# sort basis
		inds = _np.argsort(self._basis)
		self._basis = self._basis[inds]
		self._n = self._n[inds]

		if hasattr(self, "_N") and hasattr(self, "_m"):
			#self._conserved = basis_tmp._conserved.replace("M & ", "")
			self._N = self._N[inds]
			self._m = self._m[inds]
			self._op_args=[self._N,self._m,self._basis,self._L]
		elif hasattr(self, "_N"):
			#self._conserved = basis_tmp._conserved.replace("M & ", "")
			self._N = self._N[inds]
			self._op_args=[self._N,self._basis,self._L]
		else:
			#self._conserved = basis_tmp._conserved.replace("M", "")
			self._op_args=[self._basis]

		del basis_tmp

		self.ho_basis = photon(self._Ntot,n_ph=self.n_ph)
		print self._basis
		print self.ho_basis.basis


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
		temp = "\t{0:"+str(n_digits)+"d}  "+"|{1:0"+str(self._L)+"b}>|"
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

		if not _np.can_cast(J,_np.dtype(dtype)):
			raise TypeError("can't cast coupling to proper dtype")

		if self._Ns <= 0:
			return [],[],[]

		# read off spin and photon operators
		n=opstr.count("|")
		if n > 1: 
			raise ValueError("only one '|' charactor allowed")
		i = opstr.index("|")
		indx1 = indx[:i]
		indx2 = indx[i:]

		opstr1,opstr2=opstr.split("|")


		# calculates matrix elements of spin and photon basis
		
		if self._Ns <= 0:
			return [],[],[]

		ME_ph,row_ph,col_ph =  self.ho_basis.Op(dtype,J,opstr2)
		ME_s, row_s, col_s  = op[self._conserved](opstr1,indx1,1,dtype,pauli,*self._op_args,**self._blocks)
		# calculate total matrix element
		ME = ME_s*ME_ph[self._n[row_s]]

		del ME_ph, row_ph, col_ph
		del ME_s

		return ME, row_s, col_s	




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
	elif r < 0: return 0 
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























