import numpy as _np
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg import eigh_tridiagonal
from copy import deepcopy


__all__ = ["lanczos_full","lanczos_iter","combine_lv"]


def _lanczos_vec_iter_core(A,v0,a,b):
	dtype = _np.result_type(A.dtype,v0.dtype)

	q = v0.astype(dtype,copy=True)

	q_norm = _np.linalg.norm(q)
	if _np.abs(q_norm-1.0) > _np.finfo(dtype).eps:
			_np.divide(q,q_norm,out=q)

	q_view = q[:]
	q_view.setflags(write=0,uic=0)

	yield q_view # return non-writable array

	nv = a.size
	n = q.size
	axpy = get_blas_funcs('axpy', arrays=(q,))

	v = _np.zeros_like(v0,dtype=dtype)
	r = _np.zeros_like(v0,dtype=dtype)

	try:
		A.dot(q,out=r)
		use_out = True
	except TypeError:
		r[:] = A.dot(q)
		use_out = False

	axpy(q,r,n,-a[0])

	for i in range(1,nv,1):
		v[:] = q[:]

		_np.divide(r,b[i-1],out=q)

		yield q_view # return non-writable array

		if use_out:
			A.dot(q,out=r)
		else:
			r[:] = A.dot(q)

		axpy(v,r,n,-b[i-1])
		axpy(q,r,n,-a[i])


class _lanczos_vec_iter(object):
	def __init__(self,A,v0,a,b):
		self._A = A
		self._v0 = v0
		self._a = a
		self._b = b

	def __iter__(self):
		return _lanczos_vec_iter_core(self._A,self._v0,self._a,self._b)



def lanczos_full(A,v0,nv,full_ortho=False,out=None,eps=None):
	dtype = _np.result_type(A.dtype,v0.dtype)

	if v0.ndim != 1:
		raise ValueError

	n = v0.size

	if out is not None:
		if out.shape != (nv,n):
			raise ValueError
		if out.dtype != dtype:
			raise ValueError
		if not out.flags["CARRAY"]:
			raise ValueError

		Q = out
	else:
		Q = _np.zeros((nv,n),dtype=dtype)

	Q[0,:] = v0[:]
	v = _np.zeros_like(v0,dtype=dtype)
	r = _np.zeros_like(v0,dtype=dtype)

	b = _np.zeros((nv,),dtype=v.real.dtype)
	a = _np.zeros((nv,),dtype=v.real.dtype)

	# get function : y <- y + a * x
	axpy = get_blas_funcs('axpy', arrays=(r, v))

	if eps is None:
		eps = _np.finfo(dtype).eps

	q_norm = _np.linalg.norm(Q[0,:])

	if _np.abs(q_norm-1.0) > eps:
		_np.divide(Q[0,:],q_norm,out=Q[0,:])
	
	try:
		A.dot(Q[0,:],out=r) # call if operator supports 'out' argument
		use_out  = True
	except TypeError:
		r[:] = A.dot(Q[0,:])
		use_out = False

	a[0] = _np.vdot(Q[0,:],r).real

	axpy(Q[0,:],r,n,-a[0])
	b[0] = _np.linalg.norm(r)

	i = 0
	for i in range(1,nv,1):
		v[:] = Q[i-1,:]

		_np.divide(r,b[i-1],out=Q[i,:])

		if use_out:
			A.dot(Q[i,:],out=r) # call if operator supports 'out' argument
		else:
			r[:] = A.dot(Q[i,:])

		axpy(v,r,n,-b[i-1])

		a[i] = _np.vdot(Q[i,:],r).real
		axpy(Q[i,:],r,n,-a[i])

		b[i] = _np.linalg.norm(r)
		if b[i] < eps:
			nv = i
			break


	if full_ortho:
		q,_ = _np.linalg.qr(Q[:i+1].T)

		Q[:i+1,:] = q.T[...]

		h = _np.zeros((nv,nv),dtype=a.dtype)

		for i in range(nv):
			if use_out:
				A.dot(Q[i,:],out=r) # call if operator supports 'out' argument
			else:
				r[:] = A.dot(Q[i,:])

			_np.conj(r,out=r)
			h[i,i:] = _np.dot(Q[i:,:],r).real

		E,V = _np.linalg.eigh(h,UPLO="U")

	else:
		E,V = eigh_tridiagonal(a[:nv],b[:nv-1])


	return E,V,Q[:nv]




# def lanczos_vec_iter(A,v0,a,b,copy_v0=True,copy_A=False):
# 	if copy_v0:
# 		v0 = v0.copy()

# 	if copy_A:
# 		A = deepcopy(A)

# 	if v0.ndim != 1:
# 		raise ValueError

# 	if a.ndim != 1:
# 		raise ValueError

# 	if b.ndim != 1:
# 		raise ValueError

# 	if a.size != b.size+1:
# 		raise ValueError

# 	return _lanczos_vec_iter(A,v0,a.copy(),b.copy())


def lanczos_iter(A,v0,nv,return_vec_iter=True,copy_v0=True,copy_A=False,eps=None):
	if copy_v0 and return_vec_iter:
		v0 = v0.copy()

	if copy_A and return_vec_iter:
		A = deepcopy(A)

	if v0.ndim != 1:
		raise ValueError("expecting array with ndim=1 for initial Lanczos vector.")

	dtype = _np.result_type(A.dtype,v0.dtype)

	q = v0.astype(dtype,copy=True)
	v = _np.zeros_like(v0,dtype=dtype)
	r = _np.zeros_like(v0,dtype=dtype)

	n = v0.size

	b = _np.zeros((nv,),dtype=q.real.dtype)
	a = _np.zeros((nv,),dtype=q.real.dtype)

	# get function : y <- y + a * x
	axpy = get_blas_funcs('axpy', arrays=(q, v))

	if eps is None:
		eps = _np.finfo(dtype).eps

	q_norm = _np.linalg.norm(q)

	if _np.abs(q_norm-1.0) > eps:
		_np.divide(q,q_norm,out=q)

	try:
		A.dot(q,out=r) # call if operator supports 'out' argument
		use_out = True
	except TypeError:
		r[:] = A.dot(q)
		use_out = False

	a[0] = _np.vdot(q,r).real
	axpy(q,r,n,-a[0])
	b[0] = _np.linalg.norm(r)

	i = 0
	for i in range(1,nv,1):
		v[:] = q[:]

		_np.divide(r,b[i-1],out=q)

		if use_out:
			A.dot(q,out=r) # call if operator supports 'out' argument
		else:
			r[:] = A.dot(q)

		axpy(v,r,n,-b[i-1])
		a[i] = _np.vdot(q,r).real
		axpy(q,r,n,-a[i])

		b[i] = _np.linalg.norm(r)
		if b[i] < eps:
			break

	a = a[:i+1].copy()
	b = b[:i].copy()
	del q,r,v

	E,V = eigh_tridiagonal(a,b)

	if return_vec_iter:
		return E,V,_lanczos_vec_iter(A,v0,a.copy(),b.copy())
	else:
		return E,V


def _get_first_lv_iter(r,lv_iter):
	yield r
	for lv in lv_iter:
		yield lv


def _get_first_lv(lv_iter):
	r = next(lv_iter)
	return r,_get_first_lv_iter(r,lv_iter)


def combine_lv(c,lv,out=None):
	c = _np.asanyarray(c)

	if isinstance(lv,_np.ndarray):
		lv_iter = iter(lv[:])
	else:
		lv_iter = iter(lv)

	vl = next(lv_iter)

	dtype = _np.result_type(vl.dtype,c.dtype)

	if out is not None:
		if out.shape != vl.shape:
			raise ValueError("'out' must have same shape as a Lanczos vector.")
		if out.dtype != dtype:
			raise ValueError("dtype for 'out' does not match dtype for result.")
		if not out.flags["CARRAY"]:
			raise ValueError("'out' must be a contiguous array.")
	else:
		out = _np.zeros(vl.shape,dtype=dtype)

	# get function : y <- y + a * x
	axpy = get_blas_funcs('axpy', arrays=(out,vl))	
	
	n = vl.size

	_np.multiply(vl,c[0],out=out)
	for cc,vl in zip(c[1:],lv_iter):
		axpy(vl,out,n,cc)

	return out
