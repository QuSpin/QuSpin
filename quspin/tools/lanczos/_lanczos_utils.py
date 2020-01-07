import numpy as _np
from scipy.linalg.blas import get_blas_funcs
from scipy.linalg import eigh_tridiagonal
from copy import deepcopy


__all__ = ["lanczos_full","lanczos_iter","lin_comb_Q"]


def _lanczos_vec_iter_core(A,v0,a,b):
	dtype = _np.result_type(A.dtype,v0.dtype)

	q = v0.astype(dtype,copy=True)

	q_norm = _np.linalg.norm(q)
	if _np.abs(q_norm-1.0) > _np.finfo(dtype).eps:
			_np.divide(q,q_norm,out=q)

	q_view = q[:]
	q_view.setflags(write=0,uic=0)

	yield q_view # return non-writable array

	m = a.size
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

	for i in range(1,m,1):
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



def lanczos_full(A,v0,m,full_ortho=False,out=None,eps=None):
	""" Creates Lanczos basis; diagonalizes Krylov subspace in Lanczos basis.

	Given a hermitian matrix `A` of size :math:`n\\times n` and an integer `m`, the Lanczos algorithm returns 
	
	* an :math:`n\\times m` matrix  :math:`Q`, and 
	* a real symmetric tridiagonal matrix :math:`T=Q^\\dagger A Q` of size :math:`m\\times m`. The matrix :math:`T` can be represented via its eigendecomposition `(E,V)`: :math:`T=V^T\\mathrm{diag}(E)V`. 
	This function computes the triple `(E,V,Q)`. 
 
	Notes
	-----

	* the function allows for full reorthogonalization, see `full_ortho`. The resulting :math:`T` will not neccesarily be tridiagonal.
	* `V` is always real-valued, since :math:`T` is real and symmetric.
	* `A` must have a 'dot' method to perform calculation, 
	* The 'out' argument to pass back the results of the matrix-vector product will be used if the 'dot' function supports this argument.

	Parameters
	-----------
	A : LinearOperator, hamiltonian, np.ndarray, or object with a 'dot' method.
		Python object representing a linear map to compute the Lanczos approximation to the largest eigenvalues/vectors of. Must contain a dot-product method, used as `A.dot(v)`, e.g. `hamiltonian`, `quantum_operator`, `quantum_LinearOperator`, sparse or dense matrix.
	v0 : np.ndarray
		initial vector to start the Lanczos algorithm from.
	m : int
		Number of Lanczos vectors (size of the Krylov subspace)
	full_ortho : bool, optional
		Toggles normalization of the Krylov eigenvectors `V`
	out : np.ndarray, optional
		Array to store the Lanczos vectors in (e.g. `Q`). in memory efficient way.
	eps : float, optional
		Controls the deviation from unit normalization of the Lanczos basis vectors. Default is the data type precision of `v0`. 

	Returns
	--------
	tuple(E,V,Q)
		* E : numpy.ndarray[:]: eigenvalues of Krylov subspace tridiagonal matrix :math:`T`.
		* V : numpy.ndarray[:,:]: eigenvectors of Krylov subspace tridiagonal matrix :math:`T`.
		* Q : numpy.ndarray[:,:]: Lanczos eigenvectors. 

	Examples
	--------

	>>> E, V, Q = lanczos_full(H,v0,20)


	
	"""


	dtype = _np.result_type(A.dtype,v0.dtype)

	if v0.ndim != 1:
		raise ValueError

	n = v0.size

	if out is not None:
		if out.shape != (m,n):
			raise ValueError
		if out.dtype != dtype:
			raise ValueError
		if not out.flags["CARRAY"]:
			raise ValueError

		Q = out
	else:
		Q = _np.zeros((m,n),dtype=dtype)

	Q[0,:] = v0[:]
	v = _np.zeros_like(v0,dtype=dtype)
	r = _np.zeros_like(v0,dtype=dtype)

	b = _np.zeros((m,),dtype=v.real.dtype)
	a = _np.zeros((m,),dtype=v.real.dtype)

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
	for i in range(1,m,1):
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
			m = i
			break


	if full_ortho:
		q,_ = _np.linalg.qr(Q[:i+1].T)

		Q[:i+1,:] = q.T[...]

		h = _np.zeros((m,m),dtype=a.dtype)

		for i in range(m):
			if use_out:
				A.dot(Q[i,:],out=r) # call if operator supports 'out' argument
			else:
				r[:] = A.dot(Q[i,:])

			_np.conj(r,out=r)
			h[i,i:] = _np.dot(Q[i:,:],r).real

		E,V = _np.linalg.eigh(h,UPLO="U")

	else:
		E,V = eigh_tridiagonal(a[:m],b[:m-1])


	return E,V,Q[:m]




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


def lanczos_iter(A,v0,m,return_vec_iter=True,copy_v0=True,copy_A=False,eps=None):
	""" Creates generator for Lanczos basis; diagonalizes Krylov subspace in Lanczos basis.

	Given a hermitian matrix `A` of size :math:`n\\times n` and an integer `m`, the Lanczos algorithm returns 
	
	* an :math:`n\\times m` matrix  :math:`Q`, and 
	* a real symmetric tridiagonal matrix :math:`T=Q^\\dagger A Q` of size :math:`m\\times m`. The matrix :math:`T` can be represented via its eigendecomposition `(E,V)`: :math:`T=V^T\\mathrm{diag}(E)V`. 
	This function computes the triple `(E,V,Q)`. 
 
 	Notes
 	-----

 	* this function is useful to minimize any memory requirements in the calculation of the Lanczos basis. 
 	* `V` is always real-valued, since :math:`T` is real and symmetric.

	Parameters
	-----------
	A : LinearOperator, hamiltonian, np.ndarray, etc.
		Python object representing a linear map to compute the Lanczos approximation to the largest eigenvalues/vectors of. Must contain a dot-product method, used as `A.dot(v)`, e.g. `hamiltonian`, `quantum_operator`, `quantum_LinearOperator`, sparse or dense matrix.
	v0 : np.ndarray
		initial vector to start the Lanczos algorithm from.
	m : int
		Number of Lanczos vectors (size of the Krylov subspace)
	return_vec_iter : bool, optional
		Toggles whether or not to return the Lanczos basis iterator.
	copy_v0 : bool, optional
		Whether or not to produce of copy of initial vector `v0`.
	copy_A : bool, optional
		Whether or not to produce of copy of linear operator `A`.
	eps : float, optional
		Controls the deviation from unit normalization of the Lanczos basis vectors. Default is the data type precision of `v0`. 

	Returns
	--------
	tuple(E,V) if return_vec_iter=False.
	tuple(E,V,Q_iterator) if return_vec_iter=True.
		* E : numpy.ndarray[:]: eigenvalues of Krylov subspace tridiagonal matrix :math:`T`.
		* V : numpy.ndarray[:,:]: eigenvectors of Krylov subspace tridiagonal matrix :math:`T`.
		* Q_iterator : generator[numpy.ndarray[:,:]]: generator to create the Lanczos eigenvectors on-the-fly. 

	Examples
	--------

	>>> E, V, Q_iterator = lanczos_iter(H,v0,20)

	"""

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

	b = _np.zeros((m,),dtype=q.real.dtype)
	a = _np.zeros((m,),dtype=q.real.dtype)

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
	for i in range(1,m,1):
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


def _get_first_lv_iter(r,Q_iter):
	yield r
	for Q in Q_iter:
		yield Q


def _get_first_lv(Q_iter):
	r = next(Q_iter)
	return r,_get_first_lv_iter(r,Q_iter)


# I suggest the name `lv_average()` or `lv_linearcomb` or `linear_combine_Q()` instead of `lin_comb_Q()`
def lin_comb_Q(weights,Q,out=None):
	""" Computes a (weighted) linear combination of the Lanczos basis vectors. 

	
	Parameters
	-----------
	weights : array_like
		list of weights to compute the linear combination of Lanczos basis vectors with.
	Q : np.ndarray, generator
		Lanczos basis vectors or a generator for the Lanczos basis.
	out : np.ndarray, optional
		Array to store the result in.
	
	Returns
	--------
	np.ndarray
		Linear combination of Lanczos basis vectors. 

	Examples
	--------

	>>> result = lin_comb_Q(weights,Q)

	"""


	weights = _np.asanyarray(weights)

	if isinstance(Q,_np.ndarray):
		Q_iter = iter(Q[:])
	else:
		Q_iter = iter(Q)

	q = next(Q_iter)

	dtype = _np.result_type(q.dtype,weights.dtype)

	if out is not None:
		if out.shape != q.shape:
			raise ValueError("'out' must have same shape as a Lanczos vector.")
		if out.dtype != dtype:
			raise ValueError("dtype for 'out' does not match dtype for result.")
		if not out.flags["CARRAY"]:
			raise ValueError("'out' must be a contiguous array.")
	else:
		out = _np.zeros(q.shape,dtype=dtype)

	# get function : y <- y + a * x
	axpy = get_blas_funcs('axpy', arrays=(out,q))	
	
	n = q.size

	_np.multiply(q,weights[0],out=out)
	for weight,q in zip(weights[1:],Q_iter):
		axpy(q,out,n,weight)

	return out
