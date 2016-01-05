import numpy as np

try:
	from . import build_krylov
except ValueError:
	import build_krylov


from scipy.sparse import csr_matrix,issparse,dia_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import norm as sp_norm
import warnings


__all__=['expm_krylov']



_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}


def _expm_krylov_vector(A,v0,hermitian,tol,z,dtype):

	char = _type_conv[dtype.char]
	if hermitian:


		ncv_max = 10000

		if abs(z)*sp_norm(A) == 0:
			return v0

		A_data_copy = abs(z) * np.array(A.data)

		lanczos_op = build_krylov.__dict__[char + '_lanczos_op']
		alpha,beta,ncv,stat = lanczos_op(A_data_copy,A.indices,A.indptr,v0,tol,ncv_max)

		if stat < 0:
			raise KrylovError('maximum number of lanczos vectors reached')			

		alpha=alpha[:ncv]
		beta=beta[:ncv-1]

		data = np.vstack( ( np.append([0.0],beta),alpha,np.append(beta,[0.0]) ) )
		offsets = np.asarray([1,0,-1])
		H = dia_matrix((data,offsets), shape=(ncv,ncv),dtype=dtype)
		H *= z/abs(z)
		e0 = np.zeros((ncv,),dtype=dtype)
		e0[0] = 1.0

		e0 = expm_multiply(H,e0)

		
		get_vec = build_krylov.__dict__[char + '_get_vec']
		v = get_vec(A_data_copy,A.indices,A.indptr,e0,alpha,beta,v0)


	else:
		raise NotImplementedError



	return v














def expm_krylov(A, B, z=1.0,hermitian=True, tol=10**(-15)):
	# check A
	if not issparse(A):	
		raise TypeError('expected A to be sparse matrix')
	if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
		raise ValueError('expected A to be like a square matrix')
	if A.dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
		raise NotImplementedError('dtype %s is not supported' % A.dtype)
	if hermitian:
		compare = A != A.H
		if compare.nnz > 0:
			raise ValueError('A expected to be hermitian')

	# check dimensions
	if A.shape[1] != B.shape[0]:
		raise ValueError('the matrices A and B have incompatible shapes')

	if not np.isscalar(z):
		raise NotImplementedError('expected scalar arguement for z')

	A_csr = A.tocsr(True)
	if not A_csr.has_sorted_indices:
		A_csr.sort_indices()

	
	dtype=np.find_common_type([(z*A).dtype,B.dtype],[])


	A_csr = A_csr.astype(dtype)

	if B.ndim > 1:
		# multivector
		F = np.empty(B.shape)
		n = B.shape[1]
		for i in xrange(n):
			F[:,i]=_expm_krylov_vector(A_csr,B[:,i],hermitian,tol,z,dtype)
		return F
	else:
		# vector
		v = _expm_krylov_vector(A_csr,B,hermitian,tol,z,dtype)
		return v




class KrylovError(Exception):
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message




