from .csr_matvec_wrapper import _csr_matvec
from scipy.sparse import isspmatrix_csr,issparse
import numpy as _np
import os


def csr_matvec(A,v,a=None,out=None,overwrite_out=True):
	"""Calculates matrix vector products :math:`x += a A y` or :math:`x = a A y` with csr matrix.

	Notes
	-----
	For QuSpin builds which support OpenMP this function will be multithreaded. Note that using
	out=v will result in incorrect results. Also note that if format of A is not 'csr' 
	the matrix will be converted.
	
	Examples
	--------

	Parameters
	-----------
	A : scipy.spmatrix
		Sparse matrix to take the dot product. 
	v : array_like
		array which contains the vector to take the product with. 
	a : scalar, optional
		value to scale the vector with after the product with `A` is taken.
	out : array_like
		output array to put the results of the calculation.
	overwrite_out : bool, optional
		If set to `True`, the function overwrites the values in `out` with the result. otherwise the result is
		added to the values in `out`. 

	Returns
	--------
	numpy.ndarray
		result of :math:`\\a A v`. 

		If `out` is not None and `overwrite_out = True` the dunction returns `out` with the data overwritten, 
		otherwise if `overwrite_out = False` the result is added to `out`.

		If `out` is None the result is stored in a new array which is returned by the function. 
	

	"""
	if not issparse(A):
		raise ValueError("Expecting sparse matrix for 'A'.")

	if not isspmatrix_csr(A):
		A = A.tocsr()

	if a is None:
		a = 1.0

	result_type=_np.result_type(v.dtype,A.dtype,a)
	
	if A.shape[1]!=v.size:
		raise ValueError("dimension mismatch with shapes {} and {}".format(A.shape,v.shape))

	if out is None:
		out = _np.zeros_like(v,dtype=result_type)
		overwrite_out = True
	else:
		out = _np.asarray(out)

	if v.dtype != result_type:
		v = v.astype(result_type)

	if out.shape != v.shape:
		raise ValueError("ValueError: output array is not the correct shape or dtype.")

	_csr_matvec(overwrite_out,A.indptr,A.indices,A.data,a,v.ravel(),out.ravel())

	return out
