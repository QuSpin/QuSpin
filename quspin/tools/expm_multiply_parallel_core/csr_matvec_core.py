from quspin.tools.expm_multiply_parallel_core.csr_matvec_wrapper import _csr_matvec
from scipy.sparse import isspmatrix_csr,issparse
import numpy as _np
import os


def csr_matvec(A,v,a=None,out=None,overwrite_out=True):
	"""DEPRICATED (cf `matvec`). Calculates matrix vector products :math:`x += a A y` or :math:`x = a A y` with csr matrix.

	:red:`Note: we recommend the use of "tools.misc.matvec()" instead of this function. This function is now deprecated!`

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
