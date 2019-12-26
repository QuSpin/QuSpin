import numpy as _np
from ._lanczos_utils import combine_lv

__all__ = ["expm_lanczos"]

def expm_lanczos(E,V,lv,a=1.0,out=None):
	""" Calculates action of matrix exponential on vector using Lanczos algorithm. 

	The Lanczos decomposition `(E,V,Q)` with initial state `v0` of a hermitian matrix `A` can be used to compute the matrix exponential 
	:math:`exp(aA)|v_0\\rangle` applied to the quantum state :math:`|v_0\\rangle`, without actually computing the matrix exponential.   

	Note
	----
	* uses precomputed Lanczos data `(E,V,Q)`, see e.g., `lanczos_full` and `lanczos_iter` functions. 
	* the initial state `v0` used in `lanczos_full` and `lanczos_iter` is the state the matrix exponential is evaluated on.

	Parameters
	-----------
	E : np.ndarray[:]
		eigenvalues of Krylov subspace tridiagonal matrix :math:`T`.
	V : np.ndarray[:,:]
		eigenvectors of Krylov subspace tridiagonal matrix :math:`T`.
	lv : np.ndarray[:,:]
		Lanczos eigenvectors. 
	a : scalar, optional
		Scale factor `a` for the generator of the matrix exponential :math:`exp(aA)`.
	out : np.ndarray()
		Array to store the result in.
	
	Returns
	--------
	np.ndarray
		Matrix exponential applied to a state, evaluated using the Lanczos method. 

	Examples
	--------

	>>> E, V, lv = lanczos_iter(H,v0,20)
	>>> expH_v0 = expm_lanczos(E,V,lv,a=-1j)

	"""
	c = V.dot(_np.exp(a*E)*V[0,:])
	return combine_lv(c,lv,out=out)

