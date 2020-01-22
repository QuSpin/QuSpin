from scipy.linalg import eigh_tridiagonal
from six import iteritems
import numpy as _np
from ._lanczos_utils import _get_first_lv


__all__ = ["FTLM_static_iteration"]


def FTLM_static_iteration(O_dict,E,V,Q_T,beta=0):
	"""Calculate iteration for Finite-Temperature Lanczos method.

	Here we give a brief overview of this method based on notes, `arXiv:1111.5931 <https://arxiv.org/abs/1111.5931>`_. 

	One would naively think that it would require full diagonalization to calculate thermodynamic expectation values 
	for a quantum system as one has to fully diagonalize the Hamiltonian to evaluate:

	.. math::
		\\langle O\\rangle_\\beta = \\frac{1}{Z}Tr\\left(e^{-\\beta H}O\\right)
	
	with the partition function defined as: :math:`Z=Tr\\left(e^{-\\beta H}\\right)`. The idea behind the 
	Finite-Temperature Lanczos Method (FTLM) is to use quantum typicality as well as Krylov subspaces to 
	simplify this calculation. Typicality states that the trace of an operator can be approximated as an 
	average of that same operator with random vectors in the H-space samples with the Harr measure. 
	As a corollary, it is known that the fluctuations of this corollary for any finite sample set will 
	converge to 0 as the size of the Hilbert space increases. If you combine this with the Lanczos method 
	to approximate the matrix exponential :math:`e^{-\\beta H}`. Mathematically this is expressed as:

	.. math::
		\\langle O\\rangle_\\beta = \\frac{\\overline{\\langle O\\rangle_r}}{\\overline{\\langle Z\\rangle_r}}

	with the two quaitites are defined as averages over quantities calculate with Lanczos vectors that start with random initial vectors :math:`|r\\rangle`:

	.. math::
		\\overline{\\langle O\\rangle_r} = \\frac{1}{N_r}\\sum_{r} \\langle O\\rangle_r = \\frac{1}{N_r}\\sum_{r}\\sum_{n}e^{-\\beta \\epsilon^{(r)}_n}\\langle r|\\psi^{(r)}_n\\rangle\\langle\\psi^{(r)}_n|O|r\\rangle

	.. math::
		\\overline{\\langle Z\\rangle_r} = \\frac{1}{N_r}\\sum_{r} \\langle Z\\rangle_r =\\frac{1}{N_r}\\sum_{r}\\sum_{n}e^{-\\beta \\epsilon^{(r)}_n}|\\langle r|\\psi^{(r)}_n\\rangle|^2

	The purpose of this function is to calculate :math:`\\langle O\\rangle_r` and :math:`\\langle Z\\rangle_r` 
	for a Krylov subspace provided; this implies that to perform the full FTLM calculation this function 
	must be called many times to perform the average over random initial states.

	Notes
	-----
	* The amount of memory used by this function scales like: :math:`nN_{op}` with :math:`n` being the size of the full Hilbert space and :math:`N_{op}` is the number of input operators. 
	* FTLM does not converge very well at low temperatures, see function for low-temperature lanczos iterations. 

	Parameters
	-----------
	O_dict : dictionary of Python Objects
		These Objects must have a 'dot' method that calculates a matrix vector product on a numpy.ndarray[:], the effective shape of these objects should be (n,n). 
	E : array_like, (m,)
		Eigenvalues for the Krylow projection of some operator.
	V : array_like, (m,m)
		Eigenvectors for the Krylow projection of some operator.
	Q_T : iterator over rows of Q_T
		generator or ndarray that contains the lanczos basis associated with E, and V.  
	beta : scalar/array_like, any shape
		Inverse temperature values to evaluate.

	Returns
	--------
	Result_dict: dictionary
		A dictionary storying the results for a single iteration of the FTLM. The results are stored in numpy.ndarrays 
		that have the same shape as `beta`. The keys of `Result_dict` are the same as the keys in `O_dict` and the values 
		associated with the given key in `Result_dict` are the expectation values for the operator in `O_dict` with the same key.
	Z: numpy.ndarray, same shape as `beta`
		The value of the partition function for the given iterator for each beta. 


	Examples
	--------

	>>> beta = numpy.linspace(0,10,101)
	>>> E, V, Q = lanczos_full(H,v0,20)
	>>> Res,Z = FTLM_static_iteration(Obs_dict,E,V,Q,beta=beta)


	"""
	nv = E.size


	p = _np.exp(-_np.outer(E,_np.atleast_1d(beta)))
	c = _np.einsum("j,aj,j...->a...",V[0,:],V,p)

	r,Q_T = _get_first_lv(iter(Q_T))

	results_dict = {}

	Ar_dict = {key:A.dot(r) for key,A in iteritems(O_dict)}

	for i,lv in enumerate(Q_T): # nv matvecs
		for key,A in iteritems(O_dict):
			if key in results_dict:
				results_dict[key] += _np.squeeze(c[i,...] * _np.vdot(lv,Ar_dict[key]))
			else:
				results_dict[key]  = _np.squeeze(c[i,...] * _np.vdot(lv,Ar_dict[key]))

	return results_dict,_np.squeeze(c[0,...])



def _FTLM_dynamic_iteration(A_dagger,E_l,V_l,Q_iter_l,E_r,V_r,Q_iter_r,beta=0):

	nv = E_r.size

	p = _np.exp(-_np.outer(E_l,_np.atleast_1d(beta)))
	c = _np.einsum("i,j,i...->ij...",V_l[0,:],V_r[0,:],p)

	A_me = None
	for i,lv_r in enumerate(Q_iter_l):
		lv_col = iter(Q_iter_r)
		for j,lv_c in enumerate(lv_col):
			me = _np.vdot(lv_r,A_dagger.dot(lv_c))
			if A_me is None:
				A_me = _np.zeros((nv,nv),dtype=me.dtype)

			A_me[i,j] = me

	A_me_diag = V_l.T.dot(A_me.dot(V_r))
	result = _np.einsum("ij,ij...->ij...",A_me_diag,c)
	omegas = np.subtract.outer(E_l,E_r)
	Z = _np.einsum("j,j...->...",V[0,:]**2,p)

	return result,omegas,Z