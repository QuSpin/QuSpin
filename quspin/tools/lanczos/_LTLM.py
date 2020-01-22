from scipy.linalg import eigh_tridiagonal
from six import iteritems
import numpy as _np


__all__ = ["LTLM_static_iteration"]


def LTLM_static_iteration(O_dict,E,V,Q_iter,beta=0):
	"""Calculate iteration for low-temperature Lanczos method.

	Here we give a brief overview of this method based on notes, `arXiv:1111.5931 <https://arxiv.org/abs/1111.5931>`_. 

	One would niavely think that it would require full diagonalization to calculate thermodynamic expoectation values 
	for a quantum system as one has to fully diagonalize the Hamiltonian to evaluate:

	.. math::
		\\langle O\\rangle_\\beta = \\frac{1}{Z}Tr\\left(e^{-\\beta H}O\\right)
	
	with the partition function defined as: :math:`Z=Tr\\left(e^{-\\beta H}\\right)`. The idea behind the 
	Low-Temperature Lanczos Method (LTLM) is to use quantum typicality as well as Krylov subspaces to 
	simplify this calculation. Typicality states that trace of an operator can be approximated as an average 
	of that same operator with random vectors in the H-space samples with the Harr measure. As a colloary it 
	is know that the flucuations of this average for any finite sample set will converge to 0 as the size of 
	the Hilbert space increases. If you combine this with the Lanczos method to approximate the matrix 
	exponential :math:`e^{-\\beta H}`. Mathematically this is expressed as:

	.. math::
		\\langle O\\rangle_\\beta = \\frac{\\overline{\\langle O\\rangle_r}}{\\overline{\\langle Z\\rangle_r}}

	with the two quaitites are defined as averages over quantities calculate with Lanczos vectors that start 
	with random initial vectors :math:`|r\\rangle`. The difference between this method and the finte-temperature 
	Lanczos method (FTLM) is that instead of expending the expression in an asymmetric way the density matrix 
	is split up to create a symmetric form:

	.. math::
		\\overline{\\langle O\\rangle_r} = \\frac{1}{N_r}\\sum_{r} \\langle O\\rangle_r = \\frac{1}{N_r}\\sum_{r}\\sum_{nm}e^{-\\beta(\\epsilon^{(r)}_n+\\epsilon^{(r)}_m)/2}\\langle r|\\psi^{(r)}_n\\rangle\\langle\\psi^{(r)}_n|O|\\psi^{(r)}_m\\rangle\\langle\\psi^{(r)}_m|r\\rangle

	.. math::
		\\overline{\\langle Z\\rangle_r} = \\frac{1}{N_r}\\sum_{r} \\langle Z\\rangle_r =\\frac{1}{N_r}\\sum_{r}\\sum_{n}e^{-\\beta \\epsilon^{(r)}_n}|\\langle r|\\psi^{(r)}_n\\rangle|^2

	The purpose of this function is to calculate :math:`\\langle O\\rangle_r` and :math:`\\langle Z\\rangle_r` 
	for a Krylov subspace provided. This implies that in order to perform the full LTLM calculation this function 
	must be called many times to perform the average over random initial states.


	Notes
	-----

	* The amount of memory used by this function scales like: :math:`nN_{op}` with :math:`n` being the size of the full Hilbert space and :math:`N_{op}` is the number of input operators. 
	* LTLM converges equally well for low and high temperatures however it is more expensive compared to the FTLM and hence we recomend that one should use that method when dealing with high temperatures.



	Parameters
	-----------

	O_dict : dictionary of Python Objects, 
		These Objects must have a 'dot' method that calculates a matrix vector product on a numpy.ndarray[:]. The effective shape of these objects should be (n,n). 
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
		A dictionary storying the results for a single iteration of the LTLM. The results are stored in numpy.ndarrays 
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

	beta = _np.atleast_1d(beta)
	p = _np.exp(-E*beta.min())*_np.abs(V[0,:]).max()
	mask = p<_np.finfo(p.dtype).eps

	if _np.any(mask):
		nv = _np.argmax(mask)

	Ome_dict = {}

	lv_row = iter(Q_iter)
	for i,lv_r in enumerate(lv_row):
		if i >= nv:
			break

		lv_col = iter(Q_iter)
		Ar_dict = {key:A.dot(lv_r) for key,A in iteritems(O_dict)}
		
		for j,lv_c in enumerate(lv_col):
			if j >= nv:
				break
			for key,A in iteritems(O_dict):
				if key not in Ome_dict:
					dtype = _np.result_type(lv_r.dtype,A.dtype)
					Ome_dict[key] = _np.zeros((nv,nv),dtype=dtype)

				me = _np.vdot(lv_c,Ar_dict[key])
				Ome_dict[key][i,j] = me



		del lv_col
	del lv_row


	p = _np.exp(-_np.outer(E[:nv],_np.atleast_1d(beta)/2.0))
	V = V[:nv,:nv].copy()

	c = _np.einsum("j,j...->j...",V[0,:],p)

	results_dict = {}
	for key,Ame in iteritems(Ome_dict):
		A_diag = V.T.dot(Ame.dot(V))
		results_dict[key] = _np.squeeze(_np.einsum("j...,l...,jl->...",c,c,A_diag))

	p = _np.exp(-_np.outer(E[:nv],_np.atleast_1d(beta)))
	Z = _np.einsum("j,j...->...",V[0,:]**2,p)

	return results_dict,_np.squeeze(Z)
