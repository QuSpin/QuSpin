from scipy.linalg import eigh_tridiagonal
from six import iteritems
import numpy as _np
from ._lanczos_utils import _get_first_lv


__all__ = ["FTLM_static_iteration"]


def FTLM_static_iteration(A_dict,E,V,Q_iter,beta=0):

	nv = E.size


	p = _np.exp(-_np.outer(E,_np.atleast_1d(beta)))
	c = _np.einsum("j,aj,j...->a...",V[0,:],V,p)

	r,Q_iter = _get_first_lv(iter(Q_iter))

	results_dict = {}

	Ar_dict = {key:A.dot(r) for key,A in iteritems(A_dict)}

	for i,lv in enumerate(Q_iter): # nv matvecs
		for key,A in iteritems(A_dict):
			if key in results_dict:
				results_dict[key] += _np.squeeze(c[i,...] * _np.vdot(lv,Ar_dict[key]))
			else:
				results_dict[key]  = _np.squeeze(c[i,...] * _np.vdot(lv,Ar_dict[key]))

	return results_dict,_np.squeeze(c[0,...])



def FTLM_dynamic_iteration(A_dagger,E_l,V_l,Q_iter_l,E_r,V_r,Q_iter_r,beta=0):

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