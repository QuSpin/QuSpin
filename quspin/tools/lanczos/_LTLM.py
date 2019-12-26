from scipy.linalg import eigh_tridiagonal
from six import iteritems
import numpy as _np


__all__ = ["LTLM_static_iteration"]


def LTLM_static_iteration(A_dict,E,V,lv_iter,beta=0):

	nv = E.size

	beta = _np.atleast_1d(beta)
	p = _np.exp(-E*beta.min())*_np.abs(V[0,:]).max()
	mask = p<_np.finfo(p.dtype).eps

	if _np.any(mask):
		nv = _np.argmax(mask)

	Ame_dict = {}

	lv_row = iter(lv_iter)
	for i,lv_r in enumerate(lv_row):
		if i >= nv:
			break

		lv_col = iter(lv_iter)
		Ar_dict = {key:A.dot(lv_r) for key,A in iteritems(A_dict)}
		
		for j,lv_c in enumerate(lv_col):
			if j >= nv:
				break
			for key,A in iteritems(A_dict):
				if key not in Ame_dict:
					dtype = _np.result_type(lv_r.dtype,A.dtype)
					Ame_dict[key] = _np.zeros((nv,nv),dtype=dtype)

				me = _np.vdot(lv_c,Ar_dict[key])
				Ame_dict[key][i,j] = me



		del lv_col
	del lv_row


	p = _np.exp(-_np.outer(E[:nv],_np.atleast_1d(beta)/2.0))
	V = V[:nv,:nv].copy()

	c = _np.einsum("j,j...->j...",V[0,:],p)

	results_dict = {}
	for key,Ame in iteritems(Ame_dict):
		A_diag = V.T.dot(Ame.dot(V))
		results_dict[key] = _np.squeeze(_np.einsum("j...,l...,jl->...",c,c,A_diag))

	p = _np.exp(-_np.outer(E[:nv],_np.atleast_1d(beta)))
	Z = _np.einsum("j,j...->...",V[0,:]**2,p)

	return results_dict,_np.squeeze(Z)
