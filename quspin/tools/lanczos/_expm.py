import numpy as _np
from ._lanczos_utils import combine_lv

__all__ = ["expm_lanczos"]

def expm_lanczos(E,V,lv,alpha=1,out=None):
	c = V.dot(_np.exp(alpha*E)*V[0,:])
	return combine_lv(c,lv,out=out)

