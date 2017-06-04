from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

import numpy as _np
import scipy.sparse as _sp
from quspin.basis._reshape_subsys import _lattice_reshape_pure,_lattice_reshape_sparse_pure
from itertools import permutations


L=6

for L_A in range(1,L):
	for subsys_A in permutations(range(L),r=L_A):
		print("checking subsys_A: {}".format(subsys_A))
		v = _np.arange(2**L)
		v_sp = _sp.csr_matrix(v)
		v_r = _lattice_reshape_pure(v,subsys_A,L,2)
		v_sp_r = _lattice_reshape_sparse_pure(v_sp,subsys_A,L,2)
		assert(_np.linalg.norm(v_sp_r.toarray()-v_r)==0)
print("reshaping pure states checks passed!")