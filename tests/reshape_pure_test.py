from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

import numpy as np
import scipy.sparse as sp
from quspin.basis._reshape_subsys import _lattice_reshape_pure,_lattice_reshape_sparse_pure
from itertools import permutations


L=6

v = np.arange(2**L)
v_sp = sp.csr_matrix(v)

for L_A in range(1,L):
	for subsys_A in permutations(range(L),r=L_A):
		v_r = _lattice_reshape_pure(v,subsys_A,L,2)
		v_sp_r = _lattice_reshape_sparse_pure(v_sp,subsys_A,L,2)
		np.testing.assert_allclose(v_sp_r.toarray()-v_r,0,err_msg="failed subsys_A: {}".format(subsys_A))
print("reshaping pure states checks passed!")