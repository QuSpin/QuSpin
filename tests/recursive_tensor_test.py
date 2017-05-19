from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import tensor_basis,spin_basis_1d
from quspin.operators import hamiltonian
import numpy as np



L=3
Li=1
b1 = spin_basis_1d(Li)

basis2 = tensor_basis(*(b1 for i in range(L)))

J1 = [[1.0,i,(i+1)%L] for i in range(L)]
J2 = [[1.0,0,0]]
static1 = [["xx",J1],["yy",J1],["zz",J1]]
static2 = [
			["x|x|",J2],["y|y|",J2],["z|z|",J2],
			["x||x",J2],["y||y",J2],["z||z",J2],
			["|x|x",J2],["|y|y",J2],["|z|z",J2],
			]
			
H1 = hamiltonian(static1,[],N=L)
H2 = hamiltonian(static2,[],basis=basis2,check_pcon=False,check_symm=False)


np.testing.assert_allclose((H1-H2).todense(),0)




