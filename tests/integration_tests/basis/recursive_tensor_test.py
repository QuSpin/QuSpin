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

J1 = [[1.0,i] for i in range(L-1)]
J2 = [[1.0,0]]
static1 = [["x",J1],["y",J1],["z",J1]]
static2 = [
			["x||",J2],["y||",J2],["z||",J2],
			["|x|",J2],["|y|",J2],["|z|",J2],
			]
			
H1 = hamiltonian(static1,[],N=L)
H2 = hamiltonian(static2,[],basis=basis2,check_pcon=False,check_symm=False)

E1 = H1.eigvalsh()
E2 = H2.eigvalsh()

np.testing.assert_allclose((E1-E2),0,atol=1e-14)




