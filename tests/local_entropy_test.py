import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

import numpy as np
import scipy.sparse as sp
from quspin.basis import spin_basis_1d
from itertools import permutations

L=6
basis = spin_basis_1d(L)

for L_A in range(1,L):
	for subsys in permutations(range(L),r=L_A):
		
		psi = np.random.uniform(-1,1,size=(basis.Ns,)) + 1j*np.random.uniform(-1,1,size=(basis.Ns,))
		psi /= np.linalg.norm(psi)

		comp = list(set(range(L)) - set(subsys))

		out_A = basis.ent_entropy(psi,subsys)
		out_B = basis.ent_entropy(psi,comp)
		for key,value in out_A.items():
			np.testing.assert_allclose(value-out_B[key],0,atol=1e-14)

		out_A = basis.ent_entropy(psi,subsys,return_rdm="both")
		out_B = basis.ent_entropy(psi,comp,return_rdm="both")
		for key,value in out_A.items():
			if "A" in key:
				key = key.replace("A","B")
			elif "B" in key:
				key = key.replace("B","A")
			np.testing.assert_allclose(value-out_B[key].conj(),0,atol=1e-14)
