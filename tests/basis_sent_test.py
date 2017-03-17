from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_1d,tensor_basis,photon_basis,ho_basis
import numpy as np
import scipy.sparse as sp



# sparse vs. dense comparison
# symmetry vs. non-symmetry

def tensor_entropy_test():
	L=8
	b1 = spin_basis_1d(L//2)
	b2 = spin_basis_1d(L)
	bt = tensor_basis(b1,b1)
	psi = np.random.uniform(-1,1,size=(3,1,2,bt.Ns))
	psi = (psi.T/np.linalg.norm(psi,axis=-1).T).T


	Sent1 = b2.ent_entropy(psi,sub_sys_A=list(range(L//2)))["Sent"]
	Sent2 = bt.ent_entropy(psi)["Sent"]

	np.testing.assert_array_almost_equal_nulp(Sent1,Sent2,nulp=10)

	rho = np.einsum("...j,...k->...jk",psi,psi)

	Sent1 = b2.ent_entropy(rho,sub_sys_A=list(range(L//2)),state_type="mixed")["Sent"]
	Sent2 = bt.ent_entropy(rho,state_type="mixed")["Sent"]

	np.testing.assert_array_almost_equal_nulp(Sent1,Sent2,nulp=10)


def spin_entropy_test():
	L=2
	for S in ["1/2","1","3/2","2","5/2","3","7/2","4","9/2","5"]:
		b = spin_basis_1d(L,S=S)
		rho = np.zeros((b.Ns,b.Ns),dtype=np.float64)

		rho[np.arange(b.Ns),np.arange(b.Ns)] = 1.0/b.Ns

		result = b.ent_entropy(rho,sub_sys_A=[0],state_type="mixed")
		Sent = result["Sent"]
		np.testing.assert_array_almost_equal_nulp(Sent,np.log(b.sps),nulp=100)






tensor_entropy_test()
spin_entropy_test()

