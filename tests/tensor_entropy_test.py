from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_1d,tensor_basis
import numpy as np
import scipy.sparse as sp

from functools import reduce




L1 = 3
L2 = 4
L = L1+L2

np.random.seed(2)


spin_basis = spin_basis_1d(L)

spin_basis_left = spin_basis_1d(L1)
spin_basis_right = spin_basis_1d(L2)

Ns_l = spin_basis_left.Ns
Ns_r = spin_basis_right.Ns

test_basis = tensor_basis(spin_basis_left,spin_basis_right)
# print(test_basis)
# print(spin_basis)


psi = np.random.uniform(-1,1,size=(spin_basis.Ns,)) + 1j*np.random.uniform(-1,1,size=(spin_basis.Ns,))
psi /= np.linalg.norm(psi)

sp_psi = sp.csr_matrix(psi).T

psis = np.random.uniform(-1,1,size=(spin_basis.Ns,spin_basis.Ns)) + 1j*np.random.uniform(-1,1,size=(spin_basis.Ns,spin_basis.Ns))
psis /= np.linalg.norm(psis,axis=0)

p_DM=np.random.uniform(size=spin_basis.Ns)
p_DM/=p_DM.sum()

DM = reduce(np.dot,[psis,np.diag(p_DM/sum(p_DM)),psis.T.conj()])

DMs = np.dstack((DM for i in range(10)))

kwargs_list = [
				(dict(sub_sys_A="left",return_rdm=None,return_rdm_EVs=False),	dict(density=False,sub_sys_A=range(L1),return_rdm=None,return_rdm_EVs=False)),
				(dict(sub_sys_A="left",return_rdm="A",return_rdm_EVs=False),	dict(density=False,sub_sys_A=range(L1),return_rdm="A",return_rdm_EVs=False)),
				(dict(sub_sys_A="left",return_rdm="B",return_rdm_EVs=False),	dict(density=False,sub_sys_A=range(L1),return_rdm="B",return_rdm_EVs=False)),
				(dict(sub_sys_A="left",return_rdm="both",return_rdm_EVs=False),	dict(density=False,sub_sys_A=range(L1),return_rdm="both",return_rdm_EVs=False)),
				(dict(sub_sys_A="left",return_rdm=None,return_rdm_EVs=True),	dict(density=False,sub_sys_A=range(L1),return_rdm=None,return_rdm_EVs=True)),
				(dict(sub_sys_A="left",return_rdm="A",return_rdm_EVs=True),		dict(density=False,sub_sys_A=range(L1),return_rdm="A",return_rdm_EVs=True)),
				(dict(sub_sys_A="left",return_rdm="B",return_rdm_EVs=True),		dict(density=False,sub_sys_A=range(L1),return_rdm="B",return_rdm_EVs=True)),
				(dict(sub_sys_A="left",return_rdm="both",return_rdm_EVs=True),	dict(density=False,sub_sys_A=range(L1),return_rdm="both",return_rdm_EVs=True)),
				(dict(sub_sys_A="right",return_rdm=None,return_rdm_EVs=False),	dict(density=False,sub_sys_A=range(L1,L,1),return_rdm=None,return_rdm_EVs=False)),
				(dict(sub_sys_A="right",return_rdm="A",return_rdm_EVs=False),	dict(density=False,sub_sys_A=range(L1,L,1),return_rdm="A",return_rdm_EVs=False)),
				(dict(sub_sys_A="right",return_rdm="B",return_rdm_EVs=False),	dict(density=False,sub_sys_A=range(L1,L,1),return_rdm="B",return_rdm_EVs=False)),
				(dict(sub_sys_A="right",return_rdm="both",return_rdm_EVs=False),dict(density=False,sub_sys_A=range(L1,L,1),return_rdm="both",return_rdm_EVs=False)),
				(dict(sub_sys_A="right",return_rdm=None,return_rdm_EVs=True),	dict(density=False,sub_sys_A=range(L1,L,1),return_rdm=None,return_rdm_EVs=True)),
				(dict(sub_sys_A="right",return_rdm="A",return_rdm_EVs=True),	dict(density=False,sub_sys_A=range(L1,L,1),return_rdm="A",return_rdm_EVs=True)),
				(dict(sub_sys_A="right",return_rdm="B",return_rdm_EVs=True),	dict(density=False,sub_sys_A=range(L1,L,1),return_rdm="B",return_rdm_EVs=True)),
				(dict(sub_sys_A="right",return_rdm="both",return_rdm_EVs=True),	dict(density=False,sub_sys_A=range(L1,L,1),return_rdm="both",return_rdm_EVs=True)),
				]

np.set_printoptions(linewidth=10000000,precision=2)

for kwargs2,kwargs1 in kwargs_list:
	print("checking kwargs: {}".format(kwargs2))
	out_spin = spin_basis.ent_entropy(psi,**kwargs1)
	out_tensor = test_basis.ent_entropy(psi,**kwargs2)

	for key,val in out_spin.items():
		try:
			np.testing.assert_allclose((val - out_tensor[key]).toarray(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out_tensor[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))

	out_spin = spin_basis.ent_entropy(psi,sparse=True,**kwargs1)
	out_tensor = test_basis.ent_entropy(psi,sparse=True,**kwargs2)
	for key,val in out_spin.items():
		try:
			np.testing.assert_allclose((val - out_tensor[key]).toarray(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out_tensor[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))

	out_spin = spin_basis.ent_entropy(psis,enforce_pure=True,**kwargs1)
	out_tensor = test_basis.ent_entropy(psis,enforce_pure=True,**kwargs2)

	for key,val in out_spin.items():
		try:
			np.testing.assert_allclose((val - out_tensor[key]).toarray(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out_tensor[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))

	out_spin = spin_basis.ent_entropy(DM,**kwargs1)
	out_tensor = test_basis.ent_entropy(DM,**kwargs2)
	
	for key,val in out_spin.items():
		try:
			np.testing.assert_allclose((val - out_tensor[key]).toarray(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out_tensor[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))

	out_spin = spin_basis.ent_entropy(DMs,**kwargs1)
	out_tensor = test_basis.ent_entropy(DMs,**kwargs2)

	for key,val in out_spin.items():
		try:
			np.testing.assert_allclose((val - out_tensor[key]).toarray(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out_tensor[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))

print("tensor_basis ent_entropy passed!")