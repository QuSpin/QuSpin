import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import photon_basis,spin_basis_1d
import numpy as np



L = 4
Nph = 10
omega = 1.0

np.random.seed()

basis = photon_basis(spin_basis_1d,L,Ntot=Nph)
basis_full = photon_basis(spin_basis_1d,L,Nph=Nph)


psi = np.random.uniform(-1,1,size=(basis.Ns,1))+1j*np.random.uniform(-1,1,size=(basis.Ns,1))
psi /= np.linalg.norm(psi,axis=0)
psis = np.random.uniform(-1,1,size=(basis.Ns,100))+1j*np.random.uniform(-1,1,size=(basis.Ns,100))
psis /= np.linalg.norm(psi,axis=0)


psi_full = basis.get_vec(psi,sparse=False)
psis_full = basis.get_vec(psis,sparse=False)


n_state = 10
psis = np.exp(-1j*np.random.uniform(-np.pi,np.pi,size=(basis.Ns,n_state)))/np.sqrt(basis.Ns)
psis_full = basis.get_vec(psis,full_part=False,sparse=False)
proj = basis.get_proj(psis.dtype,full_part=False)


rho_d = np.random.uniform(0,1,size=n_state)
rho_d /= rho_d.sum()

DM = np.einsum("ik,jk,k->ij",psis.conj(),psis,rho_d)
DM_full = np.einsum("ik,jk,k->ij",psis_full.conj(),psis_full,rho_d)


DMs = np.dstack((DM for i in range(5)))
DMs_full = np.dstack((DM_full for i in range(5)))

kwargs_list = [
				dict(sub_sys_A="particles",return_rdm=None,return_rdm_EVs=False),
				dict(sub_sys_A="particles",return_rdm="A",return_rdm_EVs=False),
				dict(sub_sys_A="particles",return_rdm="B",return_rdm_EVs=False),
				dict(sub_sys_A="particles",return_rdm="both",return_rdm_EVs=False),
				dict(sub_sys_A="particles",return_rdm=None,return_rdm_EVs=True),
				dict(sub_sys_A="particles",return_rdm="A",return_rdm_EVs=True),
				dict(sub_sys_A="particles",return_rdm="B",return_rdm_EVs=True),
				dict(sub_sys_A="particles",return_rdm="both",return_rdm_EVs=True),
				dict(sub_sys_A="photons",return_rdm=None,return_rdm_EVs=False),
				dict(sub_sys_A="photons",return_rdm="A",return_rdm_EVs=False),
				dict(sub_sys_A="photons",return_rdm="B",return_rdm_EVs=False),
				dict(sub_sys_A="photons",return_rdm="both",return_rdm_EVs=False),
				dict(sub_sys_A="photons",return_rdm=None,return_rdm_EVs=True),
				dict(sub_sys_A="photons",return_rdm="A",return_rdm_EVs=True),
				dict(sub_sys_A="photons",return_rdm="B",return_rdm_EVs=True),
				dict(sub_sys_A="photons",return_rdm="both",return_rdm_EVs=True),
				]



for kwargs in kwargs_list:
	print("checking kwargs: {}".format(kwargs))
	out = basis.ent_entropy(psi,**kwargs)
	out_full = basis_full.ent_entropy(psi_full,**kwargs)

	for key,val in out_full.items():
		try:
			np.testing.assert_allclose((val - out[key]).todense(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))

	out = basis.ent_entropy(psi,sparse=True,**kwargs)
	out_full = basis_full.ent_entropy(psi_full,sparse=True,**kwargs)

	for key,val in out_full.items():
		try:
			np.testing.assert_allclose((val - out[key]).todense(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))



	out = basis.ent_entropy(psis,**kwargs)
	out_full = basis_full.ent_entropy(psis_full,**kwargs)

	for key,val in out_full.items():
		try:
			np.testing.assert_allclose((val - out[key]).todense(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))



	out = basis.ent_entropy(DM,**kwargs)
	out_full = basis_full.ent_entropy(DM_full,**kwargs)

	for key,val in out_full.items():
		try:
			np.testing.assert_allclose((val - out[key]).todense(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))


	out = basis.ent_entropy(DMs,**kwargs)
	out_full = basis_full.ent_entropy(DMs_full,**kwargs)

	for key,val in out_full.items():
		try:
			np.testing.assert_allclose((val - out[key]).todense(),0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))
		except AttributeError:
			np.testing.assert_allclose(val - out[key],0.0,atol=1E-5,err_msg='Failed {} comparison!'.format(key))

