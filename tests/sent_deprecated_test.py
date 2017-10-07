from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian # Hamiltonian and observables
from quspin.tools.measurements import _ent_entropy
import numpy as np
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers

seed()


dtypes={"float64":np.float64,"complex128":np.complex128}

atols={"float32":1E-4,"float64":1E-12,
		"complex64":1E-4,"complex128":1E-12}



def spin_entropy(dtype,symm,Sent_args):

	if symm:
		basis = spin_basis_1d(L,kblock=0,pblock=1,zblock=1) 
	else:
		basis = spin_basis_1d(L)
	# define operators with OBC using site-coupling lists
	J_zz = [[1.0,i,(i+1)%L,(i+2)%L] for i in range(0,L)] 
	J_xy = [[1.0,i,(i+1)%L] for i in range(0,L)]

	# static and dynamic lists
	static = [["+-",J_xy],["-+",J_xy],["zxz",J_zz]]
	# build Hamiltonian
	H=hamiltonian(static,[],basis=basis,dtype=dtype,check_herm=False,check_symm=False)
	# diagonalise H
	E,V = H.eigh()
	psi0=V[:,0]
	rho0=np.outer(psi0.conj(),psi0)
	rho_d=rho0
	Ed,Vd = np.linalg.eigh(rho_d)


	S_pure = _ent_entropy(psi0,basis,**Sent_args)
	S_DM = _ent_entropy(rho0,basis,**Sent_args)
	S_DMd = _ent_entropy({'V_rho': Vd, 'rho_d': abs(Ed)},basis,**Sent_args)
	S_all = _ent_entropy({'V_states':V},basis,**Sent_args)

	return (S_pure, S_DM, S_DMd, S_all)

def spin_photon_entropy(dtype,symm,Sent_args):

	Nph = 6

	if symm:
		basis = photon_basis(spin_basis_1d,L,Nph=Nph,kblock=0,pblock=1) 
	else:
		basis = photon_basis(spin_basis_1d,L,Nph=Nph)

	# spin
	x_field=[[-1.0,i] for i in range(L)]
	z_field=[[-1.0,i] for i in range(L)]
	J_nn=[[-1.0,i,(i+1)%L] for i in range(L)]
	# spin-photon
	absorb=[[-0.5*1.0/np.sqrt(Nph),i] for i in range(L)]
	emit=[[-0.5*np.conj(1.0)/np.sqrt(Nph),i] for i in range(L)]
	# photon
	ph_energy = [[11.0/L,i] for i in range(L)]

	static = [["zz|",J_nn], ["z|",z_field], ["x|",x_field], ["x|-",absorb], ["x|+",emit], ["I|n",ph_energy]]


	H=hamiltonian(static,[],L,dtype=np.float64,basis=basis,check_herm=False,check_symm=False)


	# diagonalise H
	E,V = H.eigh()
	psi0=V[:,0]
	rho0=np.outer(psi0.conj(),psi0)
	rho_d=rho0
	Ed,Vd = np.linalg.eigh(rho_d)


	S_pure = _ent_entropy(psi0,basis,**Sent_args)
	S_DM = _ent_entropy(rho0,basis,**Sent_args)
	S_DMd = _ent_entropy({'V_rho': Vd, 'rho_d': abs(Ed)},basis,**Sent_args)
	S_all = _ent_entropy({'V_states':V},basis,**Sent_args)

	return (S_pure, S_DM, S_DMd, S_all)

def spin_photon_entropy_cons(dtype,symm,Sent_args):

	Ntot = L

	if symm:
		basis = photon_basis(spin_basis_1d,L,Ntot=Ntot,kblock=0,pblock=1) 
	else:
		basis = photon_basis(spin_basis_1d,L,Ntot=Ntot)

	# spin
	x_field=[[-1.0,i] for i in range(L)]
	z_field=[[-1.0,i] for i in range(L)]
	J_nn=[[-1.0,i,(i+1)%L] for i in range(L)]
	# spin-photon
	absorb=[[-0.5*1.0/np.sqrt(Ntot),i] for i in range(L)]
	emit=[[-0.5*np.conj(1.0)/np.sqrt(Ntot),i] for i in range(L)]
	# photon
	ph_energy = [[11.0/L,i] for i in range(L)]

	static = [["zz|",J_nn], ["z|",z_field], ["x|",x_field], ["x|-",absorb], ["x|+",emit], ["I|n",ph_energy]]


	H=hamiltonian(static,[],L,dtype=np.float64,basis=basis,check_herm=False,check_symm=False,check_pcon=False)


	# diagonalise H
	E,V = H.eigh()
	psi0=V[:,0]
	rho0=np.outer(psi0.conj(),psi0)
	rho_d=rho0
	Ed,Vd = np.linalg.eigh(rho_d)


	S_pure = _ent_entropy(psi0,basis,**Sent_args)
	S_DM = _ent_entropy(rho0,basis,**Sent_args)
	S_DMd = _ent_entropy({'V_rho': Vd, 'rho_d': abs(Ed)},basis,**Sent_args)
	S_all = _ent_entropy({'V_states':V},basis,**Sent_args)

	return (S_pure, S_DM, S_DMd, S_all)




for _r in range(10): # do 10 random checks

	L=6
	if uniform(0.0,1.0) < 1.0: #0.5:
		chain_subsys=list(np.unique([randint(0,L) for r in range(L//2)]))
	else:
		chain_subsys=[r for r in range(L)]
	alpha=uniform(5)

	Sent_args={'chain_subsys':chain_subsys,'alpha':alpha,'density':randint(2)}

	

	for dtype_str in dtypes.keys():
		
		atol = atols[dtype_str]
		dtype=dtypes[dtype_str]
		
		S=np.zeros((2,4),dtype=dtype)
		for symm in [0,1]:

			"""
			S1=[]
			# check entropies also between symmetries
			for _i,_s in enumerate( spin_photon_entropy_cons(dtype,symm,Sent_args) ):
				
				if isinstance(_s['Sent'],np.ndarray):
					S1.append(_s['Sent'][0])
				else:
					S1.append(_s['Sent'])

			S[symm,:] = S1

			np.testing.assert_allclose(np.diff(S1),0.0,atol=atol,err_msg='Failed entropies comparison!')
			if symm == 2:
				np.testing.assert_allclose(np.diff(S.ravel()),0.0,atol=atol,err_msg='Failed entropies comparison symm <--> no_symm!')
			"""
			# check reduced DM's
			Sent_args['DM']='both'
			DM_chain_subsys=[]
			DM_other_subsys=[]
			S2=[]
			for _i,_s in enumerate( spin_entropy(dtype,symm,Sent_args) ):

				if isinstance(_s['Sent'],np.ndarray):
					DM_chain_subsys.append(_s['DM_chain_subsys'][0])
					DM_other_subsys.append(_s['DM_other_subsys'][0])
					S2.append(_s['Sent'][0])
				else:
					DM_chain_subsys.append(_s['DM_chain_subsys'])
					DM_other_subsys.append(_s['DM_other_subsys'])
					S2.append(_s['Sent'])

			np.testing.assert_allclose(np.diff(S2),0.0,atol=atol,err_msg='Failed entropies comparison!')
			np.testing.assert_allclose(np.diff(DM_chain_subsys,axis=0),0.0,atol=atol,err_msg='Failed DM_chain_subsys comparison!')
			np.testing.assert_allclose(np.diff(DM_other_subsys,axis=0),0.0,atol=atol,err_msg='Failed DM_other_subsys comparison!')

					
			# call ent entropy to make sure U, lmbda, V do not produce errors
			Sent_args['svd_return_vec']=[1,1,1]
			spin_entropy(dtype,symm,Sent_args)

	print("entropy (photon, conserved) random check {} finished successfully".format(_r))




for _r in range(10): # do 10 random checks

	L=6
	if uniform(0.0,1.0) < 0.5:
		chain_subsys=list(np.unique([randint(0,L) for r in range(L//2)]))
	else:
		chain_subsys=[r for r in range(L)]
	alpha=uniform(5)

	Sent_args={'chain_subsys':chain_subsys,'alpha':alpha,'density':randint(2)}


	for dtype_str in dtypes.keys():
		
		atol = atols[dtype_str]
		dtype=dtypes[dtype_str]
		
		S=np.zeros((2,4),dtype=dtype)
		for symm in [0,1]:
			"""
			S1=[]
			# check entropies also between symmetries
			for _i,_s in enumerate( spin_photon_entropy(dtype,symm,Sent_args) ):
				
				if isinstance(_s['Sent'],np.ndarray):
					S1.append(_s['Sent'][0])
				else:
					S1.append(_s['Sent'])

			S[symm,:] = S1

			np.testing.assert_allclose(np.diff(S1),0.0,atol=atol,err_msg='Failed entropies comparison!')
			if symm == 2:
				np.testing.assert_allclose(np.diff(S.ravel()),0.0,atol=atol,err_msg='Failed entropies comparison symm <--> no_symm!')
			
			"""
			# check reduced DM's
			Sent_args['DM']='both'
			DM_chain_subsys=[]
			DM_other_subsys=[]
			S2=[]
			for _i,_s in enumerate( spin_entropy(dtype,symm,Sent_args) ):

				if isinstance(_s['Sent'],np.ndarray):
					DM_chain_subsys.append(_s['DM_chain_subsys'][0])
					DM_other_subsys.append(_s['DM_other_subsys'][0])
					S2.append(_s['Sent'][0])
				else:
					DM_chain_subsys.append(_s['DM_chain_subsys'])
					DM_other_subsys.append(_s['DM_other_subsys'])
					S2.append(_s['Sent'])

			
			np.testing.assert_allclose(np.diff(S2),0.0,atol=atol,err_msg='Failed entropies comparison!')
			np.testing.assert_allclose(np.diff(DM_chain_subsys,axis=0),0.0,atol=atol,err_msg='Failed DM_chain_subsys comparison!')
			np.testing.assert_allclose(np.diff(DM_other_subsys,axis=0),0.0,atol=atol,err_msg='Failed DM_other_subsys comparison!')

					
			# call ent entropy to make sure U, lmbda, V do not produce errors
			Sent_args['svd_return_vec']=[1,1,1]
			spin_entropy(dtype,symm,Sent_args)

	print("entropy (photon) random check {} finished successfully".format(_r))



for _r in range(10): # do 10 random checks

	L=8
	chain_subsys=list(np.unique([randint(0,L) for r in range(L//2)]))
	alpha=uniform(5)

	Sent_args={'chain_subsys':chain_subsys,'alpha':alpha,'density':randint(2)}


	for dtype_str in dtypes.keys():
		
		atol = atols[dtype_str]
		dtype=dtypes[dtype_str]
		
		S=np.zeros((2,4),dtype=dtype)
		for symm in [0,1]:

			S1=[]
			# check entropies also between symmetries
			for _i,_s in enumerate( spin_entropy(dtype,symm,Sent_args) ):
				
				if isinstance(_s['Sent'],np.ndarray):
					S1.append(_s['Sent'][0])
				else:
					S1.append(_s['Sent'])

			S[symm,:] = S1

			np.testing.assert_allclose(np.diff(S1),0.0,atol=atol,err_msg='Failed entropies comparison!')
			if symm == 1:
				np.testing.assert_allclose(np.diff(S.ravel()),0.0,atol=atol,err_msg='Failed entropies comparison symm <--> no_symm!')
			
			# check reduced DM's
			Sent_args['DM']='both'
			DM_chain_subsys=[]
			DM_other_subsys=[]
			S2=[]
			for _i,_s in enumerate( spin_entropy(dtype,symm,Sent_args) ):

				if isinstance(_s['Sent'],np.ndarray):
					DM_chain_subsys.append(_s['DM_chain_subsys'][0])
					DM_other_subsys.append(_s['DM_other_subsys'][0])
					S2.append(_s['Sent'][0])
				else:
					DM_chain_subsys.append(_s['DM_chain_subsys'])
					DM_other_subsys.append(_s['DM_other_subsys'])
					S2.append(_s['Sent'])


			
			np.testing.assert_allclose(np.diff(S2),0.0,atol=atol,err_msg='Failed entropies comparison!')
			np.testing.assert_allclose(np.diff(DM_chain_subsys,axis=0),0.0,atol=atol,err_msg='Failed DM_chain_subsys comparison!')
			np.testing.assert_allclose(np.diff(DM_other_subsys,axis=0),0.0,atol=atol,err_msg='Failed DM_other_subsys comparison!')

					
			# call ent entropy to make sure U, lmbda, V do not produce errors
			Sent_args['svd_return_vec']=[1,1,1]
			spin_entropy(dtype,symm,Sent_args)

	print("entropy (spin) random check {} finished successfully".format(_r))

print("Entanglement entropy checks passed!")
