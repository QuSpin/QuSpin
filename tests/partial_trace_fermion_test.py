import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spinless_fermion_basis_1d, spinful_fermion_basis_1d, spinless_fermion_basis_general, spinful_fermion_basis_general
from quspin.operators import hamiltonian
from itertools import combinations

import numpy as np 
import scipy.sparse as sps
np.random.seed(0)



def test_pure_states(N=3,sparse=False,enforce_pure=False):

	if sparse:
		psi_full=np.zeros((basis_full.Ns,N),)
		inds = np.random.choice(np.arange(basis_full.N), replace=False, size=3)
		psi_full[inds,...]=1.0/np.sqrt(3.0)
		psi_full = sps.csr_matrix(psi_full)
	else:
		psi_full=np.random.uniform(size=(basis_full.Ns,N))
		psi_full/=np.linalg.norm(psi_full)

	rho_red=basis_full.partial_trace(psi_full,sub_sys_A=sub_sys_A,return_rdm='A',subsys_ordering=False, enforce_pure=False)

	np.testing.assert_allclose(A_full.expt_value(psi_full) - A_red.expt_value(rho_red.T, enforce_pure=False), 0.0,atol=1E-5,err_msg='failed local operator comparison!')

	Sent=basis_full.ent_entropy(psi_full,sub_sys_A=sub_sys_A,return_rdm='A',subsys_ordering=False, enforce_pure=False)['Sent_A']
	
	#print(Sent)




def test_mixed_states(N=3,sparse=False,enforce_pure=False):

	if sparse:

		if N>1:
			rdm_full=np.zeros((basis_full.Ns,basis_full.Ns,N),)
		else:
			rdm_full=np.zeros((basis_full.Ns,basis_full.Ns),)

		for _ in range(5): # mixed state consisting of 5 random pure states

			if N>1:
				psi_full=np.zeros((basis_full.Ns,N),)
			else:
				psi_full=np.zeros((basis_full.Ns,),)
			inds = np.random.choice(np.arange(basis_full.N), replace=False, size=3)
			psi_full[inds]=1.0/np.sqrt(3.0)
			psi_full = sps.csr_matrix(psi_full)


			rdm_full+= np.einsum('i...,j...->ij...',psi_full.conj(),psi_full)


	else:
		if N>1:
			rdm_full=np.random.uniform(0.0,1.0,size=(basis_full.Ns,basis_full.Ns,N))
			# normalize and make positive definite
			rdm_full = np.einsum('ji...,jk...->ik...',rdm_full.conj(),rdm_full)
			rdm_full/=np.trace(rdm_full,axis1=0,axis2=1)
		else:
			rdm_full=np.random.uniform(0.0,1.0,size=(basis_full.Ns,basis_full.Ns,))
			# normalize and make positive definite
			rdm_full = np.einsum('ji...,jk...->ik...',rdm_full.conj(),rdm_full)
			rdm_full/=np.trace(rdm_full)

	rho_red=basis_full.partial_trace(rdm_full,sub_sys_A=sub_sys_A,return_rdm='A',subsys_ordering=False, enforce_pure=enforce_pure)
		
	np.testing.assert_allclose(A_full.expt_value(rdm_full) - A_red.expt_value(rho_red.T, enforce_pure=False), 0.0,atol=1E-5,err_msg='failed local operator comparison!')

	Sent=basis_full.ent_entropy(rdm_full,sub_sys_A=sub_sys_A,return_rdm='A',subsys_ordering=False, enforce_pure=enforce_pure)['Sent_A']
	
	#print(Sent)





L=4
no_checks=dict(check_symm=False, check_herm=False, check_pcon=False)


spinless_bases_list = [spinless_fermion_basis_1d, ] #spinless_fermion_basis_general ]
spinful_bases_list = [spinful_fermion_basis_1d, ] #spinful_fermion_basis_general ]



for spinful, basis in enumerate(spinless_bases_list+spinful_bases_list):

	print(basis)


	if spinful:
		basis_full=basis(L,Nf=[(i,i) for i in range(0,L,1)] ) # even sector 
	else:
		basis_full=basis(L,Nf=[i for i in range(0,L+2,2)]) # even sector
		basis_red =basis(2,)


	for sub_sys_A in combinations(range(basis_full.N), 2):

		if spinful:

			subsysA=list(sub_sys_A)

			subsysA_mod = [subsysA[0]%L, subsysA[1]%L]

			if np.min(subsysA)<=L-1 and np.max(subsysA)>L-1:
				sub_sys_A=[[subsysA_mod[0],], [subsysA_mod[1]] ]
			elif np.max(subsysA)<=L-1:
				sub_sys_A=[subsysA_mod,[] ]
			elif np.min(sub_sys_A)>L-1:
				sub_sys_A=[[],subsysA_mod ]
				

			J_nn_p_full = [[+1.0]+subsysA_mod]
			J_nn_n_full = [[-1.0]+subsysA_mod]


			if np.min(subsysA)<=L-1 and np.max(subsysA)>L-1:
				basis_red =spinful_fermion_basis_1d(1,)

				J_nn_p_red = [[+1.0,0,0]]
				J_nn_n_red = [[-1.0,0,0]]

				static_full=[['+|+',J_nn_p_full], ['-|-',J_nn_n_full]]
				static_red=[['+|+',J_nn_p_red], ['-|-',J_nn_n_red]]

			elif np.max(subsysA)<=L-1:

				basis_red =spinless_fermion_basis_1d(2,)

				J_nn_p_red = [[+1.0,0,1]]
				J_nn_n_red = [[-1.0,0,1]]

				static_full=[['++|',J_nn_p_full], ['--|',J_nn_n_full]]
				static_red=[['++',J_nn_p_red], ['--',J_nn_n_red]]
				

			elif np.min(subsysA)>L-1:
				basis_red =spinless_fermion_basis_general(2,)

				J_nn_p_red = [[+1.0,0,1]]
				J_nn_n_red = [[-1.0,0,1]]

				static_full=[['|++',J_nn_p_full], ['|--',J_nn_n_full]]
				static_red=[['++',J_nn_p_red], ['--',J_nn_n_red]]
			
		else:

			sub_sys_A=list(sub_sys_A)

			J_nn_p_full = [[+1.0]+sub_sys_A]
			J_nn_n_full = [[-1.0]+sub_sys_A]

			J_nn_p_red = [[+1.0,0,1]]
			J_nn_n_red = [[-1.0,0,1]]
			

			static_full=[['++',J_nn_p_full], ['--',J_nn_n_full]]
			static_red=[['++',J_nn_p_red], ['--',J_nn_n_red]]


		A_full = hamiltonian(static_full, [], basis=basis_full, dtype=np.float64, **no_checks)
		A_red = hamiltonian(static_red, [], basis=basis_red, dtype=np.float64, **no_checks)


		test_pure_states(N=1, enforce_pure=True)
		test_pure_states(N=3, enforce_pure=True)
		test_pure_states(N=1, enforce_pure=True, sparse=True)

		test_mixed_states(N=1)
		test_mixed_states(N=3)
		test_pure_states(N=1, sparse=True)
		

