from __future__ import print_function,division
import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian # Hamiltonian and observables
from quspin.tools.measurements import diag_ensemble
import numpy as np
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers
seed()


"""
This test only makes sure the function 'diag_ensemble' runs properly.
"""
dtypes={"float32":np.float32,"float64":np.float64,
		"complex64":np.complex64,"complex128":np.complex128}

atols={"float32":1E-4,"float64":1E-13,
		"complex64":1E-4,"complex128":1E-13}

L=10
basis = spin_basis_1d(L,kblock=0,pblock=1,zblock=1)
J_zz = [[1.0,i,(i+1)%L,(i+2)%L] for i in range(0,L)] 
J_xy = [[1.0,i,(i+1)%L] for i in range(0,L)]
# static and dynamic lists
static_pm = [["+-",J_xy],["-+",J_xy]]
static_zxz = [["zxz",J_zz]]

for _i in dtypes.keys():
	dtype = dtypes[_i]
	atol = atols[_i]

	# build Hamiltonian
	O_pm=hamiltonian(static_pm,[],basis=basis,dtype=dtype,check_herm=False,check_symm=False)
	O_zxz = hamiltonian(static_zxz,[],basis=basis,dtype=dtype,check_herm=False,check_symm=False)
	#
	H1=O_pm+O_zxz
	H2=O_pm-O_zxz
	# diagonalise H
	E1,V1 = H1.eigh()
	E2,V2 = H2.eigh()
	psi0=V1[:,0]
	rho0=np.outer(psi0.conj(),psi0)


	alpha=uniform(5)

	DE_args={"density":randint(2),"alpha":uniform(5),"rho_d":True,"Srdm_args":{"basis":basis}}

	### pure state
	DE = diag_ensemble(L,psi0,E2,V2,Obs=O_zxz,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,psi0,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,psi0,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,psi0,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=True,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,psi0,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=True,Srdm_Renyi=True,**DE_args)
	
	### DM
	DE = diag_ensemble(L,rho0,E2,V2,Obs=O_zxz,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,rho0,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,rho0,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,rho0,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=True,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,rho0,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=True,Srdm_Renyi=True,**DE_args)

	### thermal average
	beta=[10.0,1.0,0.1]
	in_state={'V1':V1,'E1':E1,'f_args':[beta],'V1_state':[0,2,4,6]}
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=True,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=True,Srdm_Renyi=True,**DE_args)

	### mixed average
	beta=[10.0,1.0,0.1]
	in_state={'V1':V1,'E1':E1,'f':lambda x,v: np.exp(-v**2*(x-x[0])**0.5),'f_args':[beta],'V1_state':[0,2,4,6],'f_norm':False}
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=False,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=False,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=True,Srdm_Renyi=False,**DE_args)
	DE = diag_ensemble(L,in_state,E2,V2,Obs=O_zxz,delta_t_Obs=True,delta_q_Obs=True,Sd_Renyi=True,Srdm_Renyi=True,**DE_args)

print("diag_ensemble checks passed!")
