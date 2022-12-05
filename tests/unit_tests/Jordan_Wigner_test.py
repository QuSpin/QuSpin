from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d, boson_basis_1d, spinless_fermion_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions

##### define model parameters #####
L=10 # system size
J=1.0 # spin interaction
h=0.8592 # spin magnetic field


for PBC in [-1,1]: # periodic or antiperiodic BC
	
	# defien site-coupling lists for external field
	x_field=[[2.0*h,i] for i in range(L)]
	
	if PBC==1:
		J_pm=[[-J,i,(i+1)%L] for i in range(L)] # PBC
		J_mp=[[+J,i,(i+1)%L] for i in range(L)] # PBC
		J_pp=[[-J,i,(i+1)%L] for i in range(L)] # PBC
		J_mm=[[+J,i,(i+1)%L] for i in range(L)] # PBC

		basis_fermion = spinless_fermion_basis_1d(L=L,Nf=range(1,L+1,2))

	elif PBC==-1:

		J_pm=[[-J,i,(i+1)%L] for i in range(L-1)] # APBC
		J_mp=[[+J,i,(i+1)%L] for i in range(L-1)] # APBC
		J_pp=[[-J,i,(i+1)%L] for i in range(L-1)] # APBC
		J_mm=[[+J,i,(i+1)%L] for i in range(L-1)] # APBC

		J_pm.append([+J,L-1,0])
		J_mp.append([-J,L-1,0])
		J_pp.append([+J,L-1,0])
		J_mm.append([-J,L-1,0])

		basis_fermion = spinless_fermion_basis_1d(L=L,Nf=range(0,L+1,2))


	static_fermion =[["+-",J_pm],["-+",J_mp],["++",J_pp],["--",J_mm],['z',x_field]]

	H_fermion=hamiltonian(static_fermion,[],basis=basis_fermion,dtype=np.float64,check_pcon=False)
	E_fermion=H_fermion.eigvalsh()

	#### define spin model

	J_zz=[[-J,i,(i+1)%L] for i in range(L)] # PBC
	x_field=[[-h,i] for i in range(L)]

	if PBC==1:
		basis_spin = spin_basis_1d(L=L,zblock=-1)#,a=1,kblock=0,pblock=1)
	elif PBC==-1:
		basis_spin = spin_basis_1d(L=L,zblock=1)#,a=1,kblock=0,pblock=1)

	static_spin =[["zz",J_zz],["x",x_field]]

	H_spin=hamiltonian(static_spin,[],basis=basis_spin,dtype=np.float64)
	E_spin=H_spin.eigvalsh()


	#### define hcb model

	J_zz=[[-4.0*J,i,(i+1)%L] for i in range(L)] # PBC
	x_field=[[-h,i] for i in range(L)]

	if PBC==1:
		basis_hcb = boson_basis_1d(L=L,cblock=-1,sps=2)#,a=1,kblock=0,pblock=1)
	elif PBC==-1:
		basis_hcb = boson_basis_1d(L=L,cblock=1,sps=2)#,a=1,kblock=0,pblock=1)

	static_hcb =[["zz",J_zz],["+",x_field],["-",x_field]]

	H_hcb=hamiltonian(static_hcb,[],basis=basis_hcb,dtype=np.float64)
	E_hcb=H_hcb.eigvalsh()




	#### 

	np.testing.assert_allclose(E_fermion-E_spin,0.0,atol=1E-6,err_msg='Failed fermion and spin energies comparison for PBC={}!'.format(PBC))
	np.testing.assert_allclose(E_hcb-E_spin,0.0,atol=1E-6,err_msg='Failed hcb and spin energies comparison for PBC={}!'.format(PBC))




