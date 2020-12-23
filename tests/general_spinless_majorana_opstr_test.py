from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spinless_fermion_basis_general
from quspin.operators import hamiltonian

import numpy as np 

J=-np.sqrt(2.0) # hoppping
U=+1.0 # nn interaction


no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
	

for N in range(6,10):


	###### setting up user-defined symmetry transformations for 2d lattice ######
	s = np.arange(N) # sites [0,1,2,....]
	T = (s+1)%N # translation 
	P = s[::-1] # reflection 

	#
	###### setting up bases ######
	basis=spinless_fermion_basis_general(N, tblock=(T,0),pblock=(P,0),)
	#basis=spinless_fermion_basis_general(N,pblock=(P,0),)#pblock=(P,0),)
	#basis=spinless_fermion_basis_general(N,tblock=(T,0),)#pblock=(P,0),)

	#print(basis)

	#
	#
	##### Hamiltonian using Majorana fermions
	#
	#
	hop_term_p=[[+0.5j*J,j,(j+1)%N] for j in range(N)]
	hop_term_m=[[-0.5j*J,j,(j+1)%N] for j in range(N)]
	density_term=[[+0.5j*U,j,j] for j in range(N)]
	int_term=[[-0.25*U,j,j,(j+1)%N,(j+1)%N] for j in range(N)]
	id_term=[[0.25*U,j] for j in range(N)]
	#
	static=[['xy',hop_term_p],['yx',hop_term_m], 					# kinetic energy
			['I',id_term],['xy',density_term],['xyxy',int_term],	# nn interaction energy
			]
	dynamic=[]
	#
	H_majorana=hamiltonian(static,[],basis=basis,dtype=np.float64,**no_checks)

	#
	#
	##### Hamiltonian using complex fermions
	#
	#
	hopping_pm=[[+J,j,(j+1)%N] for j in range(N)]
	hopping_mp=[[-J,j,(j+1)%N] for j in range(N)]
	nn_int=[[U,j,(j+1)%N] for j in range(N)]
	#
	static=[["+-",hopping_pm],["-+",hopping_mp],["nn",nn_int]]
	dynamic=[]
	#
	H=hamiltonian(static,[],basis=basis,dtype=np.float64,**no_checks)



	#######################################


	print("\ntesting N={}...".format(N))
	
	print(H.toarray())
	print()
	print(H_majorana.toarray())
	print()
	print(np.linalg.norm((H-H_majorana).toarray()))

	np.testing.assert_allclose((H_majorana-H).toarray(),0,atol=1e-12)


