from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d # Hilbert spaces
import numpy as np # general math functions
from itertools import product

#
##### setting parameters for simulation
# physical parameters
J = 0.0 # hopping strength
U = np.sqrt(5.0) # interaction strength

for L in [3]: #range(6):

	##### create model
	# define site-coupling lists
	hop_right = [[-J,i,(i+1)%L] for i in range(L)] # hopping to the right PBC
	hop_left = [[J,i,(i+1)%L] for i in range(L)] # hopping to the left PBC
	int_list = [[U,i,i] for i in range(L)] # onsite interaction
	# create static lists
	static= [	
			["+-|", hop_left], # up hop left
			["-+|", hop_right], # up hop right
			["|+-", hop_left], # down hop left
			["|-+", hop_right], # down hop right
			["n|n", int_list], # onsite interaction
			]

	basis = spinful_fermion_basis_1d(L)
	

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
	H = hamiltonian(static,[],basis=basis)
	E=H.eigvalsh()

	E_symm=[]
	
	for N_up, N_down in product(range(L+1),range(L+1)):
	
		###### create the basis
		basis_symm = spinful_fermion_basis_1d(L,Nf=(N_up,N_down))

		#print(basis_symm._basis)

		H_symm = hamiltonian(static,[],basis=basis_symm)
		E_symm.append( H_symm.eigvalsh() )

	E_symm=np.sort( np.concatenate(E_symm) )

	#print(E)
	#print(E_symm)

	np.testing.assert_allclose(E-E_symm,0.0,atol=1E-5,err_msg='Failed tensor and spinfil energies comparison!')

	
