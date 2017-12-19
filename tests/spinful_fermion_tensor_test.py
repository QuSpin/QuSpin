from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import tensor_basis,spinless_fermion_basis_1d,spinful_fermion_basis_1d # Hilbert spaces
import numpy as np # general math functions
from itertools import product

#
##### setting parameters for simulation
# physical parameters
J = 1.0 # hopping strength
U = 5.0 # interaction strength

for L in range(1,8,1):

	##### create model
	# define site-coupling lists
	hop_right = [[-J,i,i+1] for i in range(L-1)] # hopping to the right OBC
	hop_left = [[J,i,i+1] for i in range(L-1)] # hopping to the left OBC
	int_list = [[U,i,i] for i in range(L)] # onsite interaction
	# create static lists
	static= [	
			["+-|", hop_left], # up hop left
			["-+|", hop_right], # up hop right
			["|+-", hop_left], # down hop left
			["|-+", hop_right], # down hop right
			["n|n", int_list], # onsite interaction
			]
	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False,dtype=np.float64)

	for N_up, N_down in product(range(L+1),range(L+1)):

		print("L=%s, Nup=%s, Ndown=%s" %(L,N_up,N_down) )

		###### create the basis
		# build the two bases to tensor together to spinful fermions
		basis_up = spinless_fermion_basis_1d(L,Nf=N_up) # up basis
		basis_down = spinless_fermion_basis_1d(L,Nf=N_down) # down basis
		basis_tensor = tensor_basis(basis_up,basis_down) # spinful fermions

		basis_spinful = spinful_fermion_basis_1d(L,Nf=(N_up,N_down))

		H_tensor = hamiltonian(static,[],basis=basis_tensor,**no_checks)
		H_spinful = hamiltonian(static,[],basis=basis_spinful,**no_checks)


		E_tensor,V_tensor=H_tensor.eigh()
		E_spinful,V_spinful=H_spinful.eigh()


		np.testing.assert_allclose(E_tensor-E_spinful,0.0,atol=1E-5,err_msg='Failed tensor and spinfil energies comparison!')
		#np.testing.assert_allclose( (H_tensor-H_spinful).toarray(),0.0,atol=1E-5,err_msg='Failed tensor and spinfil energies comparison!')

