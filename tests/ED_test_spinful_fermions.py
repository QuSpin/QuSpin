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
J = 1.0 # hopping strength
U = 5.0 # interaction strength

for L in [4]: #range(8):

	N_up=L//2
	N_down=L//2

	##### create model
	# define site-coupling lists
	hop_right = [[-J,i,(i+1)%L] for i in range(L-1)] # hopping to the right OBC
	hop_left = [[J,i,(i+1)%L] for i in range(L-1)] # hopping to the left OBC
	int_list = [[U,i,i] for i in range(L)] # onsite interaction
	# create static lists
	static= [	
			["+-|", hop_left], # up hop left
			["-+|", hop_right], # up hop right
			["|+-", hop_left], # down hop left
			["|-+", hop_right], # down hop right
			["n|n", int_list], # onsite interaction
			]
	
	
	###### create the basis
	basis = spinful_fermion_basis_1d(L,Nf=(N_up,N_down))
	basis_symm = spinful_fermion_basis_1d(L,Nf=(N_up,N_down),kblock=0)

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
	H = hamiltonian(static,[],basis=basis,**no_checks)
	H_symm = hamiltonian(static,[],basis=basis_symm,**no_checks)


	E=H.eigvalsh()
	E_sym=H_symm.eigvalsh()

	print(E)
	print(E_sym)

	#np.testing.assert_allclose(E_tensor-E_spinful,0.0,atol=1E-5,err_msg='Failed tensor and spinfil energies comparison!')

	
