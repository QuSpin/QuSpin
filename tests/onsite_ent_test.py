from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)


from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import ent_entropy
import numpy as np # generic math functions
import functools


L=10 # system size

J=1.0 # hopping
U=np.sqrt(2) # interactions strenth
h=np.sqrt(3)

for PBC in [0,1]:

	# define site-coupling lists
	field=[[h,i] for i in range(L)]
	if PBC:
		interaction=[[U,i,(i+3)%L] for i in range(L)] # PBC
		hopping=[[J,i,(i+1)%L] for i in range(L)] # PBC
	else:
		interaction=[[U,i,(i+1)%L] for i in range(L-1)] # PBC
		hopping=[[J,i,(i+1)%L] for i in range(L-1)] # PBC

	sigmaz=[[1.0,0]]

	#### define hcb model
	basis_0 = spin_basis_1d(L=1,pauli=False)
	basis = spin_basis_1d(L=L,pauli=False) 

	# Hubbard-related model
	static =[["+-",hopping],["-+",hopping],["zz",interaction],["x",field]]


	# define operators
	Sx_0=hamiltonian([['x',sigmaz]],[],basis=basis_0,dtype=np.float32)
	Sx_full= np.kron(Sx_0.todense(),np.eye(2**(L-1)))


	H=hamiltonian(static,[],basis=basis,dtype=np.float32)

	E,V=H.eigh()
	psi=V[:,0]


	DM=ent_entropy(psi,basis,chain_subsys=[0],DM='chain_subsys')['DM_chain_subsys']
	#print(np.around(DM,3))
	
 	# calculate expectation in full and reduced basis
	Exct_1 = np.trace(Sx_0.dot(DM))
	try:
		Exct_2 = float(reduce(np.dot,[psi.conjugate(),Sx_full,psi]) )
	except NameError:
		Exct_2 = float(functools.reduce(np.dot,[psi.conjugate(),Sx_full,psi]) )


	np.testing.assert_allclose(Exct_1-Exct_2,0.0,atol=1E-5,err_msg='Failed onsite DM comaprison for PBC={}!'.format(PBC))

