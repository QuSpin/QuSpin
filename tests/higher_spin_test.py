from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions

print("ENABLE COMPLEX ME and add 'y' to opstr loopS. Then delete this comment!")
print("exiting..")
exit()

##### define spin matrices disctionary
spin_ops={}
spins=['1/2','1','3/2','2']

spin_ops['1/2']={}
spin_ops['1/2']['x']=1.0/2*np.array([[0,1],[1,0]])
spin_ops['1/2']['y']=1.0/2j*np.array([[0,-1],[1,0]])
spin_ops['1/2']['z']=1.0/2*np.array([[-1.,0.0],[0.0,1.]])

spin_ops['1']={}
spin_ops['1']['x']=1.0/np.sqrt(2)*np.array([[0,1,0],[1,0,1],[0,1,0]])
spin_ops['1']['y']=1.0/np.sqrt(1j*2)*np.array([[0,-1,0],[1,0,-1],[0,1,0]])
spin_ops['1']['z']=np.array([[-1.,.0,0.0],[.0,.0,.0],[.0,.0,1.]])

spin_ops['3/2']={}
spin_ops['3/2']['x']=1.0/2.0*np.array([[0,np.sqrt(3),0,0],[np.sqrt(3),0,2,0],[0,2,0,np.sqrt(3)],[0,0,np.sqrt(3),0]])
spin_ops['3/2']['y']=1.0/2j*np.array([[0,-np.sqrt(3),0,0],[np.sqrt(3),0,-2,0],[0,2,0,-np.sqrt(3)],[0,0,np.sqrt(3),0]])
spin_ops['3/2']['z']=np.array([[-3.0/2.0,.0,0.0,0.0],[.0,-1.0/2.0,.0,.0],[.0,.0,1.0/2.0,0,],[0.0,0.0,0.0,3.0/2.0]])

spin_ops['2']={}
spin_ops['2']['x']=1.0/2.0*np.array([[0,2,0,0,0],[2,0,np.sqrt(6),0,0],[0,np.sqrt(6),0,np.sqrt(6),0],[0,0,np.sqrt(6),0,2],[0,0,0,2,0]])
spin_ops['2']['y']=1.0/2j*np.array([[0,-2,0,0,0],[2,0,-np.sqrt(6),0,0],[0,np.sqrt(6),0,-np.sqrt(6),0],[0,0,np.sqrt(6),0,-2],[0,0,0,2,0]])
spin_ops['2']['z']=np.array([[-2.0,.0,0.0,0.0,0.0],[.0,-1.0,.0,.0,0],[.0,.0,.0,.0,.0],[.0,.0,0.0,1.0,0],[0.0,0.0,0.0,0.0,2.0]])



########## test higher spin on-site

L=1 # system size

for S in spins:

	basis = spin_basis_1d(L,S=S,pauli=False)

	for opstr in ['x','z']:

		static=[[opstr, [[1.0,0]]]]
		static, _ = basis._expanded_form(static,[])

		O=hamiltonian(static,[],basis=basis,dtype=np.float32,check_herm=False,check_symm=False,check_pcon=False)

		#print(O.todense()-spin_ops[S][opstr])
		np.testing.assert_allclose(O.todense()-spin_ops[S][opstr],0.0,atol=1E-7,err_msg='Failed boson and ho energies comparison!')



################################################
########## test higher spin for two-sites

L=2

for S in spins:

	basis = spin_basis_1d(L,S=S,pauli=False)

	for opstr in ['xx','zz','xz','zx']:

		list_opstr=list(opstr)
		static=[[opstr, [[1.0,i,i+1] for i in range(L-1)]] ]
		static, _ = basis._expanded_form(static,[])

		O=hamiltonian(static,[],basis=basis,dtype=np.float32,check_herm=False,check_symm=False,check_pcon=False)
		O_kron=np.kron( spin_ops[S][list_opstr[1]], spin_ops[S][list_opstr[0]])

		#print(O.todense()-O_kron)

		np.testing.assert_allclose(O.todense()-O_kron,0.0,atol=1E-7,err_msg='Failed boson and ho energies comparison!')





