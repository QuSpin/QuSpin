from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d,tensor_basis,photon_basis,ho_basis
from quspin.tools.measurements import ent_entropy
import numpy as np
import scipy.sparse as sp
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers



# testing basis.partial trace 


J=1.0
h=0.8945

for L in range(2,8,1):

	J_zz=[[J,i,(i+1)%L] for i in range(L)] # PBC
	x_field=[[h,i] for i in range(L)]

	static =[["zz",J_zz],["+",x_field],["-",x_field]]

	for S in ["1/2","1"]:

		if uniform(0.0,1.0) < 0.5:
			chain_subsys=list(np.unique([randint(0,L) for r in range(L//2)]))
		else:
			chain_subsys=[r for r in range(L-1)]


		basis = spin_basis_1d(L,S=S,kblock=0,pblock=1)

		#print(chain_subsys)
		#print(basis.Ns)
		
		H=hamiltonian(static,[],basis=basis,check_symm=False,check_herm=False,check_pcon=False)
		_,V=H.eigh()
		psi=V[:,0]

		tools_ent=ent_entropy(psi,basis,chain_subsys=chain_subsys,DM='both')
		basis_ent=basis.ent_entropy(psi,sub_sys_A=chain_subsys,return_rdm='both')

		Sent_tools=tools_ent['Sent']
		Sent_basis=basis_ent['Sent_A']

		rho_tools_A = tools_ent['DM_chain_subsys']
		rho_basis_A = basis_ent['rdm_A']

		rho_tools_B = tools_ent['DM_other_subsys']
		rho_basis_B = basis_ent['rdm_B']

		#print(rho_tools_A)
		#print(rho_basis_A)
		#print(np.linalg.norm(rho_tools-rho_basis) )
		#print(S)
		
		np.testing.assert_array_almost_equal_nulp(rho_tools_A,rho_basis_A,nulp=1E9)
		np.testing.assert_array_almost_equal_nulp(rho_tools_B,rho_basis_B,nulp=1E9)
		np.testing.assert_array_almost_equal_nulp(Sent_tools,Sent_basis,nulp=1E9)






