from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d,tensor_basis,photon_basis,ho_basis
from quspin.tools.measurements import _ent_entropy
import numpy as np
import scipy.sparse as sp
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers



# testing basis.partial trace 


L=2

J=1.0
h=0.8945

J_zz=[[J,i,(i+1)%L] for i in range(L)] # PBC
x_field=[[h,i] for i in range(L)]

static =[["zz",J_zz],["+",x_field],["-",x_field]]

# define rho_check for L=2; J=1.0; h=0.8945; and kblock=0; pblock=1
rho_test={}
rho_test["1/2"]=np.array([  [ 0.5,-0.43644424],
					 		[-0.43644424,0.5]] )
rho_test["1"]=np.array( [   [ 0.3087624, -0.31805836, 0.22282817],
				    		[-0.31805836, 0.38247519,-0.31805836],
							[ 0.22282817,-0.31805836, 0.3087624 ]]  )
rho_test["3/2"]=np.array( [ [ 0.31042621, -0.22463578,  0.1369919,  -0.08695713],
					  		[-0.22463578,  0.18957379, -0.15510269,  0.1369919 ],
					  		[ 0.1369919 , -0.15510269,  0.18957379, -0.22463578],
					  		[-0.08695713,  0.1369919 ,-0.22463578,  0.31042621]]  )
rho_test["2"]=np.array([  [ 0.38625202, -0.19306019, 0.06839545, -0.02735268,  0.01846154],
				    		[-0.19306019 , 0.1002271, -0.0404452,   0.02502469, -0.02735268],
				   			[ 0.06839545 ,-0.0404452,  0.02704177, -0.0404452,   0.06839545],
				  			[-0.02735268 , 0.02502469, -0.0404452,   0.1002271,  -0.19306019],
							[ 0.01846154 ,-0.02735268, 0.06839545, -0.19306019,  0.38625202]]   )

for S in ["1/2","1","3/2","2"]:


	basis = spin_basis_1d(L,S=S,kblock=0,pblock=1)
	
	H=hamiltonian(static,[],basis=basis,dtype=np.float64,check_symm=False,check_herm=False,check_pcon=False)
	_,V=H.eigh()
	psi=V[:,0]

	rho_tools = _ent_entropy(psi,basis,DM='chain_subsys')['DM_chain_subsys']
	rho_basis = basis.partial_trace(psi)

	#print(rho_tools)
	#print(rho_basis)
	#print(np.linalg.norm(rho_tools-rho_basis) )
	#print(np.spacing(rho_basis),np.spacing(rho_tools))
	#print(S)
	
	np.testing.assert_array_almost_equal_nulp(rho_test[S],rho_basis,nulp=1E9)






