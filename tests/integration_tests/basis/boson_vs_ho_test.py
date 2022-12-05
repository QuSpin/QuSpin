from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d, ho_basis # Hilbert space spin basis
import numpy as np # generic math functions

##### define model parameters #####
L=1 # system size
Np=101

J=1.0
mu=1.0/3.0

hopping=[[J,0]]
chem_pot=[[mu,0]]

static=[["+",hopping],["-",hopping],['n',chem_pot]]

#### define boson model

basis_boson = boson_basis_1d(L,sps=Np)

H_boson=hamiltonian(static,[],basis=basis_boson,dtype=np.float64,check_herm=False,check_symm=False,check_pcon=False)
E_boson=H_boson.eigvalsh()


#### define photon model
basis_ho = ho_basis(Np-1)

H_ho=hamiltonian(static,[],basis=basis_ho,dtype=np.float64,check_herm=False,check_symm=False,check_pcon=False)
E_ho=H_ho.eigvalsh()


#print(max(abs(E_boson-E_ho)))


#### 

np.testing.assert_allclose(E_boson-E_ho,0.0,atol=1E-5,err_msg='Failed boson and ho energies comparison!')

