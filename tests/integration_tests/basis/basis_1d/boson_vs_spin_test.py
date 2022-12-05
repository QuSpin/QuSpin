from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)


from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d, spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions

##### define model parameters #####
L=10 # system size

J=1.0
h=0.8945

J_zz=[[J,i,(i+1)%L] for i in range(L)] # PBC
x_field=[[h,i] for i in range(L)]
hopping=[[0.5*h,i] for i in range(L)]
chem_pot=[[-J,i] for i in range(L)]
identity=[[0.25,i] for i in range(L)]

#### define spin model
basis_spin = spin_basis_1d(L=L,pauli=False,a=1,kblock=0,pblock=1)

static_spin =[["zz",J_zz],["x",x_field]]

H_spin=hamiltonian(static_spin,[],basis=basis_spin,dtype=np.float32)
E_spin=H_spin.eigvalsh()

#### define hcb model
basis_boson = boson_basis_1d(L=L,sps=2,a=1,kblock=0,pblock=1)

static_boson =[["+",hopping],["-",hopping],["n",chem_pot],["nn",J_zz],["I",identity]]

H_boson=hamiltonian(static_boson,[],basis=basis_boson,dtype=np.float32)
E_boson=H_boson.eigvalsh()


#print(max(abs(E_boson-E_spin)))

#### 

np.testing.assert_allclose(E_boson-E_spin,0.0,atol=1E-5,err_msg='Failed boson and ho energies comparison!')




