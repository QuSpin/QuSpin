from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import *
import numpy as np # generic math functions


L=12 # syste size
# coupling strenghts
J=1.0 # spin-spin coupling
h=0.8945 # x-field strength
g=0.945 # z-field strength
# create site-coupling lists
J_zz=[[J,i,(i+1)%L] for i in range(L)] # PBC
x_field=[[h,i] for i in range(L)]
z_field=[[g,i] for i in range(L)]
# create static and dynamic lists
static_1=[["x",x_field],["z",z_field]]
static_2=[["zz",J_zz],["x",x_field],["z",z_field]]
dynamic=[]
# create spin-1/2 basis
basis=spin_basis_1d(L,kblock=0,pblock=1)
# set up Hamiltonian
H1=hamiltonian(static_1,dynamic,basis=basis,dtype=np.float64)
H2=hamiltonian(static_2,dynamic,basis=basis,dtype=np.float64)
# compute GS of H1 and sigen system of H2
E1,V1=H1.eigh()
E2,V2=H2.eigh()
psi=V1[:,0]
# calculate entanglement entropy
print(basis.N)
exit()
Sent=ent_entropy(psi,basis,chain_subsys=[1,3,6,7,11])

