from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import ent_entropy
import numpy as np # generic math functions
#
L=12 # syste size
# coupling strenghts
h=0.8945 # x-field strength
g=0.945 # z-field strength
# create site-coupling lists
x_field=[[h,i] for i in range(L)]
z_field=[[g,i] for i in range(L)]
# create static and dynamic lists
static_1=[["x",x_field],["z",z_field]]
dynamic=[]
# create spin-1/2 basis
basis=spin_basis_1d(L,kblock=0,pblock=1)
# set up Hamiltonian
H1=hamiltonian(static_1,dynamic,basis=basis,dtype=np.float64)
# compute eigensystems of H1
E1,V1=H1.eigh()
psi1=V1[:,14] # pick any state as initial state
# calculate entanglement entropy
Sent=ent_entropy(psi1,basis,chain_subsys=[1,3,6,7,11])
print(Sent['Sent_A'])