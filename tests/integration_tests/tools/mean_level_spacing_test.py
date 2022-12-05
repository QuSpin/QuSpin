from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import mean_level_spacing
import numpy as np # generic math functions
#
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
static_2=[["zz",J_zz],["x",x_field],["z",z_field]]
dynamic=[]
# create spin-1/2 basis
basis=spin_basis_1d(L,kblock=0,pblock=1)
# set up Hamiltonian
H2=hamiltonian(static_2,dynamic,basis=basis,dtype=np.float64)
# compute eigensystem of H2
E2=H2.eigvalsh()
# calculate mean level spacing of spectrum E2
r=mean_level_spacing(E2)
print("mean level spacing is", r)

E2=np.insert(E2,-1,E2[-1])
r=mean_level_spacing(E2)
print("mean level spacing is", r)

E2=np.insert(E2,-1,E2[-1])
r=mean_level_spacing(E2,verbose=False)
print("mean level spacing is", r)