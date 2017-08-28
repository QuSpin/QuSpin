from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import project_op
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
# compute eigensystem of H1
E1,V1=H1.eigh()
psi1=V1[:,14] # pick any state as initial state
#
# project Hamiltonian from `kblock=0` and `pblock=1` onto full Hilbert space
proj=basis.get_proj(np.float64) # calculate projector
H1_full=project_op(H1,proj,dtype=np.float128)["Proj_Obs"]
print("dimenions of symmetry-reduced and full Hilbert spaces are %i and %i." %(H1.Ns,H1_full.Ns) )