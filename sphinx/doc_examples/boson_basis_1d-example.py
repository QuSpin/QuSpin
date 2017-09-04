from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
import numpy as np # generic math functions
#
##### define model parameters #####
L=5 # system size
J=1.0 # hopping strength
g=0.809 # amplitude for creating boson
mu=0.9045 # chemical potential
U=1.5 # onsite interaction strength
##### define periodic drive #####
Omega=4.5 # drive frequency
def drive(t,Omega):
	return np.cos(Omega*t)
drive_args=[Omega]
#
##### construct basis in the 0-total momentum and +1-parity sector
basis=boson_basis_1d(L=L,sps=3,a=1,kblock=0,pblock=1)
print(basis)
# define PBC site-coupling lists for operators
b_pot=[[g,i] for i in range(L)]
n_pot=[[-mu,i] for i in range(L)]
J_nn=[[-J,i,(i+1)%L] for i in range(L)] # PBC
U_int=[[U,i,i] for i in range(L)] # PBC
# static and dynamic lists
static=[["+-",J_nn],["-+",J_nn],["n",n_pot],["nn",U_int]]
dynamic=[["+",b_pot,drive,drive_args],["-",b_pot,drive,drive_args]]
###### construct Hamiltonian
H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)