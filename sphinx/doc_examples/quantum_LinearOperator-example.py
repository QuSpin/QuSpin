from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import quantum_LinearOperator # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
#
##### define model parameters #####
L=4 # system size
J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field
#
##### construct basis
basis=spin_basis_1d(L=L)
# define PBC site-coupling lists for operators
x_field=[[g,i] for i in range(L)]
z_field=[[h,i] for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
# static list
static=[["zz",J_nn],["z",z_field],["x",x_field]]
#
##### construct Hamiltonian, NO dynamic list passed	
H=quantum_LinearOperator(static,basis=basis,dtype=np.float64)
#
##### apply operator ono state
# compute domain wall initial state
dw_str = "".join("1" for i in range(L//2)) + "".join("0" for i in range(L-L//2))
i_0 = basis.index(dw_str) # find index of product state in basis
psi = np.zeros(basis.Ns) # allocate space for state
psi[i_0] = 1.0 # set MB state to be the given product state
# apply H on psi
psi_new=H.dot(psi)
# diagonalise operator
E,V=H.eigsh(k=1,which='SA')
# calculate operator squared
H_squared=H.dot(H)