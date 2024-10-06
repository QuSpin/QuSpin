#
import sys, os


#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spinful_fermion_basis_1d  # Hilbert space spinful fermion basis
import numpy as np  # generic math functions

#
##### define model parameters #####
L = 6  # system size
J = 1.0  # hopping strength
U = np.sqrt(2)  # onsite interaction strength
#
##### construct basis at half-filling in the 0-total momentum and +1-spin flip sector
basis = spinful_fermion_basis_1d(L=L, Nf=(L // 2, L // 2), a=1, kblock=0, sblock=1)
print(basis)
#
##### define PBC site-coupling lists for operators
# define site-coupling lists
hop_right = [[-J, i, (i + 1) % L] for i in range(L)]  # hopping to the right PBC
hop_left = [[J, i, (i + 1) % L] for i in range(L)]  # hopping to the left PBC
int_list = [[U, i, i] for i in range(L)]  # onsite interaction
# static and dynamic lists
static = [
    ["+-|", hop_left],  # up hop left
    ["-+|", hop_right],  # up hop right
    ["|+-", hop_left],  # down hop left
    ["|-+", hop_right],  # down hop right
    ["n|n", int_list],  # onsite interaction
]
dynamic = []
###### construct Hamiltonian
H = hamiltonian(static, dynamic, dtype=np.float64, basis=basis)
