#
import sys, os


#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import (
    spinless_fermion_basis_1d,
)  # Hilbert space spinless fermion basis
import numpy as np  # generic math functions

#
##### define model parameters #####
L = 10  # system size
J = 1.0  # hopping strength
mu = 0.9045  # chemical potential
U = 1.5  # nn interaction strength
##### define periodic drive #####
Omega = 4.5  # drive frequency


def drive(t, Omega):
    return np.cos(Omega * t)


drive_args = [Omega]
#
##### construct basis in the 0-total momentum and +1-parity sector
basis = spinless_fermion_basis_1d(L=L, Nf=L // 2, a=1, kblock=0, pblock=1)
print(basis)
# define PBC site-coupling lists for operators
n_pot = [[-mu, i] for i in range(L)]
J_nn_right = [[-J, i, (i + 1) % L] for i in range(L)]  # PBC
J_nn_left = [[+J, i, (i + 1) % L] for i in range(L)]  # PBC
U_nn_int = [[U, i, (i + 1) % L] for i in range(L)]  # PBC
# static and dynamic lists
static = [["+-", J_nn_left], ["-+", J_nn_right], ["n", n_pot]]
dynamic = [["nn", U_nn_int, drive, drive_args]]
###### construct Hamiltonian
H = hamiltonian(static, dynamic, dtype=np.float64, basis=basis)
