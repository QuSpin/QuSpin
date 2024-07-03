from __future__ import print_function, division

#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
#
from quspin.operators import hamiltonian  # operators
from quspin.basis import tensor_basis, spinless_fermion_basis_1d  # Hilbert spaces
from quspin.tools.measurements import obs_vs_time  # calculating dynamics
import numpy as np  # general math functions

#
##### setting parameters for simulation
# physical parameters
L = 8  # system size
N = L // 2  # number of particles
N_up = N // 2 + N % 2  # number of fermions with spin up
N_down = N // 2  # number of fermions with spin down
J = 1.0  # hopping strength
U = 5.0  # interaction strength
#
###### create the basis
# build the two bases to tensor together to spinful fermions
basis_up = spinless_fermion_basis_1d(L, Nf=N_up)  # up basis
basis_down = spinless_fermion_basis_1d(L, Nf=N_down)  # down basis
basis = tensor_basis(basis_up, basis_down)  # spinful fermions
print(basis)
#
##### create model
# define site-coupling lists
hop_right = [[-J, i, i + 1] for i in range(L - 1)]  # hopping to the right OBC
hop_left = [[J, i, i + 1] for i in range(L - 1)]  # hopping to the left OBC
int_list = [[U, i, i] for i in range(L)]  # onsite interaction
# create static lists
static = [
    ["+-|", hop_left],  # up hop left
    ["-+|", hop_right],  # up hop right
    ["|+-", hop_left],  # down hop left
    ["|-+", hop_right],  # down hop right
    ["n|n", int_list],  # onsite interaction
]
#
###### setting up operators
# set up hamiltonian dictionary and observable (imbalance I)
no_checks = dict(check_pcon=False, check_symm=False, check_herm=False)
H = hamiltonian(static, [], basis=basis, **no_checks)
