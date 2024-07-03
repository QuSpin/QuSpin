from __future__ import print_function, division

#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
#
from quspin.operators import hamiltonian  # operators
from quspin.basis import spinful_fermion_basis_general  # spin basis constructor
import numpy as np  # general math functions

#
###### define model parameters ######
Lx, Ly = 4, 4  # linear dimension of spin 1 2d lattice
N_2d = Lx * Ly  # number of sites for spin 1
#
J = 1.0  # hopping matrix element
U = 2.0  # onsite interaction
mu = 0.5  # chemical potential
#
###### setting up user-defined BASIC symmetry transformations for 2d lattice ######
s = np.arange(N_2d)  # sites [0,1,2,...,N_2d-1] in simple notation
x = s % Lx  # x positions for sites
y = s // Lx  # y positions for sites
T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
S = -(s + 1)  # fermion spin inversion in the simple case
#
###### setting up bases ######
basis_2d = spinful_fermion_basis_general(
    N_2d, Nf=(2, 2), kxblock=(T_x, 0), kyblock=(T_y, 0), sblock=(S, 0)
)
#
###### setting up hamiltonian ######
# setting up site-coupling lists for simple case
hopping_left = [[-J, i, T_x[i]] for i in range(N_2d)] + [
    [-J, i, T_y[i]] for i in range(N_2d)
]
hopping_right = [[+J, i, T_x[i]] for i in range(N_2d)] + [
    [+J, i, T_y[i]] for i in range(N_2d)
]
potential = [[-mu, i] for i in range(N_2d)]
interaction = [[U, i, i] for i in range(N_2d)]
#
static = [
    ["+-|", hopping_left],  # spin up hops to left
    ["-+|", hopping_right],  # spin up hops to right
    ["|+-", hopping_left],  # spin down hopes to left
    ["|-+", hopping_right],  # spin up hops to right
    ["n|", potential],  # onsite potenial, spin up
    ["|n", potential],  # onsite potential, spin down
    ["n|n", interaction],
]  # spin up-spin down interaction
# build hamiltonian
H = hamiltonian(static, [], basis=basis_2d, dtype=np.float64)
# diagonalise H
E = H.eigvalsh()
