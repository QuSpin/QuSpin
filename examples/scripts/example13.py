#
import sys, os

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # uncomment this line if omp error occurs on OSX for python 3
)
os.environ["OMP_NUM_THREADS"] = "1"  # set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = "1"  # set number of MKL threads to run in parallel
#

###########################################################################
#                            example 13                                   #
#  In this script we demonstrate how to construct a spinful fermion basis #
#  with no doubly occupancid sites in the Fermi-Hubbard model,            #
#  using the spinful_fermion_ basis_general class.                        #
###########################################################################
from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian
import numpy as np

#
###### define model parameters ######
Lx, Ly = 3, 3  # linear dimension of spin 1 2d lattice
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
P_x = x + Lx * (Ly - y - 1)  # reflection about x-axis
P_y = (Lx - x - 1) + Lx * y  # reflection about y-axis
S = -(s + 1)  # fermion spin inversion in the simple case
#
###### setting up bases ######
basis_2d = spinful_fermion_basis_general(
    N_2d,
    Nf=(3, 3),
    double_occupancy=False,
    kxblock=(T_x, 0),
    kyblock=(T_y, 0),
    pxblock=(P_x, 1),
    pyblock=(P_y, 0),  # contains GS
    sblock=(S, 0),
)
print(basis_2d)
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
# compute GS of H
E_GS, psi_GS = H.eigsh(k=1, which="SA")
