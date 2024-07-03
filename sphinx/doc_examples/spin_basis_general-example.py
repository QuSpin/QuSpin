from __future__ import print_function, division

#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
#
from quspin.operators import hamiltonian  # operators
from quspin.basis import spin_basis_general  # spin basis constructor
import numpy as np  # general math functions

#
###### define model parameters ######
J = 1.0  # spin=spin interaction
g = 0.5  # magnetic field strength
Lx, Ly = 4, 4  # linear dimension of spin 1 2d lattice
N_2d = Lx * Ly  # number of sites for spin 1
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d)  # sites [0,1,2,....]
x = s % Lx  # x positions for sites
y = s // Lx  # y positions for sites
T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
P_x = x + Lx * (Ly - y - 1)  # reflection about x-axis
P_y = (Lx - x - 1) + Lx * y  # reflection about y-axis
Z = -(s + 1)  # spin inversion
#
###### setting up bases ######
basis_2d = spin_basis_general(
    N_2d,
    kxblock=(T_x, 0),
    kyblock=(T_y, 0),
    pxblock=(P_x, 0),
    pyblock=(P_y, 0),
    zblock=(Z, 0),
)
#
###### setting up hamiltonian ######
# setting up site-coupling lists
Jzz = [[J, i, T_x[i]] for i in range(N_2d)] + [[J, i, T_y[i]] for i in range(N_2d)]
gx = [[g, i] for i in range(N_2d)]
#
static = [["zz", Jzz], ["x", gx]]
# build hamiltonian
H = hamiltonian(static, [], basis=basis_2d, dtype=np.float64)
# diagonalise H
E = H.eigvalsh()
