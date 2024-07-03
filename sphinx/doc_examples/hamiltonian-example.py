from __future__ import print_function, division

#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
import numpy as np  # generic math functions

#
##### define model parameters #####
L = 4  # system size
J = 1.0  # spin interaction
g = 0.809  # transverse field
h = 0.9045  # longitudinal field
##### define periodic drive #####
Omega = 4.5  # drive frequency


def drive(t, Omega):
    return np.cos(Omega * t)


drive_args = [Omega]
#
##### construct basis in the 0-total momentum and +1-parity sector
basis = spin_basis_1d(L=L, a=1, kblock=0, pblock=1)
# define PBC site-coupling lists for operators
x_field = [[g, i] for i in range(L)]
z_field = [[h, i] for i in range(L)]
J_nn = [[J, i, (i + 1) % L] for i in range(L)]  # PBC
# static and dynamic lists
static = [["zz", J_nn], ["z", z_field]]
dynamic = [["x", x_field, drive, drive_args]]
###### construct Hamiltonian
H = hamiltonian(static, dynamic, static_fmt="dia", dtype=np.float64, basis=basis)
print(H.toarray())
