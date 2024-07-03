#
import sys, os


#
from quspin.operators import hamiltonian, commutator  # Hamiltonians and operators
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.tools.measurements import diag_ensemble
import numpy as np  # generic math functions

#
L = 12  # syste size
# coupling strenghts
J = 1.0  # spin-spin coupling
h = 0.8945  # x-field strength
g = 0.945  # z-field strength
# create site-coupling lists
J_zz = [[J, i, (i + 1) % L] for i in range(L)]  # PBC
x_field = [[h, i] for i in range(L)]
z_field = [[g, i] for i in range(L)]
# create static and dynamic lists
static_1 = [["x", x_field], ["z", z_field]]
static_2 = [["zz", J_zz], ["x", x_field], ["z", z_field]]
dynamic = []
# create spin-1/2 basis
basis = spin_basis_1d(L, kblock=0, pblock=1)
# set up Hamiltonian
H1 = hamiltonian(static_1, dynamic, basis=basis, dtype=np.float64)
H2 = hamiltonian(static_2, dynamic, basis=basis, dtype=np.float64)
###### calculate commutator
comm = commutator(H1, H2)
