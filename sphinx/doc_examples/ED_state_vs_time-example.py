#
import sys, os


#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.tools.measurements import ED_state_vs_time
import numpy as np  # generic math functions

#
L = 12  # syste size
# coupling strenghts
h = 0.8945  # x-field strength
g = 0.945  # z-field strength
# create site-coupling lists
x_field = [[h, i] for i in range(L)]
z_field = [[g, i] for i in range(L)]
# create static and dynamic lists
static_1 = [["x", x_field], ["z", z_field]]
dynamic = []
# create spin-1/2 basis
basis = spin_basis_1d(L, kblock=0, pblock=1)
# set up Hamiltonian
H1 = hamiltonian(static_1, dynamic, basis=basis, dtype=np.float64)
# compute eigensystem of H1
E1, V1 = H1.eigh()
psi1 = V1[:, 14]  # pick any state as initial state
# time-evolve state by decomposing it in an eigensystem (E1,V1)
times = np.linspace(0.0, 5.0, 10)
psi1_time = ED_state_vs_time(psi1, E1, V1, times, iterate=False)
print(type(psi1_time))
# same as above but using a generator
psi1_t = ED_state_vs_time(psi1, E1, V1, times, iterate=True)
print(type(psi1_t))
for i, psi1_n in enumerate(psi1_t):
    print("psi1_n is now the evolved state at time[%i]" % (i))
