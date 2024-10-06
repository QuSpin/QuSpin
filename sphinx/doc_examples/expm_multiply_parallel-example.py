#
import sys, os


#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_1d  # bosonic Hilbert space
from quspin.tools.evolution import expm_multiply_parallel  # expm_multiply_parallel
import numpy as np  # general math functions

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
static = [["zz", J_zz], ["x", x_field], ["z", z_field]]
dynamic = []
# create spin-1/2 basis
basis = spin_basis_1d(L, kblock=0, pblock=1)
# set up Hamiltonian
H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
# prealocate computation of matrix exponential
expH = expm_multiply_parallel(H.tocsr(), a=0.2j)
print(expH)
#
##### change value of `a`
print(expH.a)
expH.set_a(0.3j)
print(expH.a)
#
##### compute expm_multiply applied on a state
_, psi = H.eigsh(k=1, which="SA")  # compute GS of H
psi = psi.squeeze().astype(
    np.complex128
)  # cast array type to complex double due to complex matrix exp
# construct c++ work array for speed
work_array = np.zeros((2 * len(psi),), dtype=psi.dtype)
print(H.expt_value(psi))  # measure energy of state |psi>
expH.dot(
    psi, work_array=work_array, overwrite_v=True
)  # compute action of matrix exponential on a state
print(H.expt_value(psi))  # measure energy of state exp(aH)|psi>
