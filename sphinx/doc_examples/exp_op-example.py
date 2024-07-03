from __future__ import print_function, division

#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
#
from quspin.operators import hamiltonian, exp_op  # Hamiltonians, operators and exp_op
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
import numpy as np  # generic math functions

#
##### define model parameters #####
L = 4  # system size
J = 1.0  # spin interaction
g = 0.809  # transverse field
h = 0.9045  # parallel field
#
##### construct basis
basis = spin_basis_1d(L=L)
# define PBC site-coupling lists for operators
x_field = [[g, i] for i in range(L)]
z_field = [[h, i] for i in range(L)]
J_nn = [[J, i, (i + 1) % L] for i in range(L)]  # PBC
# static and dynamic lists
static = [["zz", J_nn], ["z", z_field], ["x", x_field]]
dynamic = []
###### construct Hamiltonian
H = hamiltonian(static, dynamic, dtype=np.float64, basis=basis)
#
###### compute evolution operator as matrix exponential
start, stop, N_t = 0.0, 4.0, 21  # time vector parameters
# define evolution operator
U = exp_op(H, a=-1j, start=start, stop=stop, num=N_t, endpoint=True, iterate=True)
print(U)
#
# compute domain wall initial state
dw_str = "".join("1" for i in range(L // 2)) + "".join("0" for i in range(L - L // 2))
i_0 = basis.index(dw_str)  # find index of product state in basis
psi = np.zeros(basis.Ns)  # allocate space for state
psi[i_0] = 1.0  # set MB state to be the given product state
#
##### calculate time-evolved state by successive application of matrix exponential
psi_t = U.dot(
    psi
)  # create generator object to apply matrix exponential on the initial state
print(psi_t)
for psi_i in psi_t:
    print("evolved state:", psi_i)
