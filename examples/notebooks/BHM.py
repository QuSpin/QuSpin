#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import boson_basis_1d  # Hilbert space boson basis
import numpy as np  # generic math functions

#
##### define model parameters #####
L = 6  # system size
J = 1.0  # hopping
U = np.sqrt(2.0)  # interaction
mu = 0.0  # chemical potential
#
##### construct Bose-Hubbard Hamiltonian #####
# define boson basis with 3 states per site L bosons in the lattice
# basis = boson_basis_1d(L,Nb=L) # full boson basis
# basis = boson_basis_1d(L,Nb=L,sps=3) # particle-conserving basis, 3 states per site
# basis = boson_basis_1d(L,Nb=L,sps=3,kblock=0) # ... and zero momentum sector
# basis = boson_basis_1d(L,Nb=L,sps=3,kblock=1) # ... and first non-zero momentum
basis = boson_basis_1d(
    L, Nb=L, sps=3, kblock=0, pblock=1
)  # ... + zero momentum and positive parity
print(basis)
# define site-coupling lists
hop = [[-J, i, (i + 1) % L] for i in range(L)]  # PBC
interact = [[0.5 * U, i, i] for i in range(L)]  # U/2 \sum_j n_j n_j
pot = [[-mu - 0.5 * U, i] for i in range(L)]  # -(\mu + U/2) \sum_j j_n
# define static and dynamic lists
static = [["+-", hop], ["-+", hop], ["n", pot], ["nn", interact]]
dynamic = []
# build Hamiltonian
H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
# calculate eigensystem
E, V = H.eigh()
E_GS, V_GS = H.eigsh(k=2, which="SA", maxiter=1e6)  # only GS
print("eigenenergies:", E)
print("GS energy is %0.3f" % (E_GS[0]))
# calculate entanglement entropy per site of GS
subsystem = [i for i in range(L // 2)]  # sites contained in subsystem
Sent = basis.ent_entropy(V[:, 0], sub_sys_A=subsystem)["Sent_A"] / L
print("GS entanglement per site is %0.3f" % (Sent))
