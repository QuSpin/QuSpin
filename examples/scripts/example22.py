from __future__ import print_function, division

#
import sys, os

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # uncomment this line if omp error occurs on OSX for python 3
)
os.environ["OMP_NUM_THREADS"] = "4"  # set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = "1"  # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
###########################################################################
#                            example 22                                   #
#  This example shows the usage of the function `expm_multiply_parallel   #
#  to do time evolution for piece-wise constatnt Hamiltonians. For this   #
#  purpose, we show a simulation of a periodically-driven Heinseberg-ike  #
#  spin-1 system.                                                         #
###########################################################################
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import expm_multiply_parallel
import numpy as np
import time

#
##### define data type for the simulation
dtype_real = np.float64
dtype_cmplx = np.result_type(dtype_real, np.complex64)
#
##### define model parameters #####
L = 12  # system size
Jxy = np.sqrt(2.0)  # xy interaction
Jzz_0 = 1.0  # zz interaction
hz = 1.0 / np.sqrt(3.0)  # z external field
T = 1.5  # period of switching for periodic drive
N_steps = 100  # number of driving cycles to evolve for
#
##### define Hamiltonians H_0, H_1 and H_ave
# build the spin-1 basis in the zero magnetization, positive parity and zero-momentum sector
basis = spin_basis_1d(
    L,
    S="1",
    m=0,
    kblock=0,
    pblock=1,
)
print("total number of basis states {}.\n".format(basis.Ns))
# define operators with OBC using site-coupling lists
J_zz = [[Jzz_0, i, (i + 1) % L] for i in range(L)]  # PBC
J_xy = [[0.5 * Jxy, i, (i + 1) % L] for i in range(L)]  # PBC
h_z = [[hz, i] for i in range(L)]
# static and dynamic lists
static_0 = [
    ["+-", J_xy],
    ["-+", J_xy],
]
static_1 = [
    ["zz", J_zz],
    ["z", h_z],
]
dynamic = []
# compute the time-dependent Heisenberg Hamiltonian
H0 = hamiltonian(static_0, dynamic, basis=basis, dtype=dtype_real)
H1 = hamiltonian(static_1, dynamic, basis=basis, dtype=dtype_real)
H_ave = 0.5 * (H0 + H1)
#
##### compute the initial state
# calculate ground state of H_ave
E, V = H_ave.eigsh(k=1, which="SA")
psi_i = V[:, 0]
#
# preallocate arrays for the observables
#
E_density = np.zeros(N_steps + 1, dtype=dtype_real)
Sent_density = np.zeros(N_steps + 1, dtype=dtype_real)
# compute initial values for obsrvables
E_density[0] = H_ave.expt_value(psi_i).real / L
Sent_density[0] = basis.ent_entropy(psi_i, sub_sys_A=range(L // 2), density=True)[
    "Sent_A"
]
#
##### compute the time evolution using expm_multiply_parallel
#
# construct piece-wise constant unitaries
expH0 = expm_multiply_parallel(H0.tocsr(), a=-1j * 0.5 * T, dtype=dtype_cmplx)
expH1 = expm_multiply_parallel(H1.tocsr(), a=-1j * 0.5 * T, dtype=dtype_cmplx)
#
# auxiliary array for memory efficiency
psi = psi_i.copy().astype(np.complex128)
work_array = np.zeros(
    (2 * len(psi),), dtype=psi.dtype
)  # twice as long because complex-valued
#
# loop ober the time steps
for j in range(N_steps):
    #
    # apply to state psi and update psi in-place
    expH0.dot(psi, work_array=work_array, overwrite_v=True)
    expH1.dot(psi, work_array=work_array, overwrite_v=True)
    # measure 'oservables'
    E_density[j + 1] = H_ave.expt_value(psi).real / L
    Sent_density[j + 1] = basis.ent_entropy(psi, sub_sys_A=range(L // 2), density=True)[
        "Sent_A"
    ]
    #
    print("finished evolving {0:d} step".format(j))
#
# compute Page-corrected entanglement entropy value
m = basis.sps ** (L // 2)
n = basis.sps**L
s_page = (np.log(m) - m / (2.0 * n)) / (L // 2)
#
#
##### Plot data
#
import matplotlib.pyplot as plt  # plotting library

#
times = T * np.arange(N_steps + 1)
#
plt.plot(times, E_density, "-b", label="$\\mathcal{E}_\\mathrm{ave}(\\ell T)$")
plt.plot(times, Sent_density, "-r", label="$s_\\mathrm{ent}(\\ell T)$")
plt.plot(times, s_page * np.ones_like(times), "--r", label="$s_\\mathrm{Page}$")
plt.xlabel("$\\ell T$")
# plt.xlim(-T,T*(N_steps+1))
plt.legend()
plt.grid()
plt.tight_layout()
#
# plt.show()
