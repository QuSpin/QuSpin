"""
This is an exmple quspin application which can be used to benchmark the package performance.

python quspin_app L N_T  ,  e.g.,  python quspin_app.py 4 4 5 10000

L: (3<L<7) linear system size; runtime scales exponentially with L
N_T: number of time steps; runtime scales linearly with N_T

Multithreading:

set number of MKL threads
set number of OMP threads

"""

import argparse
import sys, os


from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
from quspin.tools.evolution import expm_multiply_parallel
import numpy as np
import time
from scipy.sparse.linalg import eigsh


parser = argparse.ArgumentParser()

parser.add_argument(
    "-o", "--output", help="Directs the output to a name of your choice."
)
parser.add_argument(
    "--fast_test",
    type=bool,
    nargs="?",
    const=True,
    default=False,
    help="Does a quick run of the test.",
)
args = parser.parse_args()


###### simulation parameters

# time evolution parameters: expect linear slowdown with incerasing N_T
N_T = 10  # int(sys.argv[2])
T = 1.0

# system size: expect exponential slowdown of simulation time with increasinf L

# test
if args.fast_test:
    L = 28
else:
    L = 34  # or 36 with 10 OMP.MKL threads


output_str = []


######## BASIS CONSTRUCTION ########
#
# required time scales exponentially with L


p = np.arange(L)[::-1]
t = (np.arange(L) + 1) % L
z = -(np.arange(L) + 1)

ti = time.time()
basis = spin_basis_general(L, S="1/2", m=0, kblock=(t, 0), pblock=(p, 0), zblock=(z, 0))
tf = time.time()

time_basis = tf - ti
basis_str = "\nbasis with {0:d} states took {1:0.2f} secs.\n".format(
    basis.Ns, time_basis
)
output_str.append(basis_str)
print(basis_str)


######## HAMILTONIAN CONSTRUCTION ########
#
# required time scales exponentially with L
# linear speedup is expected from both OMP and MKL


def drive(t, T):
    return np.sin(0.5 * np.pi * t / T) ** 2


J_list = [[1.0, i, t[i]] for i in range(L)]
static = [
    ["+-", J_list],
    ["-+", J_list],
]
dynamic = [
    ["zz", J_list, drive, (T,)],
]

ti = time.time()
H = hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
H_op = H.aslinearoperator()
tf = time.time()
time_H = tf - ti
H_str = "\nHamiltonian construction took {0:0.2f} secs.\n".format(time_H)
output_str.append(H_str)
print(H_str)


######## HAMILTONIAN DIAGONALIZATION ########
#
# required time scales exponentially with L
# linear speedup is expected both OMP and MKL


ti = time.time()
eigsh(H_op, k=1, which="SA")
tf = time.time()
time_eigsh_op = tf - ti
Hop_str = "\ncomputing ground state 'aslinearoperator' took {0:0.2f} secs.\n".format(
    time_eigsh_op
)
output_str.append(Hop_str)
print(Hop_str)


# calculate minimum and maximum energy only
ti = time.time()
Es, psi_s = H.eigsh(
    k=1,
    which="SA",
    maxiter=1e4,
)
tf = time.time()
time_eigsh = tf - ti
GS_str = "\ncomputing ground state 'hamiltonian' state took {0:0.2f} secs.\n".format(
    time_eigsh
)
output_str.append(GS_str)
print(GS_str)


######## TIME EVOLUTION ########
#
# required time scales linearly with N_T and exponentially with L
# linear speedup is expected from OMP


# calculate the eigenstate closest to energy E_star
psi_i = psi_s[:, 0]

ti = time.time()
times = np.linspace(0.0, T, num=N_T)
psi_t = H.evolve(psi_i, times[0], times, iterate=True)
for psi in psi_t:
    pass
tf = time.time()
time_SE = tf - ti
SE_str = "\ncomputing SE took {0:0.2f} secs.\n".format(time_SE)
output_str.append(SE_str)
print(SE_str)


dt = T / N_T
expH = expm_multiply_parallel(H.tocsr(), a=-1j * dt, dtype=np.complex128)
#
# auxiliary array for memory efficiency
work_array = np.zeros(
    (2 * len(psi),), dtype=psi.dtype
)  # twice as long because complex-valued
#
# loop ober the time steps
ti = time.time()
for j in range(N_T):
    #
    # apply to state psi and update psi in-place
    expH.dot(psi, work_array=work_array, overwrite_v=True)
tf = time.time()
time_expm = tf - ti
exp_str = "\ntime evolution took {0:0.2f} secs.\n".format(time_expm)
output_str.append(exp_str)
print(exp_str)


time_tot = time_basis + time_H + time_eigsh_op + time_SE + time_expm
tot_str = "\n\ntotal run time: {0:0.2f} secs.\n".format(time_tot)
output_str.append(tot_str)
print(tot_str)


# store output to file
if args.output is not None:
    with open(args.output, "w") as f:
        for string in output_str:
            f.write(string)
