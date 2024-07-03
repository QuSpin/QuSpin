#
import sys, os


#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.tools.Floquet import Floquet, Floquet_t_vec  # Floquet Hamiltonian
import numpy as np  # generic math functions

#
##### define model parameters #####
L = 10  # system size
J = 1.0  # spin interaction
g = 0.809  # transverse field
h = 0.9045  # parallel field
Omega = 4.5  # drive frequency


#
##### set up alternating Hamiltonians #####
# define time-reversal symmetric periodic step drive
def drive(t, Omega):
    return np.sign(np.cos(Omega * t))


drive_args = [Omega]
# compute basis in the 0-total momentum and +1-parity sector
basis = spin_basis_1d(L=L, a=1, kblock=0, pblock=1)
# define PBC site-coupling lists for operators
x_field_pos = [[+g, i] for i in range(L)]
x_field_neg = [[-g, i] for i in range(L)]
z_field = [[h, i] for i in range(L)]
J_nn = [[J, i, (i + 1) % L] for i in range(L)]  # PBC
# static and dynamic lists
static = [["zz", J_nn], ["z", z_field], ["x", x_field_pos]]
dynamic = [
    ["zz", J_nn, drive, drive_args],
    ["z", z_field, drive, drive_args],
    ["x", x_field_neg, drive, drive_args],
]
# compute Hamiltonian
H = 0.5 * hamiltonian(static, dynamic, dtype=np.float64, basis=basis)
##### define time vector of stroboscopic times with 1 driving cycles and 10 points per cycle #####
t = Floquet_t_vec(
    Omega, 1, len_T=10
)  # t.vals=times, t.i=initial time, t.T=drive period
#
##### calculate exact Floquet eigensystem #####
t_list = (
    np.array([0.0, t.T / 4.0, 3.0 * t.T / 4.0]) + np.finfo(float).eps
)  # times to evaluate H
dt_list = np.array(
    [t.T / 4.0, t.T / 2.0, t.T / 4.0]
)  # time step durations to apply H for
Floq = Floquet(
    {"H": H, "t_list": t_list, "dt_list": dt_list}, VF=True
)  # call Floquet class
VF = Floq.VF  # read off Floquet states
EF = Floq.EF  # read off quasienergies
