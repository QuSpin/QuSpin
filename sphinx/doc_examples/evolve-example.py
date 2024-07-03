#
import sys, os


#
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import boson_basis_1d  # Hilbert space spin basis
from quspin.tools.evolution import evolve  # ODE evolve tool
from quspin.tools.Floquet import Floquet_t_vec  # stroboscopic time vector
import numpy as np  # generic math functions
from six import iteritems  # loop over elements of dictionary

#
L = 50  # number of lattice sites
i_CM = L // 2 - 0.5  # centre of chain
#
### static model parameters
J = 1.0  # hopping
kappa_trap = 0.002  # harmonic trap strength
U = 1.0  # mean-field (GPE) interaction
#
### periodic driving
A = 1.0  # drive amplitude
Omega = 10.0  # drive frequency


def drive(t, Omega):
    return np.exp(-1j * A * np.sin(Omega * t))


def drive_conj(t, Omega):
    return np.exp(+1j * A * np.sin(Omega * t))


drive_args = [Omega]  # drive arguments
t = Floquet_t_vec(Omega, 30, len_T=1)  # time vector, 30 stroboscopic periods
#
### site-couping lists
hopping = [[-J, i, (i + 1) % L] for i in range(L)]
trap = [[kappa_trap * (i - i_CM) ** 2, i] for i in range(L)]
#
### operator strings for single-particle Hamiltonian
static = [["n", trap]]
dynamic = [["+-", hopping, drive, drive_args], ["-+", hopping, drive_conj, drive_args]]
# define single-particle basis
basis = boson_basis_1d(
    L, Nb=1, sps=2
)  # Nb=1 boson and sps=2 states per site [empty and filled]
#
### build Hamiltonian
H = hamiltonian(static, dynamic, basis=basis, dtype=np.complex128)
# calculate eigenvalues and eigenvectors of free particle in a harmonic trap
E, V = H.eigh(time=0)
# initial state normalised to one partcile per site
phi0 = V[:, 0] * np.sqrt(L)


#######
def GPE(time, phi):
    """Solves the complex-valued time-dependent Gross-Pitaevskii equation:"""
    # integrate static part of GPE
    phi_dot = -1j * (H.static.dot(phi) + U * np.abs(phi) ** 2 * phi)
    # integrate dynamic part of GPE
    for fun, Hdyn in iteritems(H.dynamic):
        phi_dot += -1j * fun(time) * Hdyn.dot(phi)

    return phi_dot


# solving the complex-valued GPE takes one line
phi_t = evolve(phi0, t.i, t.vals, GPE)


########
def GPE_real(time, phi, H, U):
    """Solves the Gross-Pitaevskii equation, cast into real-valued form so it can be solved with a
    real-valued ODE solver.

    """
    # preallocate memory for phi_dot
    phi_dot = np.zeros_like(phi)
    # read off number of lattice sites (array dimension of phi)
    Ns = H.Ns
    # static single-particle part
    phi_dot[:Ns] = H.static.dot(phi[Ns:]).real
    phi_dot[Ns:] = -H.static.dot(phi[:Ns]).real
    # static GPE interaction
    phi_dot_2 = np.abs(phi[:Ns]) ** 2 + np.abs(phi[Ns:]) ** 2
    phi_dot[:Ns] += U * phi_dot_2 * phi[Ns:]
    phi_dot[Ns:] -= U * phi_dot_2 * phi[:Ns]
    # dynamic single-particle term
    for func, Hdyn in iteritems(H.dynamic):
        fun = func(time)  # evaluate drive
        phi_dot[:Ns] += (
            +(fun.real) * Hdyn.dot(phi[Ns:]) + (fun.imag) * Hdyn.dot(phi[:Ns])
        ).real
        phi_dot[Ns:] += (
            -(fun.real) * Hdyn.dot(phi[:Ns]) + (fun.imag) * Hdyn.dot(phi[Ns:])
        ).real

    return phi_dot


# define ODE solver parameters
GPE_params = (H, U)
# solving the real-valued GPE takes one line
phi_t = evolve(phi0, t.i, t.vals, GPE_real, stack_state=True, f_params=GPE_params)
