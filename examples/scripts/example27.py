from __future__ import print_function, division

#
import sys, os

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # uncomment this line if omp error occurs on OSX for python 3
)
os.environ["OMP_NUM_THREADS"] = "4"  # set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = "4"  # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
###########################################################################
#                             example 27                                  #
#  In this script we demonstrate how to use QuSpin to generate            #
#  a Hamiltonian and solve the Louiville-von Neumann equation starting	  #
#  from a mixed initial state in the Fermi Hubbard model. We also         #
#  show how to write a simple fixed time-step Runge-Kutta solver          #
#  that makes use of an MKL-parllelized dot function for sparse matrices. #
###########################################################################
# import sparse_dot library, see https://github.com/flatironinstitute/sparse_dot.git
from sparse_dot_mkl import dot_product_mkl
from quspin.tools.misc import get_matvec_function
from quspin.operators import hamiltonian
from scipy.sparse import csr_matrix
from quspin.basis import spinful_fermion_basis_general
import numpy as np
import time

#
##### define model parameters #####
#
Lx, Ly = 2, 2  # expect to see an MKL speedup from Lx,Ly = 2,3 onward
N_2d = Lx * Ly  # total number of lattice sites
# model params
J = 1.0  # hopping amplitude
U = np.sqrt(2.0)  # interaction strength
# time parameters
t_max = 40.0  # total time
dt = 0.1  # time step size
N_T = int(t_max // dt)
time_vec = np.linspace(0.0, t_max, int(2 * t_max + 1))
#
##### create Hamiltonian to evolve unitarily
#
# basis
basis = spinful_fermion_basis_general(N_2d)
# define translation operators for 2D lattice
s = np.arange(N_2d)  # sites [0,1,2,...,N_2d-1] in simple notation
x = s % Lx  # x positions for sites
y = s // Lx  # y positions for sites
T_x = (x + 1) % Lx + Lx * y  # translation along x-direction
T_y = x + Lx * ((y + 1) % Ly)  # translation along y-direction
# site-coupling lists
hop_left = [[-J, i, T_x[i]] for i in range(N_2d)] + [
    [-J, i, T_y[i]] for i in range(N_2d)
]
hop_right = [[+J, i, T_x[i]] for i in range(N_2d)] + [
    [+J, i, T_y[i]] for i in range(N_2d)
]
int_list = [[U, i, i] for i in range(N_2d)]
# static opstr list
static = [
    ["+-|", hop_left],  # up hop left
    ["-+|", hop_right],  # up hop right
    ["|+-", hop_left],  # down hop left
    ["|-+", hop_right],  # down hop right
    ["n|n", int_list],
]
# construct Hamiltonian
Hcsc = hamiltonian(
    static, [], dtype=np.complex128, basis=basis, check_symm=False
).tocsr()
#
##### create the mean-field groundstate we start from
#
# compute basis with single occupancies only
basis_reduced = spinful_fermion_basis_general(
    N_2d, Nf=([(j, N_2d - j) for j in range(N_2d + 1)]), double_occupancy=False
)
# create empty list to store indices of nonzero elements for initial DM
rho_inds = []
for s in basis_reduced.states:  # loop over singly-occupied states
    rho_inds.append(np.argwhere(basis.states == s)[0][0])
# create initial state in csr format
rho_0 = csr_matrix(
    (np.ones(basis_reduced.Ns) / basis_reduced.Ns, (rho_inds, rho_inds)),
    shape=(basis.Ns, basis.Ns),
    dtype=np.complex128,
)


#
##### define Runge-Kutta solver for sparse matrix
#
# MKL-parallel function using the sparse_dot library
def LvN_mkl(rho):
    # define right-hand side of Liouville-von Neumann equation
    # see https://github.com/flatironinstitute/sparse_dot.git, needs v0.8 or higher
    return -1j * (
        dot_product_mkl(Hcsc, rho, cast=False) - dot_product_mkl(rho, Hcsc, cast=False)
    )


#
# scipy function
def LvN_scipy(rho):
    return -1j * (Hcsc @ rho - rho @ Hcsc)


#
# define fixed step-size Runge-Kutta 4th order method
def RK_solver(rho, dt, LvN):
    k1 = LvN(rho)
    k2 = LvN(rho + (0.5 * dt) * k1)
    k3 = LvN(rho + (0.5 * dt) * k2)
    k4 = LvN(rho + dt * k3)
    return rho + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


#
##### evolve DM by solving the LvN equation
#
# empty list to store the solution in
rho_t = []
# initial state
rho_mkl = rho_0.copy()
# time evolution loop
starttime = time.time()
for i in range(N_T):
    rho_mkl = RK_solver(rho_mkl, dt, LvN_mkl)
    rho_t.append(rho_mkl)
    # print("finished step {0:d}/{1:d}.".format(i+1,int(t_max/dt)-1),flush=True)
#
print(
    "\nMKL time evo done in {0:0.4f} secs.".format(time.time() - starttime), flush=True
)
#
# empty list to store the solution in
rho_t = []
# initial state
rho_scipy = rho_0.copy()
# time evolution loop
starttime = time.time()
for i in range(N_T):
    rho_scipy = RK_solver(rho_scipy, dt, LvN_scipy)
    rho_t.append(rho_scipy)
    # print("finished step {0:d}/{1:d}.".format(i+1,int(t_max/dt)-1),flush=True)
#
print(
    "\nScipy time evo done in {0:0.4f} secs.".format(time.time() - starttime),
    flush=True,
)
