from __future__ import print_function, division

#
import sys, os

os.environ["KMP_DUPLICATE_LIB_OK"] = (
    "True"  # uncomment this line if omp error occurs on OSX for python 3
)
os.environ["OMP_NUM_THREADS"] = "1"  # set number of OpenMP threads to run in parallel
os.environ["MKL_NUM_THREADS"] = "1"  # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
######################################################################
#                            example 25                              #
# This example shows how to define the Sachdev-Ye-Kitaev Hamiltonian #
######################################################################
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_general  # Hilbert space fermion basis
import numpy as np
from scipy.special import factorial

#
# set seed of RNG
seed = 0
np.random.seed(seed)
#
#
##### model parameters #####
#
L = 6  # number of lattice sites
#
J = np.random.normal(
    size=(L, L, L, L)
)  # random interaction J_{ijkl} of zero mean and unit variance
#
##### create Hamiltonian #####
#
# site-coupling list
SYK_int = [
    [-1.0 / factorial(4.0) * J[i, j, k, l], i, j, k, l]
    for i in range(L)
    for j in range(L)
    for k in range(L)
    for l in range(L)
]
# static list
static = [
    ["xxxx", SYK_int],
]
# static=[['yyyy',SYK_int],] # alternative definition, equivalent spectrum
#
##### create basis #####
#
basis = spinless_fermion_basis_general(
    L,
)
#
##### create Hamiltonian #####
#
H_SYK = hamiltonian(
    static, [], basis=basis, dtype=np.float64
)  # caution: matrix is NOT sparse (!)
#
# compute entire spectrum
E = H_SYK.eigvalsh()
#
# print the lowest 4 eigenvalues
print(E[:4])
