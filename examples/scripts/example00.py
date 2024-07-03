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
#
# line 12 and line 13 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, qspin_path)
#####################################################################
#                            example 00                             #
#    In this script we demonstrate how to use QuSpin's              #
#    `basis_general` routines to construct, interpret,              #
#    and use basis objects.                                         #
#####################################################################
from quspin.basis import spin_basis_general  # Hilbert space spin basis
import numpy as np  # generic math functions

#
L = 2  # system size
sites = np.arange(L)  # define sites
P = sites[::-1]  # define action of parity/reflection on the lattice sites
#
#############################################
print("\n----------------------------")
print("---  NO  SYMMETRIES  -------")
print("----------------------------\n")
#############################################
#
##### basis objects without symmetries
#
basis = spin_basis_general(
    L,
)
Ns = basis.Ns  # number of states in the basis
#
print(basis)
#
# states in integer representation
states = (
    basis.states
)  # = third column when printing the basis object (not consecutive if there are symmetries --> see below)
array_inds = np.arange(
    basis.Ns
)  # = first column when printing the basis object (always consecutive)
#
print("\n'array index' and 'states' columns when printing the basis:")
print("array indices:", array_inds)
print("states in int rep:", states)
# find array index of a state from its integer representation; Note: the array index is crucial for reading-off matrix elements
s = basis.states[2]
array_ind_s = basis.index(
    s
)  # = array_inds[2] whenever there are no symmetries defined in the basis
print(
    "\nprint array index of s, and s (in int rep); Note: the array index is crucial for reading-off matrix elements"
)
print(array_ind_s, s)
# --------------------------------------------
##### States: ket and integer representations
# --------------------------------------------
# find integer representation from Fock state string
fock_state_str_s = "|01>"  # works also if the ket-forming strings | > are omitted
int_rep_s = basis.state_to_int(fock_state_str_s)
print("\nprint Fock state string of s, and s (in int rep):")
print(fock_state_str_s, int_rep_s)
#
# find Fock state string from integer representation
fock_s = basis.int_to_state(int_rep_s, bracket_notation=True)
print("\nprint Fock state string of s, and s (in int rep):")
print(fock_s, int_rep_s)
# same as above but dropping the ket-forming strings | >
fock_s = basis.int_to_state(int_rep_s, bracket_notation=False)
print("print Fock state string (without | and >) of s, and s (in int rep):")
print(fock_s, int_rep_s)
#
# find Fock state from array index
array_ind_s = 2
int_rep_s = basis.states[array_ind_s]
fock_s = basis.int_to_state(int_rep_s, bracket_notation=True)
print("\nprint array index, int rep, and fock state rep of s:")
print(array_ind_s, int_rep_s, fock_s)  # compare with print(basis) output
# --------------------------------------------
##### States: array/vector representation
# --------------------------------------------
# define a zero vector of size set by the basis dimenion
psi_s = np.zeros(basis.Ns)
# obtain array index for the fock state |01>
array_ind_s = basis.index(basis.state_to_int("01"))
# construct the pure state |01>
psi_s[array_ind_s] = 1.0
print("\nprint state psi_s in the basis:")
print(psi_s)
#
#############################################
print("\n\n\n----------------------------")
print("-------  SYMMETRIES  -------")
print("----------------------------\n")
#############################################
#
##### basis objects with symmetries
#
basis_triplet = spin_basis_general(L, pblock=(P, 0))
basis_singlet = spin_basis_general(L, pblock=(P, 1))
#
print("print full basis:")
print(basis)
#
print("\n\nprint pblock=+1 basis:\n")
print(basis_triplet)
#
print(
    "\n  * integer rep column no longer consecutive! (|01> falls outside symmetry sector)"
)
print(
    "  * array index column still consecutive! (but indices differ compared to full basis, e.g. for |00>)"
)
print(
    "  * |11> and |00> invariant under parity, so they correspond to physical states |11> and |00>"
)
print(
    "  * |10> not invariant under parity! It represents the physical symmetric superposition 1/sqrt(2)(|10> + |01>) [see bottom note when printing the symmetry-reduced basis]; quspin keeps track of the coefficient 1/sqrt(2) under the hood."
)
print("\n\nprint pblock=-1 basis:\n")
#
print(basis_singlet)
print(
    "  * |10> here represents the physical ANTI-symmetric superposition 1/sqrt(2)(|10> - |01>) [see bottom note when printing the symmetry-reduced basis]"
)
print(
    "  *  NOTE: same state |01> is used to label both the symmetric and antisymmetric superposition because in this cases quspin uses the smallest integer from the integer representations of the states comprising the superposition states.\n"
)
#
# --------------------------------------------------
##### transform states from one basis to the other
# --------------------------------------------------
#
array_ind_s = basis_triplet.index(basis.state_to_int("10"))
psi_symm_s = np.zeros(basis_triplet.Ns)
psi_symm_s[array_ind_s] = 1.0  # create the state |10> + |01> in basis_triplet
print("print state psi_symm_s in the symmetry-reduced basis_triplet:")
print(psi_symm_s)
#
# compute corresponding state in the full basis
psi_s = basis_triplet.project_from(psi_symm_s, sparse=False)
print(
    "\nprint state psi_s in the full basis: (note the factor 1/sqrt(2) which comes out correct."
)
print(psi_s)
#
# one can also project a full-basis state to a symmetry-reduced basis
psi_s = np.zeros(basis.Ns)
array_ind_s = basis.index(basis.state_to_int("01"))
psi_s[array_ind_s] = 1.0  # create the state |01> in the full basis
#
psi_symm_s = basis_singlet.project_to(
    psi_s, sparse=False
)  # projects |01> to 1/sqrt(2) (|01> - |10>) in basis_singlet
print(
    "\nprint state psi_symm_s in the symmetry-reduced basis_singlet; NOTE: projection does not give a normalized state!"
)
print(psi_symm_s)
# normalize
psi_symm_s = psi_symm_s / np.linalg.norm(psi_symm_s)
#
# lift the projected state back to full basis
psi_symm_s = np.expand_dims(
    psi_symm_s, 0
)  # required only when the basis_singlet contains a single state
psi_lifted_s = basis_singlet.project_from(
    psi_symm_s, sparse=False
)  # corresponds to the projection 1/sqrt(2) (|01> - |10>) in the full basis
print(
    "\nprint state psi_lifted_s = 1/sqrt(2) (|01> - |10>) in the full basis; NOTE: info lost by the first projection!"
)
print(psi_lifted_s)
