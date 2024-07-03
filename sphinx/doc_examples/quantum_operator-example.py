from __future__ import print_function, division

#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
#
from quspin.operators import hamiltonian, quantum_operator  # operators
from quspin.basis import spin_basis_1d  # Hilbert spaces
import numpy as np  # general math functions
from numpy.random import uniform

#
##### setting parameters for simulation
# physical parameters
L = 4  # system size
J = 1.0  # interaction strength
hx = np.sqrt(2)  # transverse field strength
#
###### create the basis
basis = spin_basis_1d(L, pblock=1, pauli=False)  # up basis
##### create model
# define static (fixed) site-coupling lists
J_list = [[J, i, (i + 2) % L] for i in range(L)]  # nnn interaction PBC
hx_list = [[hx, i] for i in range(L)]  # onsite field
# create static lists for H0
operator_list_0 = [["zz", J_list], ["x", hx_list]]
# define parametric lists for H1 (corresponding to operators the coupling of which will be changed)
hz_list = [[1.0, i] for i in range(L)]  # onsite field
operator_list_1 = [["z", hz_list]]
#
###### create operator dictionary for quantum_operator class
# add keys for TFI operators tring list and the longitudinal field
operator_dict = dict(H0=operator_list_0, H1=operator_list_1)
#
###### setting up `quantum_operator` hamiltonian dictionary
H = quantum_operator(operator_dict, basis=basis)
# print Hamiltonian H = H0 + H1
params_dict = dict(
    H0=1.0, H1=1.0
)  # dict containing the couplings to operators in keys of operator_dict
H_lmbda1 = H.tohamiltonian(params_dict)
print(H_lmbda1)
# change z-coupling strength: print Hamiltonian H = H0 + 2H1
params_dict = dict(
    H0=1.0, H1=2.0
)  # dict containing the couplings to operators in keys of operator_dict
H_lmbda2 = H.tohamiltonian(params_dict)
print(H_lmbda2)
