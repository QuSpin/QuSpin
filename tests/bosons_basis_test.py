from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d, boson_basis_1d, fermion_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions



L=3
b = boson_basis_1d(L,Nb=10)

print(b)