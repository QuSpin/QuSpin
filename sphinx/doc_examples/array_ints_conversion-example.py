from __future__ import print_function, division

#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../../")
sys.path.insert(0, quspin_path)
#
import numpy as np
from quspin.basis import spin_basis_general
from quspin.tools.misc import ints_to_array, array_to_ints

N = 10
basis = spin_basis_general(N, make_basis=False)

# initial states
num_states = 2
state_array = np.random.randint(0, 2, (num_states, N), dtype=np.uint8)
print(f"The {num_states} initial states are")
print(state_array)

# convert state array to basis integers
basis_ints = array_to_ints(state_array, basis.dtype)

# apply Sx on site 4
ME, bra, ket = basis.Op_bra_ket("x", [4], 1.0, np.float64, basis_ints)

# convert outcome back to state array
bra_array = ints_to_array(bra, N)

print("Applying Sx on site 4, the new states are")
print(bra_array)
