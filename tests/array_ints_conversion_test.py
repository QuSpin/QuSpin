# import sys,os
# qspin_path = os.path.join(os.getcwd(),"../")
# sys.path.insert(0,qspin_path)

import numpy as np
import quspin
from quspin.basis import spin_basis_general
from quspin.tools.misc import ints_to_array, array_to_ints

N = 100
ints_dtype = None

state_array = np.random.randint(0, 2, (1000, N))
ints = array_to_ints(state_array, ints_dtype)
new_array = ints_to_array(ints, N)
assert np.all(state_array == new_array)

N = 50
basis = spin_basis_general(N, make_basis=False)
basis_int = np.random.randint(0, 1 << 50, dtype=basis.dtype)
state_string = basis.int_to_state(basis_int)
state_from_string = np.fromstring(state_string[1:-1], dtype=np.uint8, sep=' ')
state_array = ints_to_array(basis_int, N).flatten()
assert np.all(state_from_string == state_array)