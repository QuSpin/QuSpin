#
import sys, os

quspin_path = os.path.join(os.getcwd(), "../")
sys.path.insert(0, quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import (
    spin_basis_1d,
    spin_basis_general,
)  # Hilbert space spin basis_1d
from quspin.basis.user import user_basis  # Hilbert space user basis
from quspin.basis.user import (
    next_state_sig_32,
    op_sig_32,
    map_sig_32,
)  # user basis data types
from numba import carray, cfunc, jit  # numba helper functions
from numba import uint32, int32  # numba data types
import numpy as np
from itertools import combinations

#
#####
# this test creates a mixed basis with hardcore bosons and spinless fermions
# The code traces out various subsystems of this mixed basis and calculates expectation values
# to see if the partial trace gives consistent results with the pure state.

N_half = 6  # number of sites for each leg of the ladder
N = 2 * N_half  # total number of lattice sites


#
def make_basis(N_half):
    """Generates a list of integers to represent external, user-imported basis"""
    old_basis = spin_basis_general(N_half, m=0)
    #
    states = old_basis.states
    shift_states = np.left_shift(states, N_half)
    #
    shape = states.shape + states.shape
    #
    states_b = np.broadcast_to(states, shape)
    shift_states_b = np.broadcast_to(shift_states, shape)
    # this does the kronecker sum in a more memory efficient way.
    return (states_b + shift_states_b.T).ravel()


#
external_basis = make_basis(N_half)
#
Np = ()  # dummy argument, could be any value (particle conservation should've been


#
############   create soinless fermion user basis object   #############
#
@jit(
    uint32(uint32, uint32),
    locals=dict(
        f_count=uint32,
    ),
    nopython=True,
    nogil=True,
)
def _count_particles_32(state, site_ind):
    # auxiliary function to count number of fermions, i.e. 1's in bit configuration of the state, up to site site_ind
    # CAUTION: 32-bit integers code only!
    f_count = state & ((0x7FFFFFFF) >> (31 - site_ind))
    f_count = f_count - ((f_count >> 1) & 0x55555555)
    f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333)
    return (((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24


#
@cfunc(
    op_sig_32,
    locals=dict(s=int32, sign=int32, n=int32, b=uint32, f_count=uint32),
)
def op(op_struct_ptr, op_str, site_ind, N, args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr, 1)[0]
    err = 0
    #
    site_ind = N - site_ind - 1  # convention for QuSpin for mapping from bits to sites.
    #####
    if 2 * site_ind < N:
        f_count = _count_particles_32(op_struct.state, site_ind)
    else:
        f_count = 0
    #####
    sign = -1 if f_count & 1 else 1
    n = (op_struct.state >> site_ind) & 1  # either 0 or 1
    b = 1 << site_ind
    #
    if op_str == 43:  # "+" is integer value 43 = ord("+")
        op_struct.matrix_ele *= 0.0 if n else sign
        op_struct.state ^= b  # create fermion

    elif op_str == 45:  # "-" is integer value 45 = ord("-")
        op_struct.matrix_ele *= sign if n else 0.0
        op_struct.state ^= b  # create fermion

    elif op_str == 110:  # "n" is integer value 110 = ord("n")
        op_struct.matrix_ele *= n

    elif op_str == 73:  # "I" is integer value 73 = ord("I")
        pass

    else:
        op_struct.matrix_ele = 0
        err = -1
    #
    return err


op_args = np.array([], dtype=np.uint32)


#
######  function to read user-imported basis into QuSpin
#
# function to call when generating next_state
@cfunc(next_state_sig_32)
def next_state(s, counter, N, args):
    # return pre-calculated basis state.
    # add one to counter because the first state is already checked.
    return args[counter + 1]  # = basis


#
next_state_args = (
    external_basis  # this has to be an array of same dtype as the user_basis
)


#
class function_wrapper(object):
    """
    This class provides a wrapper for the user-imported basis,
    as well as the functions required for the `user_basis` functionality.
    #
    This is needed to easily pass parameters (defined as class attributes) to the
    functions `get_so_pcon()` and `get_Ns_pcon`.
    """

    def __init__(self, basis):
        self.basis = basis

    #
    # python function to calculate the starting state to generate the particle conserving basis
    def get_s0_pcon(self, N, Np):
        """calculates the starting state to generate the particle conserving basis."""
        # ignore input arguments as basis is already calculated.
        return self.basis[0]

    #
    # python function to calculate the size of the particle-conserved basis,
    # i.e. BEFORE applying pre_check_state and symmetry maps
    def get_Ns_pcon(self, N, Np):
        """calculates the size of the particle conservation basis (ignoring symmetries at this stage)."""
        # ignore input arguments as basis is already calculated.
        return self.basis.size


#
######  construct user_basis
# define maps dict
maps = dict()
# define particle conservation and op dicts
FW = function_wrapper(external_basis)
noncommuting_bits = [(np.arange(N_half), -1)]
pcon_dict = dict(
    Np=Np,
    next_state=next_state,
    next_state_args=next_state_args,
    get_Ns_pcon=FW.get_Ns_pcon,
    get_s0_pcon=FW.get_s0_pcon,
)
op_dict = dict(op=op, op_args=op_args)
# create user basis
basis = user_basis(
    np.uint32,
    N,
    op_dict,
    allowed_ops=set("n+-"),
    sps=2,
    pcon_dict=pcon_dict,
    noncommuting_bits=noncommuting_bits,
    **maps,
)
# create basis for subsystem
noncommuting_bits = [(np.arange(3), -1)]
subsys_basis = user_basis(
    np.uint32,
    6,
    op_dict,
    allowed_ops=set("n+-"),
    sps=2,
    noncommuting_bits=noncommuting_bits,
)

# pure state in full basis
psi = np.random.normal(0, 1, size=(basis.Ns,))
psi /= np.linalg.norm(psi)

for spin_part in combinations(range(N_half), 3):
    for fermion_part in combinations(range(N_half, 2 * N_half, 1), 3):
        sub_sys_A = np.hstack((spin_part, fermion_part))
        print("testing sub_sys_A: {}".format(sub_sys_A))
        kwargs = dict(
            sub_sys_A=sub_sys_A,
            return_rdm="A",
            subsys_ordering=False,
            enforce_pure=False,
        )
        rho = basis.partial_trace(psi, **kwargs)
        for i, j in combinations(range(sub_sys_A.size), 2):

            J_sub_sys = [[1.0, i, j], [1.0, j, i]]
            J_full = [
                [1.0, sub_sys_A[i], sub_sys_A[j]],
                [1.0, sub_sys_A[j], sub_sys_A[i]],
            ]

            static_ss = [["+-", J_sub_sys]]
            static_full = [["+-", J_full]]

            O_ss = hamiltonian(static_ss, [], basis=subsys_basis, dtype=np.float64)
            O_full = hamiltonian(static_full, [], basis=basis, dtype=np.float64)

            assert np.abs(O_ss.expt_value(rho) - O_full.expt_value(psi)) < 1e-14
