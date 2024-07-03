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
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spin_basis_1d  # Hilbert space spin basis_1d
from quspin.basis.user import user_basis  # Hilbert space user basis
from quspin.basis.user import (
    next_state_sig_32,
    op_sig_32,
    map_sig_32,
    count_particles_sig_32,
)  # user basis data types signatures
from numba import carray, cfunc  # numba helper functions
from numba import uint32, int32  # numba data types
import numpy as np
from scipy.special import comb

#
N = 6  # lattice sites
Np = N // 2  # total number of spin ups


#
############   create spin-1/2 user basis object   #############
#
######  function to call when applying operators
@cfunc(
    op_sig_32,
    locals=dict(s=int32, n=int32, b=uint32),
)
def op(op_struct_ptr, op_str, site_ind, N, args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr, 1)[0]
    err = 0
    #
    site_ind = N - site_ind - 1  # convention for QuSpin for mapping from bits to sites.
    n = (op_struct.state >> site_ind) & 1  # either 0 or 1
    s = (((op_struct.state >> site_ind) & 1) << 1) - 1  # either -1 or 1
    b = 1 << site_ind
    #
    if op_str == 120:  # "x" is integer value 120 = ord("x")
        op_struct.state ^= b

    elif op_str == 121:  # "y" is integer value 120 = ord("y")
        op_struct.state ^= b
        op_struct.matrix_ele *= 1.0j * s

    elif op_str == 43:  # "+" is integer value 43 = ord("+")
        if n:
            op_struct.matrix_ele = 0
        else:
            op_struct.state ^= b  # create spin

    elif op_str == 45:  # "-" is integer value 45 = ord("-")
        if n:
            op_struct.state ^= b  # destroy spin
        else:
            op_struct.matrix_ele = 0

    elif op_str == 122:  # "z" is integer value 120 = ord("z")
        op_struct.matrix_ele *= s

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
######  function to implement magnetization/particle conservation
#
@cfunc(
    next_state_sig_32,
    locals=dict(t=uint32),
)
def next_state(s, counter, N, args):
    """implements magnetization conservation."""
    if s == 0:
        return s
    #
    t = (s | (s - 1)) + 1
    return t | ((((t & (0 - t)) // (s & (0 - s))) >> 1) - 1)


next_state_args = np.array([], dtype=np.uint32)  # compulsory, even if empty


# python function to calculate the starting state to generate the particle conserving basis
def get_s0_pcon(N, Np):
    return sum(1 << i for i in range(Np))


# python function to calculate the size of the particle-conserved basis,
# i.e. BEFORE applying pre_check_state and symmetry maps
def get_Ns_pcon(N, Np):
    return comb(N, Np, exact=True)


#
######  define symmetry maps
#
@cfunc(
    map_sig_32,
    locals=dict(
        shift=uint32,
        xmax=uint32,
        x1=uint32,
        x2=uint32,
        period=int32,
        l=int32,
    ),
)
def translation(x, N, sign_ptr, args):
    """works for all system sizes N."""
    shift = args[0]  # translate state by shift sites
    period = N  # periodicity/cyclicity of translation
    xmax = args[1]
    #
    l = (shift + period) % period
    x1 = x >> (period - l)
    x2 = (x << l) & xmax
    #
    return x2 | x1


T_args = np.array([1, (1 << N) - 1], dtype=np.uint32)


#
@cfunc(
    map_sig_32,
    locals=dict(
        out=uint32,
        s=int32,
    ),
)
def parity(x, N, sign_ptr, args):
    """works for all system sizes N."""
    out = 0
    s = args[0]  # N-1
    #
    out ^= x & 1
    x >>= 1
    while x:
        out <<= 1
        out ^= x & 1
        x >>= 1
        s -= 1
    #
    out <<= s
    return out


P_args = np.array([N - 1], dtype=np.uint32)


#
@cfunc(
    map_sig_32,
    locals=dict(
        xmax=uint32,
    ),
)
def spin_inversion(x, N, sign_ptr, args):
    """works for all system sizes N."""
    xmax = args[0]  # maximum integer
    return x ^ xmax


Z_args = np.array([(1 << N) - 1], dtype=np.uint32)


#
######  define function to count particles in bit representation
#
@cfunc(count_particles_sig_32, locals=dict(s_count=uint32))
def count_particles(x, p_number_ptr, args):
    """Counts number of particles/spin-ups in a state stored in integer representation for up to N=32 sites"""
    #
    s_count = x & ((0x7FFFFFFF) >> (31 - args[0]))
    s_count = s_count - ((s_count >> 1) & 0x55555555)
    s_count = (s_count & 0x33333333) + ((s_count >> 2) & 0x33333333)
    s_count = (((s_count + (s_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    #
    p_number_ptr[0] = s_count


n_sectors = 1  # number of particle sectors
count_particles_args = np.array([N], dtype=np.int32)
#
######  construct user_basis
# define maps dict
maps = dict(
    T_block=(translation, N, 0, T_args),
    P_block=(parity, 2, 0, P_args),
    Z_block=(spin_inversion, 2, 0, Z_args),
)
# define particle conservation and op dicts
pcon_dict = dict(
    Np=Np,
    next_state=next_state,
    next_state_args=next_state_args,
    get_Ns_pcon=get_Ns_pcon,
    get_s0_pcon=get_s0_pcon,
    count_particles=count_particles,
    count_particles_args=count_particles_args,
    n_sectors=n_sectors,
)
op_dict = dict(op=op, op_args=op_args)
# create user basis
basis = user_basis(
    np.uint32,
    N,
    op_dict,
    allowed_ops=set("+-xyznI"),
    sps=2,
    pcon_dict=pcon_dict,
    **maps,
)
#
#
#
############   create same spin-1/2 basis_1d object   #############
basis_1d = spin_basis_1d(N, Nup=Np, pauli=True, kblock=0, pblock=1, zblock=1)  #
#
#
print(basis)
print(basis_1d)
#
############   create and compare Hamiltonians   #############
#
J = 1.0
spin_spin = [[J, j, (j + 1) % N] for j in range(N)]
static = [["xx", spin_spin], ["yy", spin_spin], ["zz", spin_spin]]
dynamic = []
#
no_checks = dict(check_symm=False, check_pcon=False, check_herm=False)
H = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)
H_1d = hamiltonian(static, [], basis=basis_1d, dtype=np.float64)
print(H.toarray())
print(H_1d.toarray())
print(np.linalg.norm((H - H_1d).toarray()))
