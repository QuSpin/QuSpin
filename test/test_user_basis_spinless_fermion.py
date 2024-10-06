from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d  # Hilbert space spin basis_1d
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
############   create soinless fermion user basis object   #############
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
    # count number of fermions, i.e. 1's in bit configuration of the state, up to site sit_ind
    # CAUTION: 32-bit integers code only!
    f_count = op_struct.state & ((0x7FFFFFFF) >> (31 - site_ind))
    f_count = f_count - ((f_count >> 1) & 0x55555555)
    f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333)
    f_count = (((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
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


#
######  function to implement magnetization/particle conservation
#
@cfunc(
    next_state_sig_32,
    locals=dict(t=uint32),
)
def next_state(s, counter, N, args):
    """implements particle number conservation."""
    if s == 0:
        return s
    #
    t = (s | (s - 1)) + 1
    return t | ((((t & (0 - t)) // (s & (0 - s))) >> 1) - 1)




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
        f_count1=int32,
        f_count2=int32,
    ),
)
def translation(x, N, sign_ptr, args):
    """works for all system sizes N."""
    shift = args[0]  # translate state by shift sites
    period = N  # periodicity/cyclicity of translation
    xmax = args[1]
    #
    l_full = (shift + period) % period
    x1 = x >> (period - l_full)
    x2 = (x << l_full) & xmax
    #
    #####
    # count number of fermions, i.e. 1's in bit configuration of x1
    # CAUTION: 32-bit integers code only!
    f_count1 = x1 & ((0x7FFFFFFF) >> (31 - period))
    f_count1 = f_count1 - ((f_count1 >> 1) & 0x55555555)
    f_count1 = (f_count1 & 0x33333333) + ((f_count1 >> 2) & 0x33333333)
    f_count1 = (((f_count1 + (f_count1 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    #####
    # count number of fermions, i.e. 1's in bit configuration of the state, up to site sit_ind
    # CAUTION: 32-bit integers code only!
    f_count2 = x2 & ((0x7FFFFFFF) >> (31 - period))
    f_count2 = f_count2 - ((f_count2 >> 1) & 0x55555555)
    f_count2 = (f_count2 & 0x33333333) + ((f_count2 >> 2) & 0x33333333)
    f_count2 = (((f_count2 + (f_count2 >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    #####
    # compute fermion sign
    sign_ptr[0] *= -1 if ((f_count1 & 1) & (f_count2 & 1) & 1) else 1
    #
    return x2 | x1


#
@cfunc(map_sig_32, locals=dict(out=uint32, s=uint32, f_count=int32))
def parity(x, N, sign_ptr, args):
    """works for all system sizes N."""
    out = 0
    s = args[0]
    #
    #####
    # count number of fermions, i.e. 1's in bit configuration of the state
    # CAUTION: 32-bit integers code only!
    f_count = x & ((0x7FFFFFFF) >> (31 - N))
    f_count = f_count - ((f_count >> 1) & 0x55555555)
    f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333)
    f_count = (((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    #####
    sign_ptr[0] *= -1 if (f_count & 2) & 1 else 1
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


#
######  define function to count particles in bit representation
#
@cfunc(count_particles_sig_32, locals=dict(f_count=uint32))
def count_particles(x, p_count_ptr, args):
    """Counts number of particles/spin-ups in a state stored in integer representation for up to N=32 sites"""
    #
    f_count = x & ((0x7FFFFFFF) >> (31 - args[0]))
    f_count = f_count - ((f_count >> 1) & 0x55555555)
    f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333)
    f_count = (((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
    #
    p_count_ptr[0] = f_count

def test():
    N = 8  # lattice sites
    Np = N // 2  # total number of fermions
    op_args = np.array([], dtype=np.uint32)
    next_state_args = np.array([], dtype=np.uint32)  # compulsory, even if empty
    T_args = np.array([1, (1 << N) - 1], dtype=np.uint32)
    P_args = np.array([N - 1], dtype=np.uint32)
    n_sectors = 1  # number of particle sectors
    count_particles_args = np.array([N], dtype=np.int32)
    #
    ######  construct user_basis
    # define maps dict
    maps = dict(
        T_block=(translation, N, 0, T_args),
        P_block=(parity, 2, 0, P_args),
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
    # create user basiss
    basis = user_basis(
        np.uint32, N, op_dict, allowed_ops=set("+-nI"), sps=2, pcon_dict=pcon_dict, **maps
    )
    #
    #
    #
    ############   create same spinless fermion basis_1d object   #############
    basis_1d = spinless_fermion_basis_1d(N, Nf=Np, kblock=0, pblock=1)  #
    #
    #
    print(basis)
    print(basis_1d)
    np.testing.assert_allclose(
        basis.states - basis_1d.states, 0.0, atol=1e-5, err_msg="Failed bases comparison!"
    )
    #
    ############   create Hamiltonians   #############
    #
    J = -1.0
    U = +1.0
    #
    hopping_pm = [[+J, j, (j + 1) % N] for j in range(N)]
    hopping_mp = [[-J, j, (j + 1) % N] for j in range(N)]
    nn_int = [[U, j, (j + 1) % N] for j in range(N)]
    #
    static = [["+-", hopping_pm], ["-+", hopping_mp], ["nn", nn_int]]
    #
    no_checks = dict(check_symm=False, check_herm=False, check_pcon=False)
    H = hamiltonian(static, [], basis=basis, dtype=np.float64, **no_checks)
    H_1d = hamiltonian(static, [], basis=basis_1d, dtype=np.float64, **no_checks)

    np.testing.assert_allclose(
        (H - H_1d).toarray(), 0.0, atol=1e-5, err_msg="Failed Hamiltonians comparison!"
    )
