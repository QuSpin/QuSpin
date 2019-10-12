from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
###########################################################################
#                            example 16                                   #
#  In this script we demonstrate how to apply the user_basis to reduce    #
#  user-imported arrays of bases states (in integer representation)       #
#  to user-defined symmetry-reduced subspaces.                            #
###########################################################################
from quspin.basis import spin_basis_1d,spin_basis_general # Hilbert space spin basis_1d
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_32,op_sig_32,map_sig_32 # user basis data types
from numba import carray,cfunc # numba helper functions
from numba import uint32,int32 # numba data types
import numpy as np
from scipy.special import comb
#
#####
N_half = 10 # number of sites for each leg of the ladder
N = 2*N_half # total number of lattice sites
#
def make_basis(N_half):
    """ Generates a list of integers to represent external, user-imported basis """
    old_basis = spin_basis_general(N_half,m=0)
    #
    states = old_basis.states
    shift_states = np.left_shift(states,N_half)
    #
    shape=states.shape+states.shape
    #
    states_b = np.broadcast_to(states,shape)
    shift_states_b = np.broadcast_to(shift_states,shape)
    # this does the kronecker sum in a more memory efficient way. 
    return (states_b+shift_states_b.T).ravel()
#
external_basis = make_basis(N_half)
#
Np = () # dummy argument, could be any value (particle conservation should've been 
# taken into account when constructing the basis object)
#
######  function to call when applying operators
@cfunc(op_sig_32, locals=dict(s=int32,b=uint32))
def op(op_struct_ptr,op_str,ind,N,args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr,1)[0]
    err = 0
    ind = N - ind - 1 # convention for QuSpin for mapping from bits to sites.
    s = (((op_struct.state>>ind)&1)<<1)-1
    b = (1<<ind)
    #
    if op_str==120: # "x" is integer value 120 (check with ord("x"))
        op_struct.state ^= b
    elif op_str==121: # "y" is integer value 120 (check with ord("y"))
        op_struct.state ^= b
        op_struct.matrix_ele *= 1.0j*s
    elif op_str==122: # "z" is integer value 120 (check with ord("z"))
        op_struct.matrix_ele *= s
    else:
        op_struct.matrix_ele = 0
        err = -1
    #
    return err
#
op_args=np.array([],dtype=np.uint32)
#
######  function to read user-imported basis into QuSpin 
#
# function to call when generating next_state
@cfunc(next_state_sig_32)
def next_state(s,counter,N,args):
    # return pre-calculated basis state.
    # add one to counter because the first state is already checked.
    return args[counter+1] # = basis
#
next_state_args = external_basis # this has to be an array of same dtype as the user_basis
#
class function_wrapper(object):
    """
    This class provides a wrapper for the user-imported basis,
    as well as the functions required for the `user_basis` functionality.
    #
    This is needed to easily pass parameters (defined as class attributes) to the
    functions `get_so_pcon()` and `get_Ns_pcon`.
    """
    def __init__(self,basis):
        self.basis = basis
    #
    # python function to calculate the starting state to generate the particle conserving basis
    def get_s0_pcon(self,N,Np):
        """ calculates the starting state to generate the particle conserving basis. """
        # ignore input arguments as basis is already calculated.
        return self.basis[0]
    # 
    # python function to calculate the size of the particle-conserved basis, 
    # i.e. BEFORE applying pre_check_state and symmetry maps
    def get_Ns_pcon(self,N,Np):
        """ calculates the size of the particle conservation basis (ignoring symmetries at this stage). """
        # ignore input arguments as basis is already calculated.
        return self.basis.size
#
######  define symmetry maps
#
if N_half!=10:
    print ("symmetry masks are hard-coded and work only for N=10; \n\
To do a different system size, it is required to update the masks accordingly.\n\
exiting...")
    exit()
#
@cfunc(map_sig_32)
def translation(x,N,sign_ptr,args):
    # bit permutation target bits: 1 2 3 4 5 6 7 8 9 0 11 12 13 14 15 16 17 18 19 10
    # code generated here: http://programming.sirrida.de/calcperm.php
    # only works for N_half=10 
    return ((x & 0x0007fdff) << 1) | ((x & 0x00080200) >> 9)
T_args=np.array([],dtype=np.uint32)
#
@cfunc(map_sig_32)
def parity(x,N,sign_ptr,args):
    # bit permutation target bits: 9 8 7 6 5 4 3 2 1 0 19 18 17 16 15 14 13 12 11 10
    # code generated here: http://programming.sirrida.de/calcperm.php
    # only works for N_half=10
    return  (   ((x & 0x00004010) << 1)
              | ((x & 0x00002008) << 3)
              | ((x & 0x00001004) << 5)
              | ((x & 0x00000802) << 7)
              | ((x & 0x00000401) << 9)
              | ((x & 0x00080200) >> 9)
              | ((x & 0x00040100) >> 7)
              | ((x & 0x00020080) >> 5)
              | ((x & 0x00010040) >> 3)
              | ((x & 0x00008020) >> 1))
P_args=np.array([],dtype=np.uint32)
#
######  construct user_basis 
# define maps dict
maps = dict(T_block=(translation,N_half,0,T_args), P_block=(parity,2,0,P_args), )
# define particle conservation and op dicts
FW = function_wrapper(external_basis)
pcon_dict = dict(Np=Np,next_state=next_state,next_state_args=next_state_args,
                 get_Ns_pcon=FW.get_Ns_pcon,get_s0_pcon=FW.get_s0_pcon)
op_dict = dict(op=op,op_args=op_args)
# create user basis
basis = user_basis(np.uint32,N,op_dict,allowed_ops=set("xyz"),sps=2,pcon_dict=pcon_dict,**maps)
# print basis
print(basis)