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
#                            example 15                                   #
#  In this script we demonstrate how to apply the user_basis to           #
#  construct a spin-1/2 model with sublattice particle conservation.      #
###########################################################################
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d # Hilbert space spin basis_1d
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_32,op_sig_32,map_sig_32 # user basis data types
from numba import carray,cfunc # numba helper functions
from numba import uint32,int32 # numba data types
import numpy as np
from scipy.special import comb
#
N_half = 4 # sublattice total number of sites
N = 2*N_half # total number of sites
Np = (N_half//2,N_half//2) #sublattice magnetizations
#
######  function to call when applying operators
@cfunc(op_sig_32,
    locals=dict(n=int32,b=uint32))
def op(op_struct_ptr,op_str,ind,N,args):
    # using struct pointer to pass op_struct_ptr back to C++ see numba Records
    op_struct = carray(op_struct_ptr,1)[0]
    err = 0
    ind = N - ind - 1 # convention for QuSpin for mapping from bits to sites.
    n = (op_struct.state>>ind)&1 # either 0 or 1
    b = (1 << ind)
    #
    if op_str==110: # "n" is integer value 110 (check with ord("n"))
        op_struct.matrix_ele *= n
    elif op_str==43: # "+" is integer value 43 (check with ord("+"))
        if n: op_struct.matrix_ele = 0
        else: op_struct.state ^= b # create hcb
    elif op_str==45: # "-" is integer value 45 (check with ord("-"))
        if n: op_struct.state ^= b # destroy hcb
        else: op_struct.matrix_ele = 0
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
@cfunc(next_state_sig_32, locals=dict(N_half=int32,t=uint32,s_right=uint32,s_left=uint32))
def next_state(s,counter,N,args):
    # unpack args
    mask = args[0]
    s_right_min = args[1]
    s_right_max = args[2]
    N_half = args[3] # = (N>>1), sublattice system size
    #
    # split sublattice
    s_left = s >> N_half
    s_right = s & mask
    # increment s_right unless it has reached the last state,
    if s_right < s_right_max:
        if s_right > 0:
            t = (s_right | (s_right - 1)) + 1
            s_right = t | ((((t & (0-t)) // (s_right & (0-s_right))) >> 1) - 1) 
    # otherwise op_structet s_right to first state and increment s_left.
    else:
        s_right = s_right_min
        if s_left > 0:
            t = (s_left | (s_left - 1)) + 1
            s_left = t | ((((t & (0-t)) // (s_left & (0-s_left))) >> 1) - 1)
    # combine and return next state.
    return (s_left << N_half) + s_right
#
### optional arguments to pass into next_state
s_right_min = sum(1<<i for i in range(Np[1])) # fill first bits
s_right_max = sum(1<<(N_half-i-1) for i in range(Np[1])) # fill last bits
mask = 2**N_half - 1 # fill all bits 
next_state_args = np.array([mask,s_right_min,s_right_max,N >> 1],dtype=np.uint32)
#
# python function to calculate the starting state to generate the particle conserving basis
def get_s0_pcon(N,Np):
    """ calculates the starting state to generate the particle conserving basis. """
    N_half = N>>1
    Np_left,Np_right = Np

    s_left  = sum(1<<i for i in range(Np_left ))
    s_right = sum(1<<i for i in range(Np_right))
    return (s_left << N_half) + s_right
# 
# python function to calculate the size of the particle-conserved basis, 
# i.e. BEFORE applying pre_check_state and symmetry maps
def get_Ns_pcon(N,Np):
    """ calculates the size of the particle conservation basis (ignoring symmetries at this stage). """
    N_half = (N>>1)
    return comb(N_half,Np[0],exact=True)*comb(N_half,Np[1],exact=True)
#
######  construct user_basis 
# define maps dict
maps = dict() # no symmetries
# define particle conservation and op dicts
pcon_dict = dict(Np=Np,next_state=next_state,next_state_args=next_state_args,
                get_Ns_pcon=get_Ns_pcon,get_s0_pcon=get_s0_pcon)
op_dict = dict(op=op,op_args=op_args)
# create user basis
basis = user_basis(np.uint32,N,op_dict,allowed_ops=set("n+-"),sps=2,pcon_dict=pcon_dict,**maps)
# print basis
print(basis)
#
###### construct Hamiltonian
# site-coupling lists
t_list  = [[1.0,i,(i+1)%N_half] for i in range(N_half)] # first sublattice/leg of the ladder
t_list += [[t,N_half+i,N_half+j] for t,i,j in t_list] # second sublattice/leg of the ladder
U_list = [[1.0,i,i+N_half] for i in range(N_half)]
# operator string lists
static = [["+-",t_list],["-+",t_list],["nn",U_list]]
# compute Hamiltonian, no checks have been implemented
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H = hamiltonian(static,[],basis=basis,dtype=np.float64,**no_checks)