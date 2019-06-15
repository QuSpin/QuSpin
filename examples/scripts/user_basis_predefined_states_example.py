from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
from quspin.basis.user import user_basis,next_state_sig_32,op_sig_32,map_sig_32
from quspin.basis import spin_basis_general
from scipy.special import comb
from numba import carray,cfunc
from numba import uint32,int32
import numpy as np


# function to call when generating next_state
@cfunc(next_state_sig_32)
def next_state(s,counter,N,basis):
    # return pre-calculated basis state.
    # add one because the first state is already checked.
    return basis[counter+1]

# costumized opstr function
@cfunc(op_sig_32,
    locals=dict(s=int32,b=uint32))
def op_func(op_struct_ptr,op_str,ind,N):
    # using struct pointer to pass op_structults 
    # back to C++ see numba Records
    op_struct = carray(op_struct_ptr,1)[0]
    err = 0
    ind = N - ind - 1 # convention for QuSpin for mapping from bits to sites.
    s = (((op_struct.state>>ind)&1)<<1)-1
    b = (1<<ind)

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

    return err

# wrapper to wrap up the data array as
# well as the functions required for the user_basis
# functionality. 
class function_wrapper(object):
    def __init__(self,basis):
        self.basis = basis

    # function to calculate the starting state to generate the 
    # particle conserving basis
    def get_s0_pcon(self,N,Np):
        # ignore input arguments as basis is already calculated.
        return self.basis[0]

    # calculate the size of the non-symmetry
    # reduced particle conservation basis
    def get_Ns_pcon(self,N,Np):
        # ignore input arguments as basis is already calculated.
        return self.basis.size

@cfunc(map_sig_32,locals=dict(mask=uint32,N_half=int32,
    s_left=uint32,s_right=uint32,s1=uint32,s2=uint32))
def translate(x,N,sign_ptr):
    # bit permutation target bits: 1 2 3 4 5 6 7 8 9 0 11 12 13 14 15 16 17 18 19 10
    # code generated here: http://programming.sirrida.de/calcperm.php
    # only works for N_half=10 
    return ((x & 0x0007fdff) << 1) | ((x & 0x00080200) >> 9)

@cfunc(map_sig_32)
def parity(x,N,sign_ptr):
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


def make_basis(N_half):
    old_basis = spin_basis_general(N_half,m=0)

    states = old_basis.states
    shift_states = np.left_shift(states,N_half)
    
    shape=states.shape+states.shape
    
    states_b = np.broadcast_to(states,shape)
    shift_states_b = np.broadcast_to(shift_states,shape)
    # this does the kronecker sum in a more memory efficient way. 
    return (states_b+shift_states_b.T).ravel()


N_half = 10
N = 2*N_half
Ns_full = 2**N
Np = () # dummy argument, could be any value

new_basis = make_basis(N_half)


next_state_args = new_basis
FW = function_wrapper(new_basis)

# define pcon_args dictionary
pcon_args = dict(Np=Np,next_state=next_state,next_state_args=next_state_args,
    get_Ns_pcon=FW.get_Ns_pcon,get_s0_pcon=FW.get_s0_pcon)

# symmetries to apply.
maps = dict(
    tr=(translate,N_half,0), # function, periodicity, number
    p=(parity,2,0)
    )
# maps = dict()
# construct user defined basis
basis = user_basis(np.uint32,N,Ns_full,op_func,allowed_ops=set("xyz"),sps=2,pcon_args=pcon_args,**maps)
# print basis
print(basis)
