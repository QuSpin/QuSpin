from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
from quspin.basis.user import user_basis,next_state_sig_32,op_sig_32,map_sig_32
from quspin.operators import hamiltonian
from scipy.special import comb
from numba import carray,cfunc
from numba import uint32,int32
import numpy as np

# function to call when generating next_state
@cfunc(next_state_sig_32,
    locals=dict(N_half=int32,t=uint32,
        s_right=uint32,s_left=uint32))
def next_state(s,counter,N,args):
    # unpack args
    mask = args[0]
    s_right_min = args[1]
    s_right_max = args[2]

    # get sublattice system size.
    N_half = N >> 1

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

# costumized opstr function
@cfunc(op_sig_32,
    locals=dict(n=int32,b=uint32))
def op_func(op_struct_ptr,op_str,ind,N):
    # using struct pointer to pass op_structults 
    # back to C++ see numba Records
    op_struct = carray(op_struct_ptr,1)[0]
    err = 0
    ind = N - ind - 1 # convention for QuSpin for mapping from bits to sites.
    n = (op_struct.state>>ind)&1 # either 0 or 1
    b = 1
    b = b << ind

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

    return err


# function to calculate the starting state to generate the 
# particle conserving basis
def get_s0_pcon(N,Np):
    N_half = N >> 1
    Np_left,Np_right = Np

    s_left  = sum(1<<i for i in range(Np_left ))
    s_right = sum(1<<i for i in range(Np_right))
    return (s_left << N_half) + s_right

# calculate the size of the non-symmetry
# reduced particle conservation basis
def get_Ns_pcon(N,Np):
    N_half = N >> 1
    return comb(N_half,Np[0],exact=True)*comb(N_half,Np[1],exact=True)


# define sublattice system size
N_half = 4
# define full lattice size
N = 2*N_half 
# define sublattice magnetizations
Np = (N_half//2,N_half//2) 
# full hilbert space is required
Ns_full = 2**N 

# optional arguments to pass into next_state
# fill first bits
s_right_min = sum(1<<i for i in range(Np[1])) 
# fill last bits
s_right_max = sum(1<<(N_half-i-1) for i in range(Np[1])) 
# fill all bits 
mask = 2**N_half - 1
next_state_args = np.array([mask,s_right_min,s_right_max],dtype=np.uint32)
# define pcon_args dictionary
pcon_args = dict(Np=Np,next_state=next_state,next_state_args=next_state_args,
    get_Ns_pcon=get_Ns_pcon,get_s0_pcon=get_s0_pcon)

# no symmetries to apply.
maps = dict()
# construct user defined basis
basis = user_basis(np.uint32,N,Ns_full,op_func,allowed_ops=set("n+-"),sps=2,pcon_args=pcon_args,**maps)
print(basis)

t_list  = [[1.0,i,(i+1)%N_half] for i in range(N_half)]
t_list += [[t,N_half+i,N_half+j] for t,i,j in t_list]
U_list = [[1.0,i,i+N_half] for i in range(N_half)]

static = [["+-",t_list],["-+",t_list],["nn",U_list]]

H = hamiltonian(static,[],basis=basis,dtype=np.float64)


