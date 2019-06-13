from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
from quspin.basis.user import user_basis,check_state_nosymm_sig_32,op_sig_32,map_sig_32
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from numba import carray,cfunc
from numba import uint32,int32
import numpy as np
import sys

# costumized opstr function
@cfunc(op_sig_32,
    locals=dict(s=int32,b=uint32))
def op_func(resptr,op,ind,N):
    # using struct pointer to pass results 
    # back to C++ see numba Records
    res = carray(resptr,1)
    err = 0
    ind = N - ind - 1 # convention for QuSpin for mapping from bits to sites.
    s = (((res[0].state>>ind)&1)<<1)-1
    b = (1<<ind)

    if op==120: # "x" is integer value 120 (check with ord("x"))
        res[0].state ^= b
    elif op==121: # "y" is integer value 120 (check with ord("y"))
        res[0].state ^= b
        res[0].matrix_ele *= 1.0j*s
    elif op==122: # "z" is integer value 120 (check with ord("z"))
        res[0].matrix_ele *= s
    else:
        res[0].matrix_ele = 0
        err = -1

    return err

@cfunc(check_state_nosymm_sig_32,locals=dict(
    s_shift_left=uint32,s_shift_right=uint32))
def check_state_constraint(s,N,args):
    # require that a bit with 1 must be preceeded and followed by 0

    mask = (0xffffffff >> (32 - N))
    # cycle bits left by 1 periodically
    s_shift_left = (((s << 1) & mask) | ((s >> (N - 1)) & mask))

    # cycle bits right by 1 periodically
    s_shift_right = (((s >> 1) & mask) | ((s << (N - 1)) & mask))

    return (((s_shift_right|s_shift_left)&s))==0


# define full lattice size
N = 10
# full hilbert space is required
Ns_full = 2**N 

check_state_nosymm=(check_state_constraint,None) # None gives a null pinter to args
# no symmetries to apply.
maps = dict()
# construct user defined basis
basis = user_basis(np.uint32,N,Ns_full,op_func,allowed_ops=set("xyz"),sps=2,
    check_state_nosymm=check_state_nosymm,Ns_block_est=300000,**maps)
# print basis
print(basis)

