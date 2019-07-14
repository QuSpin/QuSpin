from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d # Hilbert space spin basis_1d
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import pre_check_state_sig_32,op_sig_32,map_sig_32 # user basis data types
from numba import carray,cfunc # numba helper functions
from numba import uint32,int32 # numba data types
import numpy as np
#
N = 10 # total number of lattice sites
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
######  function to filter states/project states out of the basis
#
@cfunc(pre_check_state_sig_32,
    locals=dict(s_shift_left=uint32,s_shift_right=uint32), )
def pre_check_state(s,N,args):
    """ imposes that that a bit with 1 must be preceded and followed by 0,
    i.e. a particle on a given site must have empty neighboring sites.
    #
    Works only for lattices of up to N=32 sites (otherwise, change mask)
    #
    """
    mask = (0xffffffff >> (32 - N)) # works for lattices of up to 32 sites
    # cycle bits left by 1 periodically
    s_shift_left = (((s << 1) & mask) | ((s >> (N - 1)) & mask))
    #
    # cycle bits right by 1 periodically
    s_shift_right = (((s >> 1) & mask) | ((s << (N - 1)) & mask))
    #
    return (((s_shift_right|s_shift_left)&s))==0
#
pre_check_state_args=None
#
######  construct user_basis 
# define maps dict
maps = dict() # no symmetries to apply.
# define op_dict
op_dict = dict(op=op,op_args=op_args)
# define pre_check_state
pre_check_state=(pre_check_state,pre_check_state_args) # None gives a null pinter to args
# create user basis
basis = user_basis(np.uint32,N,op_dict,allowed_ops=set("xyz"),sps=2,
                    pre_check_state=pre_check_state,Ns_block_est=300000,**maps)
# print basis
print(basis)
#
###### construct Hamiltonian
# site-coupling lists
h_list  = [[1.0,i] for i in range(N)]
# operator string lists
static = [["x",h_list],]
# compute Hamiltonian, no checks have been implemented
H = hamiltonian(static,[],basis=basis,dtype=np.float64)