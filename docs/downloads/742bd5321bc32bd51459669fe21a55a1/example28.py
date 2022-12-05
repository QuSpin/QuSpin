from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='4' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#########################################################################
#                             example 28                                #
#  In this script we demonstrate how to use QuSpin's user_basis         #
#  to define symmetries, which are not supported by the basis_general   #
#  classes. We take an 8-site honeycomb lattice with PBC and define     #
#  the Kitaev model in the spectral sectors given by the Wilson loop /  #
#  plaquette operators W.                                               #
#########################################################################
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_32,op_sig_32,map_sig_32,count_particles_sig_32 # user basis data types signatures
from numba import carray,cfunc,jit # numba helper functions
from numba import uint32,int32 # numba data types
import numpy as np
#
N=8 # lattice sites
assert(N==8) # code below works only for N=8!
#
############   create spin-1/2 user basis object   #############
#
######  function to call when applying operators
@cfunc(op_sig_32,
	locals=dict(s=int32,n=int32,b=uint32), )
def op(op_struct_ptr,op_str,site_ind,N,args):
	# using struct pointer to pass op_struct_ptr back to C++ see numba Records
	op_struct = carray(op_struct_ptr,1)[0]
	err = 0
	#
	site_ind = N - site_ind - 1 # convention for QuSpin for mapping from bits to sites.
	n = (op_struct.state>>site_ind)&1 # either 0 or 1
	s = (((op_struct.state>>site_ind)&1)<<1)-1 # either -1 or 1
	b = (1<<site_ind)
	#
	if op_str==120: # "x" is integer value 120 = ord("x")
		op_struct.state ^= b

	elif op_str==121: # "y" is integer value 120 = ord("y")
		op_struct.state ^= b
		op_struct.matrix_ele *= 1.0j*s

	elif op_str==43: # "+" is integer value 43 = ord("+")
		if n: op_struct.matrix_ele = 0
		else: op_struct.state ^= b # create spin

	elif op_str==45: # "-" is integer value 45 = ord("-")
		if n: op_struct.state ^= b # destroy spin
		else: op_struct.matrix_ele = 0

	elif op_str==122: # "z" is integer value 120 = ord("z")
		op_struct.matrix_ele *= s

	elif op_str==110: # "n" is integer value 110 = ord("n")
		op_struct.matrix_ele *= n

	elif op_str==73: # "I" is integer value 73 = ord("I")
		pass

	else:
		op_struct.matrix_ele = 0
		err = -1
	#
	return err
op_args=np.array([],dtype=np.uint32)
#
######  define symmetry maps
#
@jit(uint32(uint32,uint32,int32),)
def _compute_occupation(state,site_ind,N):
	# auxiliary function to check occupation of state at site_ind
	# CAUTION: 32-bit integers code only!
	site_ind = N - site_ind - 1 # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
	return (state>>site_ind)&1 # occupation: either 0 or 1
#
@jit(uint32(uint32,uint32,int32),locals=dict(b=uint32,))
def _flip_occupation(state,site_ind,N):
	# auxiliary function to flip occupation of state at site_ind
	# CAUTION: 32-bit integers code only!
	site_ind = N - site_ind - 1     # compute lattice site in reversed bit configuration (cf QuSpin convention for mapping from bits to sites)
	b = 1; b <<= site_ind   	    # compute a "mask" integer b which is 1 on site site_ind and zero elsewhere      			
	return state^b # flip occupation on site site_ind
#
@cfunc(map_sig_32,
	locals=dict(out=uint32,) )
def W_symm(state,N,sign_ptr,args):
	""" 
	applies plaquette operator W = "xyzxyz" on sites defined in args. 
	# CAUTION: 32-bit integers code only!

	- the two 'z'-operators in W are easiest to apply, since just return the occupation number in the spin basis.
	- the two 'x'-operators in W are alo easy -- they only flip the occupation
	- the two 'y'-operators in W do both: they flip the occupation and mutiply the state by +/-i (each) 
	"""
	assert(N==8) # works only for N=8!
	out = 0
	#
	# compute sign by applying operators from W on state
	sign_ptr[0] *= -1 + 2*_compute_occupation(state,args[1],N) # 'y'; factor i taken into account below
	sign_ptr[0] *= -1 + 2*_compute_occupation(state,args[2],N) # 'z'
	sign_ptr[0] *= -1 + 2*_compute_occupation(state,args[4],N) # 'y'; factor i taken into account below
	sign_ptr[0] *= -1 + 2*_compute_occupation(state,args[5],N) # 'z'
	sign_ptr[0] *= -1 # -1=i**2, yy 
	#
	# flip occupation of state on sites 0,1, 3,4
	out = _flip_occupation(state,args[0],N) # 'x'
	out = _flip_occupation(out,  args[1],N) # 'y'
	out = _flip_occupation(out,  args[3],N) # 'x'
	out = _flip_occupation(out,  args[4],N) # 'y'
	#
	return out
# the argument list stores the sites for the four W operators, defined clockwise
W_args_1=np.array([0,1,2,3,4,5],dtype=np.uint32) # plaquette 1
W_args_2=np.array([4,3,6,1,0,7],dtype=np.uint32) # plaquette 2
W_args_3=np.array([6,7,0,5,2,1],dtype=np.uint32) # plaquette 3
W_args_4=np.array([2,5,4,7,6,3],dtype=np.uint32) # plaquette 4
#
######  construct user_basis 
#
# define maps dict
q_nums=[0,0,0,0] # not all sectors contain a finite-number of states
maps = dict( W1_block=(W_symm,2,q_nums[0],W_args_1), 
			 W2_block=(W_symm,2,q_nums[1],W_args_2), 
			 W3_block=(W_symm,2,q_nums[2],W_args_3), 
			 W4_block=(W_symm,2,q_nums[3],W_args_4),
			) 
op_dict = dict(op=op,op_args=op_args)
# create user basis
basis = user_basis(np.uint32,N,op_dict,allowed_ops=set("+-xyznI"),sps=2,**maps)
#print(basis)
#
##### define Kitaev Hamiltonian
#
### coupling strengths
Jx,Jy,Jz=1.0,1.0,1.0 
#
### site-coupling lists
zz_int = [[Jz,0,1], [Jz,4,3], [Jz,2,5], [Jz,6,7] ] 
yy_int = [[Jy,2,3], [Jy,0,5], [Jy,6,1], [Jy,4,7] ]
xx_int = [[Jx,2,1], [Jx,4,5], [Jx,6,3], [Jx,0,7] ] 
#
### construct Hamitonian
no_checks=dict(check_pcon=False, check_herm=False, check_symm=False)
#
Hzz = hamiltonian([['zz',zz_int],], [], basis=basis, dtype=np.complex128, **no_checks) 
Hyy = hamiltonian([['yy',yy_int],], [], basis=basis, dtype=np.complex128, **no_checks) 
Hxx = hamiltonian([['xx',xx_int],], [], basis=basis, dtype=np.complex128, **no_checks) 
#
H_Kitaev = Hzz + Hyy  + Hxx
#
### diagonalize H_Kitaev
#
#E, V = H_Kitaev.eigsh(k=4, which='SA')
E, V = H_Kitaev.eigh()
print('energy spectrum:', E[:4])
#
### construct plaquette operator and check its e'state expectation
#
W_int = [[1.0, *list(W_args_1) ]]
W = hamiltonian([['xyzxyz',W_int],], [], basis=basis, dtype=np.float64, **no_checks) 
print('check expectation of W-operator:', W.expt_value(V[:,:4],).real)
