from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
################################################################################
#                              example 24                                      #
# This example shows how to use the `user_basis` to define Majorana operators. #
################################################################################
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d # Hilbert space fermion basis_1d
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_32,op_sig_32,map_sig_32,count_particles_sig_32 # user basis data types signatures
from numba import carray,cfunc,jit # numba helper functions
from numba import uint32,int32 # numba data types
import numpy as np
from scipy.special import comb
np.set_printoptions(suppress='True', precision=6)
#
N=6 # lattice sites
#
############   create soinless fermion user basis object   #############
#
@jit(uint32(uint32,uint32),locals=dict(f_count=uint32,),nopython=True,nogil=True)
def _count_particles_32(state,site_ind):
	# auxiliary function to count number of fermions, i.e. 1's in bit configuration of the state, up to site site_ind
	# CAUTION: 32-bit integers code only!
	f_count = state & ((0x7FFFFFFF) >> (31 - site_ind));
	f_count = f_count - ((f_count >> 1) & 0x55555555);
	f_count = (f_count & 0x33333333) + ((f_count >> 2) & 0x33333333);
	return (((f_count + (f_count >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24
#
@cfunc(op_sig_32,
	locals=dict(s=int32,sign=int32,n=int32,b=uint32,f_count=uint32), )
def op(op_struct_ptr,op_str,site_ind,N,args):
	# using struct pointer to pass op_struct_ptr back to C++ see numba Records
	op_struct = carray(op_struct_ptr,1)[0]
	err = 0
	#
	site_ind = N - site_ind - 1 # convention for QuSpin for mapping from bits to sites.
	#####
	f_count = _count_particles_32(op_struct.state,site_ind)
	#####
	sign = -1 if f_count&1 else 1
	n = (op_struct.state>>site_ind)&1 # either 0 or 1
	b = (1<<site_ind)
	#
	if op_str==120: # "x" is integer value 120 = ord("x")
		op_struct.state ^= b
		op_struct.matrix_ele *= sign

	elif op_str==121: # "y" is integer value 120 = ord("y")
		op_struct.state ^= b
		op_struct.matrix_ele *= -1.0j*sign*((n<<1)-1)
	
	elif op_str==43: # "+" is integer value 43 = ord("+")
		op_struct.matrix_ele *= (0.0 if n else sign)
		op_struct.state ^= b # create fermion

	elif op_str==45: # "-" is integer value 45 = ord("-")
		op_struct.matrix_ele *= (sign if n else 0.0)
		op_struct.state ^= b # create fermion
		
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
@cfunc(map_sig_32,
	locals=dict(shift=uint32,xmax=uint32,x1=uint32,x2=uint32,period=int32,l=int32,f_count1=int32,f_count2=int32) )
def translation(x,N,sign_ptr,args):
	""" works for all system sizes N. """
	shift = args[0] # translate state by shift sites
	period = N # periodicity/cyclicity of translation
	xmax = args[1]
	#
	l = (shift+period)%period
	x1 = (x >> (period - l))
	x2 = ((x << l) & xmax)
	#
	#####
	# count number of fermions, i.e. 1's in bit configuration of x1
	f_count1 = _count_particles_32(x1,period)
	# count number of fermions, i.e. 1's in bit configuration of x2
	f_count2 = _count_particles_32(x2,period)
	#####
	# compute fermion sign
	sign_ptr[0] *= (-1 if ((f_count1&1)&(f_count2&1)&1) else 1)
	#
	return (x2 | x1)
T_args=np.array([1,(1<<N)-1],dtype=np.uint32)
#
@cfunc(map_sig_32,
	locals=dict(out=uint32,s=uint32,f_count=int32) )
def parity(x,N,sign_ptr,args):
	""" works for all system sizes N. """
	out = 0
	s = args[0]
	#
	#####
	# count number of fermions, i.e. 1's in bit configuration of the state
	f_count = _count_particles_32(x,N)
	#####
	sign_ptr[0] *= (-1 if ((f_count&2) and 1) else 1)
	#
	out ^= (x&1)
	x >>= 1
	while(x):
		out <<= 1
		out ^= (x&1)
		x >>= 1
		s -= 1
	#
	out <<= s
	return out
P_args=np.array([N-1],dtype=np.uint32)
#
######  construct user_basis
# define anti-commuting bits -- fermion signs on the integer bits (not sites!) that represent a fermion degree of freedom
noncommuting_bits = [(np.arange(N),-1)] # fermion signs are counted w.r.t. the shift operator << 
# define maps dict
maps = dict(T_block=(translation,N,0,T_args), P_block=(parity,2,0,P_args), ) 
#maps = dict(P_block=(parity,2,0,P_args), )
#maps = dict(T_block=(translation,N,0,T_args) ) 
op_dict = dict(op=op,op_args=op_args)
# create user basiss
basis = user_basis(np.uint32,N,op_dict,allowed_ops=set("xy+-nI"),sps=2,noncommuting_bits=noncommuting_bits,**maps)
#
#
print(basis)
#
############   create and compare Hamiltonians   #############
#
##### Hamiltonian in using Majoranas 
#
J=-np.sqrt(2.0) # hoppping
U=+1.0 # nn interaction
#
hop_term_p=[[+0.5j*J,j,(j+1)%N] for j in range(N)]
hop_term_m=[[-0.5j*J,j,(j+1)%N] for j in range(N)]
density_term=[[+0.5j*U,j,j] for j in range(N)]
int_term=[[-0.25*U,j,j,(j+1)%N,(j+1)%N] for j in range(N)]
id_term=[[0.25*U,j] for j in range(N)]
#
static=[['xy',hop_term_p],['yx',hop_term_m], 					# kinetic energy
		['I',id_term],['xy',density_term],['xyxy',int_term],	# nn interaction energy
		]
dynamic=[]
#
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H_majorana=hamiltonian(static,[],basis=basis,dtype=np.float64,**no_checks)
#
#
##### Hamiltonian using complex fermions
#
#
hopping_pm=[[+J,j,(j+1)%N] for j in range(N)]
hopping_mp=[[-J,j,(j+1)%N] for j in range(N)]
nn_int=[[U,j,(j+1)%N] for j in range(N)]
#
static=[["+-",hopping_pm],["-+",hopping_mp],["nn",nn_int]]
dynamic=[]
#
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H=hamiltonian(static,[],basis=basis,dtype=np.float64,**no_checks)
print(H.toarray())
print()
print(H_majorana.toarray())
print()
print(np.linalg.norm((H-H_majorana).toarray()))