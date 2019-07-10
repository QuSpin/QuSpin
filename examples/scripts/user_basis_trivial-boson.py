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
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis_1d
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_32,op_sig_32,map_sig_32,count_particles_sig_32 # user basis data types signatures
from numba import carray,cfunc # numba helper functions
from numba import uint32,int32,float64 # numba data types
import numpy as np
from scipy.special import comb
#
N=6 # lattice sites
sps=3 # states per site
Np=N//2 # total number of bosons
#
############   create boson user basis object   #############
#
######  function to call when applying operators
@cfunc(op_sig_32,
	locals=dict(s=int32,n=int32,b=uint32,occ=int32,sps=uint32,me_offdiag=float64,me_diag=float64), )
def op(op_struct_ptr,op_str,site_ind,N,args):
	# using struct pointer to pass op_struct_ptr back to C++ see numba Records
	op_struct = carray(op_struct_ptr,1)[0]
	err = 0
	sps=3 #args[0]
	me_offdiag=1.0;
	me_diag=1.0;
	#
	site_ind = N - site_ind - 1 # convention for QuSpin for mapping from bits to sites.
	occ = (op_struct.state//sps**site_ind)%sps # occupation
	n = (op_struct.state>>site_ind)&1 # either 0 or 1
	s = (((op_struct.state>>site_ind)&1)<<1)-1 # either -1 or 1
	b = sps**site_ind
	#
	if op_str==43: # "+" is integer value 43 = ord("+")
		me_offdiag *= (occ+1)%sps
		op_struct.state += (b if (occ+1)<sps else 0) 

	elif op_str==45: # "-" is integer value 45 = ord("-")
		me_offdiag *= occ;
		op_struct.state -= (b if occ>0 else 0)

	elif op_str==110: # "n" is integer value 110 = ord("n")
		me_diag *= occ

	elif op_str==73: # "I" is integer value 73 = ord("I")
		pass

	else:
		me_diag = 0.0
		err = -1
	#
	op_struct.matrix_ele *= me_diag*np.sqrt(me_offdiag)
	#
	return err
#
op_args=np.array([sps],dtype=np.uint32)
######  function to implement magnetization/particle conservation
#
@cfunc(next_state_sig_32,
	locals=dict(t=uint32,i=int32,j=int32,n=int32,sps=uint32,b1=int32,b2=int32,l=int32,n_left=int32), )
def next_state(s,counter,N,args):
	""" implements particle number conservation. Particle number set by initial state, cf `get_s0_pcon()` below. """
	t = s;
	sps=args[1]
	n=0 # running total of number of particles
	for i in range(N): # loop over lattices sites
		b1 = (t//args[i])%sps # get occupation at site i
		if b1>0: # if there is a boson
			n += b1 
			b2 = (t/args[i+1])%sps # get occupation st site ahead
			if b2<(sps-1): # if I can move a boson to this site
				n -= 1 # decrease one from the running total
				t -= args[i] # remove one boson from site i 
				t += args[i+1] # add one boson to site i+1
				if n>0: # if any bosons left
					# so far: moved one boson forward; 
					# now: take rest of bosons and fill first l sites with maximum occupation 
					# to keep lexigraphic order
					l = n//(sps-1) # how many sites can be fully occupied with n bosons
					n_left = n%(sps-1) # leftover of particles on not maximally occupied sites
					for j in range(i+1):
						t -= (t//args[j])%sps * args[j];
						if j<l: # fill in with maximal occupation
							t += (sps-1)*args[j]
						elif j==l: # fill with leftover
							t += n_left*args[j]
				break # stop loop
	return t
next_state_args=np.array([sps**i for i in range(N)],dtype=np.uint32)
# python function to calculate the starting state to generate the particle conserving basis
def get_s0_pcon(N,Np):
	sps = 3
	l = Np//(sps-1)
	s  = sum((sps-1) * sps**i for i in range(l))
	s += (Np%(sps-1)) * sps**l
	return s
# python function to calculate the size of the particle-conserved basis, i.e. 
# BEFORE applying pre_check_state and symmetry maps
def get_Ns_pcon(N,Np):
	Ns=0
	sps=3
	for r in range(Np//sps+1):
		r_2 = Np - r*sps
		if r % 2 == 0:
			Ns +=  comb(N,r,exact=True) * comb(N + r_2 - 1,r_2,exact=True)
		else:
			Ns += -comb(N,r,exact=True) * comb(N + r_2 - 1,r_2,exact=True)

	return Ns
#
######  define symmetry maps
#
@cfunc(map_sig_32,
	locals=dict(shift=uint32,out=uint32,sps=uint32,i=int32,j=int32,) )
def translation(x,N,sign_ptr,args):
	""" works for all system sizes N. """
	out = 0
	shift = args[0]
	sps = args[1]
	for i in range(N):
		j = (i+shift+N)%N
		out += ( x%sps ) * sps**j
		x //= sps
	#
	return out
T_args=np.array([1,sps],dtype=np.uint32)
#
@cfunc(map_sig_32,
	locals=dict(out=uint32,sps=uint32,i=int32,j=int32) )
def parity(x,N,sign_ptr,args):
	""" works for all system sizes N. """
	out = 0
	sps = args[0]
	for i in range(N):
		j = (N-1) - i
		out += ( x%sps ) * (sps**j)
		x //= sps
	#
	return out
P_args=np.array([sps],dtype=np.uint32)
#
######  define function to count particles in bit representation
#
@cfunc(count_particles_sig_32,
	locals=dict(s=uint32,))
def count_particles(x,p_count_ptr,args):
	""" Counts number of particles/spin-ups in a state stored in integer representation for up to N=32 sites """
	#
	s = x # integer x cannot be changed
	for i in range(args[0]):
		p_count_ptr[0] += s%args[1]
		s /= args[1]
n_sectors=1 # number of particle sectors
count_particles_args=np.array([N,sps],dtype=np.int32)
#
######  construct user_basis 
# define maps dict
maps = dict(T_block=(translation,N,0,T_args),P_block=(parity,2,0,P_args), ) 
# define particle conservation and op dicts
pcon_dict = dict(Np=Np,next_state=next_state,get_Ns_pcon=get_Ns_pcon,get_s0_pcon=get_s0_pcon,
				 count_particles=count_particles,n_sectors=n_sectors)
op_dict = dict(op=op,op_args=op_args)
# create user basiss
basis = user_basis(np.uint32,N,op_dict,allowed_ops=set("+-nI"),sps=sps,pcon_dict=pcon_dict,**maps)
#
#
#
############   create same boson basis_1d object   #############
basis_1d=boson_basis_1d(N,Nb=Np,sps=sps,kblock=0,pblock=1) 
#
#
print(basis)
print(basis_1d)
#
############   create Hamiltonians   #############
#
J=-1.0
U=+1.0
#
hopping=[[+J,j,(j+1)%N] for j in range(N)]
int_bb=[[0.5*U,j,j] for j in range(N)]
int_b=[[-0.5*U,j] for j in range(N)]
#
static=[["+-",hopping],["-+",hopping],["nn",int_bb],["n",int_b]]
dynamic=[]
#
H=hamiltonian(static,[],basis=basis,dtype=np.float64)
H_1d=hamiltonian(static,[],basis=basis_1d,dtype=np.float64)
#
print(H.toarray())
print(H_1d.toarray())
print(np.linalg.norm((H-H_1d).toarray()))