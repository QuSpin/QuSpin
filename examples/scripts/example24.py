# -*- coding: utf-8 -*-
from __future__ import unicode_literals
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
from numba import uint32,int32,float64,complex128 # numba data types
import numpy as np
from scipy.special import comb
#
N=1 # lattice sites
sps=3 # states per site
Np=N # N//2 # total number of bosons
#
############   create boson user basis object   #############
#
######  function to call when applying operators
@cfunc(op_sig_32,
	locals=dict(b=uint32,occ=int32,sps=uint32,me_offdiag=complex128,me_diag=float64), )
def op(op_struct_ptr,op_str,site_ind,N,args):
	# using struct pointer to pass op_struct_ptr back to C++ see numba Records
	op_struct = carray(op_struct_ptr,1)[0]
	err = 0
	sps=3
	me_offdiag=1.0;
	me_diag=1.0;
	#
	site_ind = N - site_ind - 1 # convention for QuSpin for mapping from bits to sites.
	occ = (op_struct.state//sps**site_ind)%sps # occupation
	b = sps**site_ind
	#
	if op_str==43: # "+" is integer value 43 = ord("+")
		if (occ+1)%sps != 1:
			me_offdiag *= np.sqrt((occ+1)%sps)
		else:
			me_offdiag *= np.sqrt(2.0)
		op_struct.state += (b if (occ+1)<sps else 0)

	elif op_str==45: # "-" is integer value 45 = ord("-")
		if occ != 1:
			me_offdiag *= np.sqrt(occ);
		else:
			me_offdiag *= np.sqrt(2.0);
		op_struct.state -= (b if occ>0 else 0)

	elif op_str==122: # "z" is integer value 122 = ord("n")
		me_diag *= occ-1

	# define the 8 Gell-Mann matrices

	elif op_str==49: # "1" is integer value 49 = ord("1"): lambda_1 Gell-Mann matrix
		if occ==2:
			op_struct.state -= b
		elif occ==1:
			op_struct.state += b
		else:
			me_offdiag *= 0.0

	elif op_str==50: # "2" is integer value 50 = ord("2"): lambda_2 Gell-Mann matrix
		if occ==2:
			op_struct.state -= b
			me_offdiag *= 1.0j
		elif occ==1:
			op_struct.state += b
			me_offdiag *= -1.0j
		else:
			me_offdiag *= 0.0

	elif op_str==51: # "3" is integer value 51 = ord("3"): lambda_3 Gell-Mann matrix
		if occ==1:
			me_diag*=-1.0
		elif occ==0:
			me_diag*=0.0

	elif op_str==52: # "4" is integer value 52 = ord("4"): lambda_4 Gell-Mann matrix
		if occ==2:
			op_struct.state -= 2*b
		elif occ==0:
			op_struct.state += 2*b
		else:
			me_offdiag *= 0.0

	elif op_str==53: # "5" is integer value 53 = ord("5"): lambda_5 Gell-Mann matrix
		if occ==2:
			op_struct.state -= 2*b
			me_offdiag *= 1.0j
		elif occ==0:
			op_struct.state += 2*b
			me_offdiag *= -1.0j
		else:
			me_offdiag *= 0.0

	elif op_str==54: # "6" is integer value 54 = ord("6"): lambda_6 Gell-Mann matrix
		if occ==1:
			op_struct.state -= b
		elif occ==0:
			op_struct.state += b
		else:
			me_offdiag *= 0.0

	elif op_str==55: # "7" is integer value 55 = ord("7"): lambda_7 Gell-Mann matrix
		if occ==1:
			op_struct.state -= b
			me_offdiag *= 1.0j
		elif occ==0:
			op_struct.state += b
			me_offdiag *= -1.0j
		else:
			me_offdiag *= 0.0

	elif op_str==56: # "8" is integer value 56 = ord("8"): lambda_8 Gell-Mann matrix
		if occ==2:
			me_diag*=1.0/np.sqrt(3.0)
		elif occ==1:
			me_diag*=1.0/np.sqrt(3.0)
		else:
			me_diag*=-2.0/np.sqrt(3.0)

	elif op_str==73: # "I" is integer value 73 = ord("I")
		pass
	else:
		me_diag = 0.0
		err = -1
	#
	op_struct.matrix_ele *= me_diag*me_offdiag
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
	sps = 3 # use as global variable
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
@cfunc(map_sig_32,
	locals=dict(out=uint32,sps=uint32,i=int32) )
def inversion(x,N,sign_ptr,args):
	""" works for all system sizes N. """
	out = 0

	sps = args[0]
	for i in range(N):
		out += ( sps-x%sps-1 ) * (sps**i)
		x //= sps
	#
	return out
Z_args=np.array([sps],dtype=np.uint32)
#
######  define function to count particles in bit representation
#
@cfunc(count_particles_sig_32,
	locals=dict(s=uint32,))
def count_particles(x,p_number_ptr,args):
	""" Counts number of particles/spin-ups in a state stored in integer representation for up to N=32 sites """
	#
	s = x # integer x cannot be changed
	for i in range(args[0]):
		p_number_ptr[0] += s%args[1]
		s /= args[1]
n_sectors=1 # number of particle sectors
count_particles_args=np.array([N,sps],dtype=np.int32)
#
######  construct user_basis
# define maps dict
maps = dict(T_block=(translation,N,0,T_args),P_block=(parity,2,0,P_args),Z_block=(inversion,2,0,Z_args), )
# define particle conservation and op dicts
pcon_dict = dict(Np=Np,next_state=next_state,next_state_args=next_state_args,
				 get_Ns_pcon=get_Ns_pcon,get_s0_pcon=get_s0_pcon,
				 count_particles=count_particles,count_particles_args=count_particles_args,n_sectors=n_sectors)
op_dict = dict(op=op,op_args=op_args)
# create user basiss
basis = user_basis(np.uint32,N,op_dict,allowed_ops=set("+-zI12345678"),sps=sps,) # **maps) #,pcon_dict=pcon_dict
#
print(basis)
#
############   create Hamiltonians   #############
#
J=-1.0
U=+1.0
#
hopping=[[+J,j,(j+1)%N] for j in range(N)]
int_bb=[[0.5*U,j,j] for j in range(N)]
int_b=[[-0.5*U,j] for j in range(N)]
ones=[[1.0,j] for j in range(N)]
#
static=[['1',ones],] #[["+-",hopping],["-+",hopping],["nn",int_bb],["n",int_b]]
dynamic=[]
#
no_checks=dict(check_symm=False, check_pcon=False, check_herm=False)
H=hamiltonian(static,[],basis=basis,dtype=np.complex128,**no_checks)
#
print(H.toarray())


