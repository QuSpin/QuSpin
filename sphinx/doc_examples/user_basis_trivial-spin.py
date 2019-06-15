from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis_1d
from quspin.basis.user import user_basis # Hilbert space user basis
from quspin.basis.user import next_state_sig_32,op_sig_32,map_sig_32 # user basis data types
from numba import carray,cfunc # numba helper functions
from numba import uint32,int32 # numba data types
#
N=6 # lattice sites
#
######   create spin-1/2 user basis object   ######
#
###  function to call when applying operators
@cfunc(op_sig_32,
	locals=dict(s=int32,n=int32,b=uint32), )
def op(op_struct_ptr,op_str,ind,N):
	# using struct pointer to pass op_structults 
	# back to C++ see numba Records
	op_struct = carray(op_struct_ptr,1)[0]
	err = 0
	#
	ind = N - ind - 1 # convention for QuSpin for mapping from bits to sites.
	n = (op_struct.state>>ind)&1 # either 0 or 1
	s = (((op_struct.state>>ind)&1)<<1)-1 # either -1 or 1
	b = (1<<ind)
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
#
###  function to implement magnetization/particle conservation
@cfunc(next_state_sig_32,
	locals=dict(t=uint32), )
def next_state(s,counter,N,args):
	""" implements magnetization conservation. """
	if(s==0): return s;
	#
	t = (s | (s - 1)) + 1
	return t | ((((t & (0-t)) // (s & (0-s))) >> 1) - 1)
#
### function to count number of particles
def bit_count(x,l):
	""" works for 32-bit integers. """
	x = x & ((0x7FFFFFFF) >> (31 - l));
	x = x - ((x >> 1) & 0x55555555);
	x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
	return (((x + (x >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
#
###  define symmetry maps
#
def translation(x,N,sign_ptr):
	""" works for all system sizes N. """
	#
	shift = 1 # translate state by shift sites
	period = N # periodicity/cyclicity of translation
	Imax = (1<<N)-1 # maximum integer
	#
	l = (shift+period)%period
	I1 = (x >> (period - l))
	I2 = ((x << l) & Imax)
	#
	return (I2 | I1)
#
def parity(x,N,sign_ptr):
	""" works for all system sizes N. """
	#
	out = 0
	s = N-1
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
#
@cfunc(map_sig_32,
	locals=dict(Imax=uint32,))
def spin_inversion(x,N,sign_ptr):
	""" works for all system sizes N. """
	#
	Imax = (1<<N)-1 # maximum integer
	return x^Imax
#
#####  create maps dictionary  #####
#
maps = dict(T=(translation,N,0), P=(parity,2,0), Z=(spin_inversion,2,0))



# create spin-1/2 basis_1d object
basis_1d=spin_basis_1d(N,Nup=N//2)#,kblock=0,pblock=1,zblock=1)

print(basis_1d)
s=2
t=spin_inversion(s,N,0)
print(basis_1d.int_to_state(s), basis_1d.int_to_state(t))

