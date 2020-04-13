from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
########################################################################
#                            example 22                                #	
#                      ...                                             #
########################################################################
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import expm_multiply_parallel
import numpy as np
#
##### define model parameters #####
L=10 # system size
T=1.5 # period of switching
N_steps=100 # of driving cycles
Jxy=np.sqrt(2.0) # xy interaction
Jzz_0=1.0 # zz interaction
hz=1.0/np.sqrt(3.0) # z external field
#
##### set up Heisenberg Hamiltonian in an external z-field #####
# compute basis
basis = spin_basis_1d(L,S="1",Nup=L//2,kblock=0,pblock=1,) # and positive parity sector
print('total number of basis states {}'.format(basis.Ns))
# define operators with OBC using site-coupling lists
J_zz = [[Jzz_0,i,(i+1)%L] for i in range(L)] # OBC
J_xy = [[0.5*Jxy,i,(i+1)%L] for i in range(L)] # OBC
h_z=[[hz,i] for i in range(L)]
# static and dynamic lists
static_ave = [["+-",J_xy],["-+",J_xy],["zz",J_zz],["z",h_z],]
static_0 = [["+-",J_xy],["-+",J_xy],]
static_1 = [["zz",J_zz],["z",h_z],]
dynamic=[]
# compute the time-dependent Heisenberg Hamiltonian
H_ave = 0.5*hamiltonian(static_ave,dynamic,basis=basis,dtype=np.float64)
H0 = hamiltonian(static_0,dynamic,basis=basis,dtype=np.float64)
H1 = hamiltonian(static_1,dynamic,basis=basis,dtype=np.float64)
#
##### various exact diagonalisation routines #####
# calculate full eigensystem
E,V=H_ave.eigsh(k=1,which='SA')
psi_i=V[:,0]
#
# auxiliary array for memory efficiency
psi=psi_i.copy().astype(np.complex128)
work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued

# construct unitaries
expH0 = expm_multiply_parallel(H0.tocsr(),a=-1j*0.5*T)
expH1 = expm_multiply_parallel(H1.tocsr(),a=-1j*0.5*T) 

# preallocate variables
E_density=np.zeros(N_steps+1,dtype=np.float64)
Sent_density=np.zeros(N_steps+1,dtype=np.float64)


E_density[0]=H_XXZ.expt_value(psi).real/L
Sent_density[0]=basis.ent_entropy(psi,sub_sys_A=range(L//2),density=True)['Sent_A']


for j in range(N_steps):
	
	# apply to state psi and update psi in-place
	expH0.dot(psi,work_array=work_array,overwrite_v=True) 
	expH1.dot(psi,work_array=work_array,overwrite_v=True)

	E_density[j+1]=H_XXZ.expt_value(psi).real/L
	Sent_density[j+1]=basis.ent_entropy(psi,sub_sys_A=range(L//2),density=True)['Sent_A']

	print("finished evolving {0:d} step".format(j))


import matplotlib.pyplot as plt # plotting library
times=T*np.arange(N_steps+1)
plt.plot(times, E_density)
plt.plot(times, Sent_density)
plt.show()


#plt.close()
