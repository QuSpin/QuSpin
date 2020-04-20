"""
This is an exmple quspin application which can be used to benchmark the package performance.

python quspin_app N_MKL N_OMP L N_T  ,  e.g.,  python quspin_app.py 4 4 5 10000

N_NKL: number of MKL threads
N_OMP: number of OMP threads
L: (3<L<7) linear system size; runtime scales exponentially with L
N_T: number of time steps; runtime scales linearly with N_T

"""

import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['MKL_NUM_THREADS']=sys.argv[1] # set number of MKL threads to run in parallel
os.environ['OMP_NUM_THREADS']=sys.argv[2]

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian,quantum_LinearOperator
from quspin.basis import spin_basis_general
from quspin.tools.evolution import expm_multiply_parallel
import numpy as np 
import time
from scipy.sparse.linalg import eigsh

###### simulation parameters
# system size: expect exponential slowdown of simulation time with increasinf L
L=int(sys.argv[3])
# time evolution parameters: expect linear slowdown with incerasing N_T
N_T=int(sys.argv[4])


######## BASIS CONSTRUCTION ########
#
# required time scales exponentially with L
# linear speedup is expected from OMP

if L>6:
	print("running time expected to take too long!\n \
		   If you'd nonetheless still like to run it, change these lines explicitly \
		   and increase Ns_block_est in the basis construction below appropriately.\n")
	assert(L<=6)

Lx=L
Ly=L
N_sites=L*L

sites = np.arange(N_sites,dtype=np.int32) # sites [0,1,2,....]
	
x = sites%Lx # x positions for sites
y = sites//Lx # y positions for sites

T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction

P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
P_d = y + Lx*x

Z   = -(sites+1) # spin inversion

###### setting up bases ######
ti=time.time()
basis = spin_basis_general(L*L, pauli=False, Ns_block_est=16000000,
								Nup=N_sites//2,
								kxblock=(T_x,0),kyblock=(T_y,0),
								pdblock=(P_d,0),
								pxblock=(P_x,0),pyblock=(P_y,0),
								zblock=(Z,0),
								block_order=['zblock','pdblock','pyblock','pxblock','kyblock','kxblock']
							)
tf=time.time()
time_basis=tf-ti
print("\nbasis with {0:d} states took {1:0.2f} secs.\n".format(basis.Ns, time_basis))



######## HAMILTONIAN CONSTRUCTION ########
#
# required time scales exponentially with L
# linear speedup is expected from both OMP and MKL

J_list=[[1.0,i,T_x[i]] for i in range(N_sites)] + [[1.0,i,T_y[i]] for i in range(N_sites)]
static=[ ["+-",J_list], ["-+",J_list], ["zz",J_list] ]

ti=time.time()
H=hamiltonian(static, [], basis=basis, dtype=np.float64)
H_op=H.aslinearoperator()
H_qop=quantum_LinearOperator(static,basis=basis,dtype=np.float64)
tf=time.time()
time_H=tf-ti
print("\nHamiltonian construction took {0:0.2f} secs.\n".format(time_H))



######## HAMILTONIAN DIAGONALIZATION ########
#
# required time scales exponentially with L
# linear speedup is expected both OMP and MKL


ti=time.time()
eigsh(H_op,k=1,which='SA')
tf=time.time()
time_eigsh_op=tf-ti
print("\ncomputing ground state took {0:0.2f} secs.\n".format(time_eigsh_op))


ti=time.time()
H_qop.eigsh(k=1,which='SA')
tf=time.time()
time_eigsh_qop=tf-ti
print("\ncomputing ground state for quantum_LinearOperator took {0:0.2f} secs.\n".format(time_eigsh_qop))


# calculate minimum and maximum energy only
ti=time.time()
Es, psi_s=H.eigsh(k=2,which="BE",maxiter=1E4,)
tf=time.time()
time_eigsh=tf-ti
print("\ncomputing ground state & most excited state took {0:0.2f} secs.\n".format(time_eigsh))




######## project_from & project_to ########
#
# required time scales exponentially with L
# linear speedup is expected from OMP
ti=time.time()
psi_full = basis.project_from(psi_s[:,0],sparse=False) 
basis.project_to(psi_full)
tf=time.time()
time_proj=tf-ti
print("\nbasis.project_* took {0:0.2f} secs.\n".format(time_proj))



######## TIME EVOLUTION ########
#
# required time scales linearly with N_T and exponentially with L
# linear speedup is expected from OMP


# calculate the eigenstate closest to energy E_star
psi_i=1.0/np.sqrt(2)*(psi_s[:,0] + psi_s[:,1])


dt=0.1
expH = expm_multiply_parallel(H.tocsr(),a=-1j*dt,dtype=np.complex128) 
#
# auxiliary array for memory efficiency
psi=psi_i.copy().astype(np.complex128)
work_array=np.zeros((2*len(psi),), dtype=psi.dtype) # twice as long because complex-valued
#
# loop ober the time steps
ti=time.time()
for j in range(N_T):
	#
	# apply to state psi and update psi in-place
	expH.dot(psi,work_array=work_array,overwrite_v=True)
	
tf=time.time()
time_expm=tf-ti
print("\ntime evolution took {0:0.2f} secs.\n".format(time_expm))





time_tot=time_basis + time_H + time_eigsh_op + time_eigsh_qop + time_eigh + time_expm + time_proj
print("\n\ntotal run time: {0:0.2f} secs.\n".format(time_tot))







