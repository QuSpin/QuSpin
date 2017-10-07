from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # bosonic Hilbert space
from quspin.tools.block_tools import block_ops # dynamics in symmetry blocks
import numpy as np # general math functions
#
##### define model parameters
# initial seed for random number generator
np.random.seed(0) # seed is 0 to produce plots from QuSpin2 paper
# setting up parameters of simulation
L = 6 # length of chain
J = 1.0 # top side of ladder hopping
U = 20.0 # Hubbard interaction
#
##### set up Hamiltonian and observables
# define site-coupling lists
int_list_1 = [[-0.5*U,i] for i in range(L)] # interaction $-U/2 \sum_i n_i$
int_list_2 = [[0.5*U,i,i] for i in range(L)] # interaction: $U/2 \num_i n_i^2$
# setting up hopping lists
hop_list = [[-J,i,(i+1)%L] for i in range(L)] # PBC 
# set up static and dynamic lists
static = [
			["+-",hop_list], # hopping
			["-+",hop_list], # hopping h.c.
			["nn",int_list_2], # U n_i^2
			["n",int_list_1] # -U n_i
		]
dynamic = [] # no dynamic operators
# create block_ops object
blocks=[dict(kblock=kblock) for kblock in range(L)] # blocks to project on to
baisis_args = (L,) # boson_basis_1d manditory arguments
basis_kwargs = dict(Nb=L//2,sps=3) # boson_basis_1d optional args
get_proj_kwargs = dict(pcon=True) # set projection to full particle basis
H_block = block_ops(blocks,static,dynamic,boson_basis_1d,baisis_args,np.complex128,
					basis_kwargs=basis_kwargs,get_proj_kwargs=get_proj_kwargs)
#
# setting up local Fock basis
basis = boson_basis_1d(L,Nb=L//2,sps=3)
# setting up observables
no_checks = dict(check_herm=False,check_symm=False,check_pcon=False)
n_list = [hamiltonian([["n",[[1.0,i]]]],[],basis=basis,dtype=np.float64,**no_checks) for i in range(L)]
#
##### time evolution
# set up initial state
i0 = basis.index("111000") # pick state from basis set
psi = np.zeros(basis.Ns,dtype=np.float64)
psi[i0] = 1.0
# print info about setup
state_str = "".join(str(int((basis[i0]//basis.sps**(L-i-1))%basis.sps)) for i in range(L))
print("total H-space size: {}, initial state: |{}>".format(basis.Ns,state_str))
#
##### compute all momentum blocks
H_block.compute_all_blocks()
#
##### calculating the evolved states using matrix exponentiation
# setting up parameters for evolution
start,stop,num = 0,30,301 # 0.1 equally spaced points
times = np.linspace(start,stop,num)
n_jobs = 1 # paralelisation: increase to see if calculation runs faster!
psi_t = H_block.expm(psi,a=-1j,start=start,stop=stop,num=num,block_diag=False,n_jobs=n_jobs)
# calculating the local densities as a function of time
n_t = np.vstack([n.expt_value(psi_t).real for n in n_list]).T
#
##### calculating the evolved state using the evolve method
# setting up parameters for evolution
start,stop,num = 0,30,301 # 0.1 equally spaced points
times = np.linspace(start,stop,num)
psi_t = H_block.evolve(psi,times[0],times)
n_t = np.vstack([n.expt_value(psi_t).real for n in n_list]).T