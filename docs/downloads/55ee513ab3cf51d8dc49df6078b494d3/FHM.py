from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinless_fermion_basis_1d, tensor_basis # Hilbert space fermion and tensor bases
import numpy as np # generic math functions
##### define model parameters #####
L=4 # system size
J=1.0 # hopping
U=np.sqrt(2.0) # interaction
mu=0.0 # chemical potential
##### construct Fermi-Hubbard Hamiltonian #####
# define boson basis with 3 states per site L bosons in the lattice
N_up = L//2 + L % 2 # number of fermions with spin up
N_down = L//2 # number of fermions with spin down
basis_up=spinless_fermion_basis_1d(L,Nf=N_up)
basis_down=spinless_fermion_basis_1d(L,Nf=N_down)
basis = tensor_basis(basis_up,basis_down) # spinful fermions
print(basis)
# define site-coupling lists
hop_right=[[-J,i,(i+1)%L] for i in range(L)] #PBC
hop_left= [[+J,i,(i+1)%L] for i in range(L)] #PBC 
pot=[[-mu,i] for i in range(L)] # -\mu \sum_j n_{j \sigma}
interact=[[U,i,i] for i in range(L)] # U/2 \sum_j n_{j,up} n_{j,down}
# define static and dynamic lists
static=[
		['+-|',hop_left],  # up hops left
		['-+|',hop_right], # up hops right
		['|+-',hop_left],  # down hops left
		['|-+',hop_right], # down hops right
		['n|',pot],		   # up on-site potention
		['|n',pot],		   # down on-site potention
		['n|n',interact]   # up-down interaction
							]
dynamic=[]
# build Hamiltonian
no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
