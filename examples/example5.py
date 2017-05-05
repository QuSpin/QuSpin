from __future__ import print_function, division

import sys,os
import argparse

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from quspin.basis import fermion_basis_1d # Hilbert space spin basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation tool
import numpy as np # generic math functions
import matplotlib.pyplot as plt
try: # import python 3 zip function in python 2 and pass if python 3
    import itertools.izip as zip
except ImportError:
    pass 
	


# define model params
L=64 # system size
J=1.0 #uniform hopping
deltaJ=0.1 # hopping difference
Delta=0.0 # staggered potential
# define site-coupling lists
hop_pm=[[+J+deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
hop_mp=[[-J-deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
stagg_pot=[[Delta*(-1)**i,i] for i in range(L)]	
# define basis
basis=fermion_basis_1d(L,Nf=1)
basis_args = (L,)
blocks=[dict(Nf=1,kblock=i,a=2) for i in range(L//2)]
# define static and dynamic lists
static=[["+-",hop_pm],["-+",hop_mp],['n',stagg_pot]]
dynamic=[]
#### calculate Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
E,V=H.eigh()
# define Fourier transform and momentum-space Hamiltonian
FT,Hblock = block_diag_hamiltonian(blocks,static,dynamic,fermion_basis_1d,basis_args,np.complex128,get_proj_kwargs=dict(pcon=True))
Eblock,Vblock=Hblock.eigh()

plt.scatter(np.arange(L),E)
plt.show()

# construct Fermi sea
psi_0=Vblock





# construct operator n_{j=0} and n_{j=L/2}
n_i_static=[['n',[[1.0,0]]]]
n_i=hamiltonian(n_i_static,[],basis=basis,dtype=np.float64,check_herm=False,check_pcon=False)
n_f_static=[['n',[[1.0,L//2]]]]
n_f=hamiltonian(n_f_static,[],basis=basis,dtype=np.float64,check_herm=False,check_pcon=False)
# transform n_j to momentum space
n_i=n_i.rotate_by(FT,generator=False)
n_f=n_f.rotate_by(FT,generator=False)


# evaluate nonequal time correlator <FS|n_f(t) n_i(0)|FS>
t=np.linspace(0.0,90.0,901)
psi_n_0=n_i.dot(psi_0)

# evolve via exp_op class (sometimes faster )
U = exp_op(Hblock,a=-1j,start=t.min(),stop=t.max(),num=len(t),iterate=True)
psi_t=U.dot(psi_0)
psi_n_t = U.dot(psi_n_0)

# evolve using Hamiltonian class
#psi_t=Hblock.evolve(psi_0,0.0,t,iterate=True)
#psi_n_t=Hblock.evolve(psi_n_0,0.0,t,iterate=True)

# compute correlator
n_i_0=n_i.matrix_ele(psi_0,psi_0,diagonal=True) # expectation of n_i at t=0
correlators=np.zeros(t.shape+psi_0.shape[1:])

for i, (psi,psi_n) in enumerate( zip(psi_t,psi_n_t) ):
	correlators[i,:]= ( n_f.matrix_ele(psi,psi_n,diagonal=True) \
						 - n_f.matrix_ele(psi, psi,diagonal=True)*n_i_0  ).real





# evaluate thermal expectation value
T=1.0 # set temperature for Fermi-Dirac distribution
correlator = (correlators/(np.exp(E/T)+1.0)).sum(axis=-1) 
plt.plot(t,correlator)
plt.show()




