from __future__ import print_function, division

import sys,os
import argparse

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation tool
import numpy as np # generic math functions

import matplotlib.pyplot as plt

# define model params
L=6 # system size
J=1.0 #uniform hopping
deltaJ=0.2 # hopping difference
Delta=0.2 # staggered potential
# define site-coupling lists
hopping=[[J+deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
stagg_pot=[[Delta,i] for i in range(L)]	
# define basis
basis=spin_basis_1d(L,Nup=1,pauli=False)
basis_args = (L,)
blocks=(dict(Nup=1,pauli=False,kblock=i,a=2) for i in range(L//2))
# define static and dynamic lists
static=[["+-",hopping],["-+",hopping],['z',stagg_pot]]
dynamic=[]
#### calculate Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
E=H.eigvalsh()

_,Hblock = block_diag_hamiltonian(blocks,static,dynamic,spin_basis_1d,basis_args,np.complex128)
Eblock=Hblock.eigvalsh()

#print(_.shape)

print(E)
print(Eblock)

plt.scatter(np.arange(L),E)
plt.show()

