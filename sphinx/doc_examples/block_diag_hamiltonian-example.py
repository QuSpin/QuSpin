from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space fermion basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation
import numpy as np # generic math functions
##### define model parameters #####
L=6 # system size
J=1.0 # uniform hopping contribution
deltaJ=0.1 # bond dimerisation
Delta=0.5 # staggered potential
##### construct single-particle Hamiltonian #####
# define site-coupling lists
hop=[[-J-deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
stagg_pot=[[Delta*(-1)**i,i] for i in range(L)]	
# define static and dynamic lists
static=[["+-",hop],["-+",hop],['n',stagg_pot]]
dynamic=[]
# define basis
basis=boson_basis_1d(L,Nb=1,sps=2)
# build real-space Hamiltonian in full basis
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
# diagonalise real-space Hamiltonian
E,V=H.eigh()
print(np.around(H.toarray(),2))
##### compute Fourier transform and block-diagonal momentum-space Hamiltonian #####
# define momentm blocks and basis arguments
blocks=[dict(Nb=1,sps=2,kblock=i,a=2) for i in range(L//2)] # only L//2 distinct momenta
basis_args = (L,)
# construct block-diagonal Hamiltonian
FT,Hblock = block_diag_hamiltonian(blocks,static,dynamic,boson_basis_1d,
						basis_args,np.complex128,get_proj_kwargs=dict(pcon=True))
# diagonalise momentum-space Hamiltonian
Eblock,Vblock=Hblock.eigh()
print(np.around(Hblock.toarray(),2))