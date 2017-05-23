from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import fermion_basis_1d # Hilbert space fermion basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation
import numpy as np # generic math functions
#
##### define model parameters #####
L=6 # system size
J=1.0 # uniform hopping contribution
deltaJ=0.1 # bond dimerisation
Delta=0.5 # staggered potential
#
##### construct single-particle Hamiltonian #####
# define basis
basis=fermion_basis_1d(L,Nf=1)
print(basis)
# define site-coupling lists
hop_pm=[[-J-deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
hop_mp=[[+J+deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
stagg_pot=[[Delta*(-1)**i,i] for i in range(L)]	
# define static and dynamic lists
static=[["+-",hop_pm],["-+",hop_mp],['n',stagg_pot]]
dynamic=[]
# build real-space Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
print(H.toarray())
# diagonalise real-space Hamiltonian
E,V=H.eigh()
#
##### compute Fourier transform and momentum-space Hamiltonian #####
# define basis blocks and arguments
blocks=[dict(Nf=1,kblock=i,a=2) for i in range(L//2)] # only L//2 distinct momenta
basis_args = (L,)
# construct block-diagonal Hamiltonian
FT,Hblock = block_diag_hamiltonian(blocks,static,dynamic,fermion_basis_1d,
						basis_args,np.complex128,get_proj_kwargs=dict(pcon=True))
print(np.around(Hblock.toarray(),2))
# diagonalise momentum-space Hamiltonian
Eblock,Vblock=Hblock.eigh()
#
##### plot spectra
import matplotlib.pyplot as plt # plotting library
plt.plot(np.arange(H.Ns),E/L,
					marker='o',color='b',label='real space')
plt.plot(np.arange(Hblock.Ns),Eblock/L,
					marker='x',color='r',markersize=2,label='momentum space')
plt.xlabel('state number',fontsize=16)
plt.ylabel('energy',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid()
plt.tight_layout()
plt.show()
