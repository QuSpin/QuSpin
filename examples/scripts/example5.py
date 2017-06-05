from quspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from quspin.basis import fermion_basis_1d # Hilbert space fermion basis
from quspin.tools.block_tools import block_diag_hamiltonian # block diagonalisation
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
try: # import python 3 zip function in python 2 and pass if using python 3
    import itertools.izip as zip
except ImportError:
    pass 
##### define model parameters #####
L=100 # system size
J=1.0 # uniform hopping contribution
deltaJ=0.1 # bond dimerisation
Delta=0.5 # staggered potential
beta=100.0 # set inverse temperature for Fermi-Dirac distribution
##### construct single-particle Hamiltonian #####
# define site-coupling lists
hop_pm=[[-J-deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
hop_mp=[[+J+deltaJ*(-1)**i,i,(i+1)%L] for i in range(L)] # PBC
stagg_pot=[[Delta*(-1)**i,i] for i in range(L)]	
# define static and dynamic lists
static=[["+-",hop_pm],["-+",hop_mp],['n',stagg_pot]]
dynamic=[]
# define basis
basis=fermion_basis_1d(L,Nf=1)
# build real-space Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
# diagonalise real-space Hamiltonian
E,V=H.eigh()
##### compute Fourier transform and momentum-space Hamiltonian #####
# define basis blocks and arguments
blocks=[dict(Nf=1,kblock=i,a=2) for i in range(L//2)] # only L//2 distinct momenta
basis_args = (L,)
# construct block-diagonal Hamiltonian
FT,Hblock = block_diag_hamiltonian(blocks,static,dynamic,fermion_basis_1d,
						basis_args,np.complex128,get_proj_kwargs=dict(pcon=True))
# diagonalise momentum-space Hamiltonian
Eblock,Vblock=Hblock.eigh()
##### prepare the density observables and initial states #####
# grab single-particle states and treat them as initial states
psi0=Vblock
# construct operator n_1 = $n_{j=0}$
n_1_static=[['n',[[1.0,0]]]]
n_1=hamiltonian(n_1_static,[],basis=basis,dtype=np.float64,
				check_herm=False,check_pcon=False)
# construct operator n_2 = $n_{j=L/2}$
n_2_static=[['n',[[1.0,L//2]]]]
n_2=hamiltonian(n_2_static,[],basis=basis,dtype=np.float64,
				check_herm=False,check_pcon=False)
# transform n_j operators to momentum space
n_1=n_1.rotate_by(FT,generator=False)
n_2=n_2.rotate_by(FT,generator=False)
##### evaluate nonequal time correlator <FS|n_2(t) n_1(0)|FS> #####
# define time vector
t=np.linspace(0.0,90.0,901)
# calcualte state acted an by n_1
n_psi0=n_1.dot(psi0)
# construct time-evolution operator using exp_op class (sometimes faster)
U = exp_op(Hblock,a=-1j,start=t.min(),stop=t.max(),num=len(t),iterate=True)
# evolve states
psi_t=U.dot(psi0)
n_psi_t = U.dot(n_psi0)
# alternative method for time evolution using Hamiltonian class
#psi_t=Hblock.evolve(psi0,0.0,t,iterate=True)
#n_psi_t=Hblock.evolve(n_psi0,0.0,t,iterate=True)
# preallocate variable
correlators=np.zeros(t.shape+psi0.shape[1:])
# loop over the time-evolved states
for i, (psi,n_psi) in enumerate( zip(psi_t,n_psi_t) ):
	correlators[i,:]=n_2.matrix_ele(psi,n_psi,diagonal=True).real
# evaluate correlator at finite temperature
n_FD=1.0/(np.exp(beta*E)+1.0)
correlator = (n_FD*correlators).sum(axis=-1)
##### plot spectra
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
##### plot correlator
plt.plot(t,correlator,linewidth=2)
plt.xlabel('$t$',fontsize=16)
plt.ylabel('$C_{0,L/2}(t,\\beta)$',fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid()
plt.tight_layout()
plt.show()




