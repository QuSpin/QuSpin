# check quspin installation
import sys, os
sys.path.append("/usr/local/lib/python3.10/site-packages") # path to local installation in google drive

import quspin
print('\nquspin {} installed successfully and ready for use!\n'.format(quspin.__version__) )

# set options and import required packages
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='4' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

# import plotting tool
from matplotlib import pyplot as plt
plt.rcParams.update({
    "text.usetex": True, # enable latex font
    "font.family": "Helvetica", # set font style
    "text.latex.preamble": r'\usepackage{amsmath}', # add latex packages
    "font.size": "16", # set font size
})

# import numpy
import numpy as np
# suppress machine precision; print numbers up to three decimals
np.set_printoptions(suppress=True,precision=3)

from quspin.basis import spin_basis_1d

basis_1 = spin_basis_1d(L=1) # single spin-1/2 particle / qubit / two-level system
print(basis_1)



# define state |0> = |down>
psi_down = np.array([0.0,1.0])
print('\n|down> = {0}'.format(psi_down) )

# define state |1> = |up>
psi_up = np.array([1.0,0.0])
print('\n|up> = {0}'.format(psi_up) )

# define a superposition state: (|0>+|1>)/sqrt(2)
psi_plus = np.array([1.0,1.0])/np.sqrt(2)
print('|+> = \n1/sqrt(2)(|0>+|1>) = {0}'.format(psi_plus) )


# compute overlap <up|+>
overlap = psi_up.conj() @ psi_plus
print('\n<up|+> = {0}'.format(overlap) )

# compute overlap squared |<up|+>|^2
print('\n|<up|+>|^2 = {0}'.format(np.abs(overlap)**2) )


from quspin.operators import hamiltonian

# define coupling strengths
hx, hy, hz = 1.0, 1.0, 1.0

# associate each coupling to the spin labeled 0
hx_list = [[hx,0],] # coupling: hx multiplies spin labeled 0
hy_list = [[hy,0],] # coupling: hy multiplies spin labeled 0
hz_list = [[hz,0],] # coupling: hz multiplies spin labeled 0

# associate a Pauli matrix to each coupling list
static_terms = [['x',hx_list], # assign coupling list hx_list to Pauli operator sigma^x
                ['y',hy_list], # assign coupling list hy_list to Pauli operator sigma^y
                ['z',hz_list], # assign coupling list hz_list to Pauli operator sigma^z
               ]
dynamic_terms = [] # ignore this for the time being (we'll come to it later)

# construct Hamiltonian in the single-spin basis defined above
H_1 = hamiltonian(static_terms, dynamic_terms, basis=basis_1,)


print(H_1)

print('cast as a dense array:\n{}\n'.format(H_1.toarray()))

print('cast as a scipy csr (compressed sparse row) array:\n{}\n'.format(H_1.tocsr()) )

print('cast as a scipy csc (compressed sparse column) array:\n{}\n'.format(H_1.tocsc()) )


phi = H_1.dot(psi_plus) # |phi> = H|+>

print('\n|+> = {}'.format(psi_plus) )

print('\nH_1|+> = {}'.format(phi) )

# compute expectation value <down|H|down>
expt_H_1_psi = H_1.expt_value(psi_down)

print('\n<down|H|down> = {}'.format(expt_H_1_psi) )


### two coupled spins / qubits
basis_2 = spin_basis_1d(L=2)
print(basis_2)

# compute integer to its bit string
bit_str = '{0:02b}'.format(1)
print('\nthe integer 1 corresponds to the bitstring of length 2: {}'.format(bit_str) )

# map integer repr. to quant state
print('\nuse quspin to map the integer 1 to a bitstring of length 2: {}'.format(basis_2.int_to_state(1)) )


print('\nuse quspin to map the quantum state |01> to its integer rep: {}'.format(basis_2.state_to_int('01')) )

# find array index of a given basis state
print('\narray index of a basis state |01>: {}'.format(basis_2.index('01')) )



### define state |01>
psi_01 = np.zeros(basis_2.Ns) # initialize an empty array
psi_01[basis_2.index('01')]=1.0 # set index corresponding to state '01' to value 1.0
print('\nstate |01> = {}'.format(psi_01) )

### define state the Bell state 1/sqrt(2)(|01> + |10>)
psi_Bell = np.zeros(basis_2.Ns) # empty array
psi_Bell[basis_2.index('01')]=1.0 # set index corresponding to state '01' to value 1.0
psi_Bell[basis_2.index('10')]=1.0 # set index corresponding to state '10' to value 1.0
psi_Bell/=np.linalg.norm(psi_Bell) # normalize state
print('\nBell state 1/sqrt(2)(|01> + |10>) = {}'.format(psi_Bell) )



# define couplings
Jxy, Jzz = 1.0, 1.0

# define site-coupling lists
Jxy_list = [[0.5*Jxy, 0,1],] # coupling strength Jxy/2 couples spin 0 and spin 1
Jzz_list = [[Jzz, 0,1],] # coupling strength Jzz couples spin 0 and spin 1

# associate the Pauli matrix terms to each coupling list
H_terms = [['+-',Jxy_list], # + operator of spin 0 multiplies - operator of spin 1
           ['-+',Jxy_list], # - operator of spin 0 multiplies + operator of spin 1
           ['zz',Jzz_list], # z operator of spin 0 multiplies z operator of spin 1
          ]
# create Hamiltonian using basis_2
H_Heis = hamiltonian(H_terms,[], basis=basis_2)
#
# print Hamiltonian matrix
print('\n2-spin Heisenberg Hamiltonian:\n{}\n'.format(H_Heis.toarray()) )


# diagonalize Hamiltonian
E, V = H_Heis.eigh()

# ordered e'energies
print("\ne'energies (spectrum) = {}".format(E) )
# corresponding e'states in columns
print("\ne'states stored in columns:\n{}".format(V) )

# read off ground state (GS) and its energy
E_GS, psi_GS = E[0], V[:,0]


entropy = basis_2.ent_entropy(psi_GS, sub_sys_A=[0,],)
print(entropy)

entropy = basis_2.ent_entropy(psi_GS, sub_sys_A=[0,],return_rdm='A', alpha=1)
print(entropy)



Sent_A = entropy['Sent_A']
rdm_A = entropy['rdm_A']

print('\nvN entropy per spin = {}'.format(Sent_A) )
print('\nreduced density matrix of subsystem A:\n{}'.format(rdm_A))


basis_symm = spin_basis_1d(L=2, pblock=1)
print(basis_symm)


# create Hamiltonian
H_Heis_symm = hamiltonian(H_terms,[], basis=basis_symm)
#
# print Hamiltonian matrix
print('\n2-spin Heisenberg model in with parity symmetry\n{}'.format(H_Heis_symm.toarray()) )

# print e'values
E = H_Heis_symm.eigvalsh()
print("\ne'values in the p=+1 sector: {}".format(E) )


# system size / number of spins
L = 10

# compute basis
basis_ising = spin_basis_1d(L=L,)

# define site-coupling lists
Jzz_list = [[Jzz, j,j+1] for j in range(L-1)] # L-1 bonds
hz_list = [[hz,j] for j in range(L)] # L sites
hx_list = [[hx,j] for j in range(L)] # L sites

# define Hamiltonian terms
H_terms = [['zz',Jzz_list],
           ['x',hx_list],
           ['z',hz_list],
          ]
H_Ising = hamiltonian(H_terms,[], basis=basis_ising)

# cast to array
print('\nIsing Hamiltonian matrix:\n {}\n'.format(H_Ising.toarray()) )

# compute eigenenergies and eigenstates
E, V = H_Ising.eigh()
# ground state
psi_GS = V[:,0]
E_GS = E[0]
print('\nGS energy = {}'.format(E_GS) )
# first-excited state
psi_ex_1 = V[:,1]
E_exc = E[1]
print('\nfirst excited state energy = {}\n'.format(E_exc) )



# compute histogram
DOS, energy =  np.histogram(E, bins=50) # number of bins can be adjusted

# plot histogram
plt.plot(energy[1:], DOS) # energy array has one vertex more than the DOS array
plt.xlabel('energy, $E/J$')
plt.ylabel('$\\rho(E)$')
plt.show()


# half-chain entanglement entropy of ground state
vN_entropy = basis_ising.ent_entropy(psi_GS, sub_sys_A=[j for j in range(L//2)], alpha=1, density=False)
print('\nvon Neumann entropy = {}'.format(vN_entropy['Sent_A']) )

# half-chain Reniy-2 entropy of ground state
Renyi_entropy = basis_ising.ent_entropy(psi_GS, sub_sys_A=[j for j in range(L//2)], alpha=2, density=False)
print('\nRenyi-2 entropy = {}\n'.format(Renyi_entropy['Sent_A']) )


from scipy.sparse.linalg import expm # matrix exponential for sparse generators

### define times vector and timestep
N_timesteps = 101 # number of timesteps
times, dt = np.linspace(0.0,5.0,N_timesteps, retstep=True)

### compute unitary
U_dt = expm(-1j*dt*H_Ising.tocsc()) # cast `hamiltonian' object H as a sparse matrix first

### initial state: cat state (|00...0> + |11...1>)/sqrt(2)
psi_cat = np.zeros(basis_ising.Ns) # empty array of size basis_ising.Ns = 2**L
psi_cat[basis_ising.index('0'*L)]=1.0 # set index corresponding to state '00...0' to value 1.0
psi_cat[basis_ising.index('1'*L)]=1.0 # set index corresponding to state '11...1' to value 1.0
psi_cat/=np.linalg.norm(psi_cat) # normalize state

### pre-allocate memory to store value of observables
psi_cat_t = np.zeros((basis_ising.Ns,N_timesteps), dtype=np.complex128 ) # to-be evolved state
E_t = np.zeros(N_timesteps, dtype=np.float64 ) # to-be energy expectations
Sent_t = np.zeros(N_timesteps, dtype=np.float64 ) # to-be entanglement entropies density

### compute time evolution using a recursive time loop
psi_cat_t[:,0] = psi_cat[:] # set initial state, [:] copies memory view
E_t[0] = H_Ising.expt_value(psi_cat).real # set initial energy value
Sent_t[0] = basis_ising.ent_entropy(psi_cat,)['Sent_A'] # set initial entanglement entropy value

# run time evolution
for j in np.arange(N_timesteps-1):
    # evolve
    psi_cat_t[:,j+1] = U_dt @ psi_cat_t[:,j] # evolve state one step forward
    # measure
    E_t[j+1] = H_Ising.expt_value(psi_cat_t[:,j+1]).real # <psi(t)|H_Ising|psi(t)>
    Sent_t[j+1] = basis_ising.ent_entropy(psi_cat_t[:,j+1])['Sent_A'] # entanglement density of half chain

### plot results
plt.plot(times, E_t/basis_ising.L, label='$E(t)/L$')
plt.plot(times, Sent_t, label='$S_\mathrm{ent}^\mathrm{vN}(t)/L_A$')
plt.xlabel('time $Jt$')
plt.legend()
plt.show()



# construct basis
L = 10 # three spins
basis_ising_t = spin_basis_1d(L=L,)

# define coupling lists
Jzz_list = [[Jzz,j,j+1] for j in range(L-1)] # L-1 bonds
hz_list  = [[hz,j] for j in range(L)] # L sites
hx_list  = [[hx,j] for j in range(L)] # L sites

# define drive function
omega = 4.0 # drive frequency
def drive(t, omega): # first argument must be time, followed by extra parameters if any
    return np.cos(omega*t)
drive_args=(omega,) # tuple containing parameters of drive (all arguments passed, except time t)

# define Hamiltonian
static_terms = [['zz',Jzz_list], ['x',hx_list], ] # as before (see above)
dynamic_terms =[['z',hz_list, drive, drive_args],] # add drive function and its arguments to this list
H_Ising_t = hamiltonian(static_terms,dynamic_terms, basis=basis_ising_t) # note we also pass dynamic_terms

print(H_Ising_t)


print(H_Ising_t(time=0.4321).toarray()) # evaluate H(t) at some fixed time t

E_inst = H_Ising_t.expt_value(psi_cat,time=0.9876) # compute <cat|H(t)|cat>

# evolving the initial state psi_0, from initial time times[0]
psi_t = H_Ising_t.evolve(psi_cat, times[0], times) # contains solution at all time points in times
print('shape of evolved state array is {}'.format(psi_t.shape) )


# compute observables
E_t_drive = H_Ising_t.expt_value(psi_t,time=0.0).real # measure <cat(t)|H(0)|cat(t)>
Sent_t_drive = basis_ising_t.ent_entropy(psi_t,)['Sent_A'] # measure Sent (density) in evolved state |psi(t)>

# plot results
plt.plot(times, E_t_drive/basis_ising_t.L, label='$E(t)/L$')
plt.plot(times, Sent_t_drive, label='$S_\mathrm{ent}^\mathrm{vN}(t)/L_A$')
plt.xlabel('time $Jt$')
plt.legend()
plt.show()

