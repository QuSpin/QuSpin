from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
###########################################################################
#                            example 11                                   #
#  In this script we demonstrate how to use QuSpin's methods of           #
#  the general_basis class which do not require explicit calculation      #
#  of the basis itself. Using the J1-J2 model on a square lattice, we     #
#  show how  to estimate the energy of a state using Monte-Carlo sampling.#
###########################################################################
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
from quspin.operators._make_hamiltonian import _consolidate_static
import numpy as np
from scipy.special import comb
np.random.seed(1) #fixes seed of rng
from time import time # timing package
#
ti = time() # start timer
###### define model parameters ######
J1=1.0 # nn interaction
J2=0.5 # nnn interaction
Lx, Ly = 4, 4 # linear dimension of 2d lattice
N_2d = Lx*Ly # number of sites
#
###### setting up user-defined symmetry transformations for 2d lattice ######
sites = np.arange(N_2d) # site labels [0,1,2,....]
x = sites%Lx # x positions for sites
y = sites//Lx # y positions for sites
#
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
#
T_a = (x+1)%Lx + Lx*((y+1)%Ly) # translation along anti-diagonal
T_d = (x-1)%Lx + Lx*((y+1)%Ly) # translation along diagonal
#
P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis
P_d = y + Lx*x # reflection about diagonal
#
Z   = -(sites+1) # spin inversion
#
###### setting up operator string for Hamiltonian matrix elements H_{ss'} ######
# setting up site-coupling lists for the J1-J2 model on a 2d square lattice
J1_list=[[J1,i,T_x[i]] for i in range(N_2d)] + [[J1,i,T_y[i]] for i in range(N_2d)]
J2_list=[[J2,i,T_d[i]] for i in range(N_2d)] + [[J2,i,T_a[i]] for i in range(N_2d)]
# setting up opstr list
static=[["xx",J1_list],["yy",J1_list],["zz",J1_list],  ["xx",J2_list],["yy",J2_list],["zz",J2_list]]
# convert static list to format which is easy to use with the basis_general.Op and basis_general.Op_bra_ket methods. 
static_formatted = _consolidate_static(static)
#
###### setting up basis object without computing the basis (make=False) ######
basis = spin_basis_general(N_2d, pauli=0, make_basis=False, 
			Nup=N_2d//2, 
			kxblock=(T_x,0), kyblock=(T_y,0),
			pxblock=(P_x,0), pyblock=(P_y,0), pdblock=(P_d,0),
			zblock=(Z,0),
			block_order=['zblock','pdblock','pyblock','pxblock','kyblock','kxblock'] # momentum symmetry comes last for speed
		)
print(basis) # examine basis: contains a single element because it is not calculated due to make_basis=False argument above.
print('basis is empty [note argument make_basis=False]')
#
###### define quantum state to compute the energy of using Monte-Carlo sampling ######
#
# auxiliary basis, only needed for probability_amplitude(); not needed in a proper variational ansatz.
aux_basis = spin_basis_general(N_2d, pauli=0, make_basis=True, 
			Nup=N_2d//2, 
			kxblock=(T_x,0), kyblock=(T_y,0),
			pxblock=(P_x,0), pyblock=(P_y,0), pdblock=(P_d,0),
			zblock=(Z,0),
			block_order=['zblock','pdblock','pyblock','pxblock','kyblock','kxblock'] # momentum symmetry comes last for speed
		)
# set quantum state to samplee from to be GS of H 
H = hamiltonian(static,[],basis=aux_basis,dtype=np.float64)
E, V = H.eigsh(k=2, which='SA') # need NOT be (but can be) normalized
psi=(V[:,0] + V[:,1])/np.sqrt(2)
#
##### define proposal function #####
#
def swap_bits(s,i,j):
	""" Swap bits i, j in integer s.

	Parameters
	-----------
	s: int
		spin configuration stored in bit representation.
	i: int
		lattice site position to be swapped with the corresponding one in j.
	j: int
		lattice site position to be swapped with the corresponding one in i.

	"""
	x = ( (s>>i)^(s>>j) ) & 1
	return s^( (x<<i)|(x<<j) )
#
##### define function to compute the amplitude `psi_s` for every spin configuration `s` #####
basis_state_inds_dict=dict()
for s in aux_basis.states:
	basis_state_inds_dict[s]=np.where(aux_basis.states==s)[0][0]
def probability_amplitude(s,psi):
	''' Computes probability amplitude `psi_s` of quantum state `psi` in z-basis state `s`.

	Parameters
	----------
	s: array_like(int)
		array of spin configurations [stored in their bit representation] to compute their local energies `E_s`.
	psi_s: array
		(unnormalized) probability amplitude values, corresponding to the states `s`. 

	'''
	return psi[[basis_state_inds_dict[ss] for ss in s]]
#
##### define function to compute local energy `E_s` #####
#
def compute_local_energy(s,psi_s,psi):
	"""Computes local energy E_s for a spin configuration s.

	Parameters
	----------
	s: array_like(int)
		array of spin configurations [stored in their bit representation] to compute their local energies `E_s`.
	psi_s: array
		(unnormalized) probability amplitude values, corresponding to the states `s` in the symmetry-reduced basis. 
	psi: array
		(unnormalized) array which encodes the mapping $s \\to \\psi_s$ (e.g. quantum state vector) in the symmetry-reduced basis.
		
	"""
	# preallocate variable
	E_s=np.zeros(s.shape,dtype=np.float64)
	#
	# to compute local energy `E_s` we need matrix elements `H_{ss'}` for the operator `H`.
	# These can be computed by looping overthe static list without constructing the operator matrix. 
	for opstr,indx,J in static_formatted:
		# for every state `s`, compute the state it connects to `s'`, and the corresponding matrix element `ME`
		ME,bras,kets = basis.Op_bra_ket(opstr,indx,J,np.float64,s,reduce_output=False)
		# performs sum over `s'`
		E_s+=ME * probability_amplitude(bras,psi)
	# normalize by `psi_s`
	E_s/=psi_s
	return E_s 
#
##### perform Monte Carlo sampling from `|psi_s|^2` ##### 
#
# draw random spin configuratio
s=[0 for _ in range(N_2d//2)] + [1 for _ in range(N_2d//2)]
np.random.shuffle(s)
s = np.array(s)
s=''.join([str(j) for j in s])
print('random initial state in bit representation:', s)
# transform state in bit representation
s=int(s,2)
print('same random initial state in integer representation:', s)
# compute representative of state `s` under basis symmetries (here only Z-symmetry)
s=basis.representative(s)
print('representative of random initial state in integer representation:', s)
# compute amplitude in state s
psi_s=probability_amplitude(s,psi)
#
psi_s_full_basis=np.copy(psi_s)
# overwrite the symmetry-reduced space psi_s_full_basis with its amplitude in the full basis
basis.get_amp(s,amps=psi_s_full_basis,mode='representative') # has the advantage that basis need not be made.
#
# define MC sampling parameters
equilibration_time=200
autocorrelation_time=N_2d
# number of MC sampling points
N_MC_points = 1000
#
##### run Markov chain MC #####
#
# compute all distinct site pairs to swap
#
# preallocate variables
E_s=np.zeros(N_MC_points,dtype=np.float64)
MC_sample=np.zeros(N_MC_points,dtype=basis.dtype)
#
j=0 # set MC chain counter
k=0 # set MC sample counter 
while k<N_MC_points:
	# propose new state t by swapping two random bits
	t=s
	while t==s: # repeat until a different state is reached
		# draw two random sites
		site_i=np.random.randint(0,N_2d)
		site_j=np.random.randint(0,N_2d)
		# swap bits in spin configurations
		t = swap_bits(s,site_i,site_j) # new state t corresponds to integer with swapped bites i and j
	#
	# compute representatives or proposed configurations to bring them back to symmetry sector
	t=basis.representative(t)
	psi_t=probability_amplitude(t, psi)
	# CAUTION: symmetries break detailed balance which we need to restore by using the amplitudes in the full basis.
	psi_t_full_basis=np.copy(psi_t)
	# overwrite the symmetry-reduced space psi_t_full_basis with its amplitude in the full basis
	basis.get_amp(t,amps=psi_t_full_basis,mode='representative') # has the advantage that basis need not be made.
	#
	### accept/reject new state
	#
	# use amplitudes psi_t_full_basis and psi_s_full_basis to restore detailed balance
	eps=np.random.uniform(0,1)
	if eps * np.abs(psi_s_full_basis)**2 <= np.abs(psi_t_full_basis)**2: 
		s=t
		psi_s=psi_t
		psi_s_full_basis=psi_t_full_basis
	#
	# wait for MC chain to quilibrate and collect uncorrelated samples
	if (j>equilibration_time) and (j%autocorrelation_time==0):
		# compute local energy
		print('computing local energy E_s for MC sample {0:d}'.format(k))
		E_s[k] = compute_local_energy(s,psi_s,psi)
		# update sample
		MC_sample[k]=s
		# update MC samples counter
		k+=1
	#
	j+=1 # update MC chain counter
#
##### compute MC-sampled average energy #####
# compute energy expectation and MC variance
E_mean=np.mean(E_s)
E_var_MC=np.std(E_s)/np.sqrt(N_MC_points)
#
# compute exact expectation value
E_exact=H.expt_value(psi/np.linalg.norm(psi))
#####   compute full basis   #####
# so far the functions representative(), get_amp(), and Op_bra_ket() did not require to compute the full basis
basis.make(Ns_block_est=16000)
print(basis) # after the basis is made, printing the basis returns the states
# compare results
print('mean energy: {0:.4f}, MC variance: {1:.4f}, exact energy {2:.4f}'.format(E_mean, E_var_MC, E_exact) )
print("simulation took {0:.4f} sec".format(time()-ti))
