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
J1=1.0 # spin=spin interaction
J2=0.5 # magnetic field strength
Lx, Ly = 4, 4 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
#
###### setting up user-defined symmetry transformations for 2d lattice ######
sites = np.arange(N_2d) # sites [0,1,2,....]
x = sites%Lx # x positions for sites
y = sites//Lx # y positions for sites
#
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction
#
T_a = (x+1)%Lx + Lx*((y+1)%Ly) # translation along anti-diagonal
T_d = (x-1)%Lx + Lx*((y+1)%Ly) # translation along diagonal
#
Z   = -(sites+1) # spin inversion
#
###### setting up operator string for Hamiltonian matrix elements H_{ss'} ######
# setting up site-coupling lists
J1_list=[[J1,i,T_x[i]] for i in range(N_2d)] + [[J1,i,T_y[i]] for i in range(N_2d)]
J2_list=[[J2,i,T_d[i]] for i in range(N_2d)] + [[J2,i,T_a[i]] for i in range(N_2d)]
# setting up opstr list
static=[["xx",J1_list],["yy",J1_list],["zz",J1_list],  ["xx",J2_list],["yy",J2_list],["zz",J2_list]]
# convert static list to format which is easy to use with the basis_general.Op and basis_general.Op_bra_ket methods. 
static_formatted = _consolidate_static(static)
#
###### setting up basis object without computing the basis ######
# Z-symmetry allowed in sampling since it commutes with the swap operation which proposes new configurations in MC
basis = spin_basis_general(N_2d, pauli=0, make_basis=False,Nup=N_2d//2,zblock=(Z,0) )
print(basis) # examine basis: contains a single element because it is not calculated due to make_basis=False argument above.
print('basis is empty [note argument make_basis=False]')
#
###### define quantum state to compute the energy of using Monte-Carlo sampling ######
#
Ns=int(comb(N_2d,N_2d//2)/2) # number of states in Hilbert space 
psi = np.random.normal(size=Ns) # need NOT be normalized
#
##### define proposal function #####
#
def swap_bits(s,i,j):
	""" Swap bits i, j in integer s.

	Parameters
	-----------
	s: array_like(int)
		array of spin configurations stored in their bit representation.
	i: array_like(int)
		array of positions to be swapped with the corresponding pair in j.
	j: array_like(int)
		array of positions to be swapped with the corresponding pair in i.

	"""
	x = np.bitwise_and( np.bitwise_xor( np.right_shift(s,i), np.right_shift(s,j) ) , 1)
	return np.bitwise_xor(s, np.bitwise_or( np.left_shift(x,i), np.left_shift(x,j) ) )
#
##### define function to compute the amplitude `psi_s` for every spin configuration `s` #####
#
aux_basis = spin_basis_general(N_2d,make_basis=True,Nup=N_2d//2,zblock=(Z,0) ) # auxiliary basis, only needed for probability_amplitude()
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
		(unnormalized) probability amplitude values, corresponding to the states `s`. 
	psi: array
		(unnormalized) state which encodes the mapping $s \\to \\psi_s$.
		
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
# define MC sampling parameters
equilibration_time=200
# number of MC sampling points
N_MC_points = 100000
#
##### run Markov chain MC #####
#
# compute all distinct site pairs to swap
site_pairs=np.array([(i,j) for i in sites for j in sites if i!=j])
np.random.shuffle(site_pairs)
# branch MC chains: allows to run mutiple Markov chains in a vectorized fashion
s=np.tile(s,N_MC_points)
psi_s=np.tile(psi_s,N_MC_points)
#
# sample from MC after allowing the equilibration time to pass
for k in range(equilibration_time):
	# draw set of pair of sites to swap their spins 
	inds=np.random.randint(site_pairs.shape[0],size=N_MC_points)
	inds=site_pairs[inds]
	# swap bits in spin configurations
	t = swap_bits(s,inds[:,0], inds[:,1])
	# compute representatives or proposed configurations to bring them back to symmetry sector
	# CAUTION: MC works only with Z-symmetry because it commutes with swap_bits(); other symmetries break detailed balance.
	t=basis.representative(t) 
	### accept/reject new states
	psi_t=probability_amplitude(t, psi)
	eps=np.random.uniform(0,1,size=N_MC_points)
	# compute mask to determine whether to accept/reject every state
	mask=np.where(eps <= np.abs(psi_t)**2/np.abs(psi_s)**2)
	# apply mask to only update accepted states
	s[mask]=t[mask]
	psi_s[mask]=psi_t[mask]
	#
	print('{0:d}-th MC chain equilibration step complete'.format(k))
#
##### compute MC-sampled energy #####
# compute local energy
print('computing local energies E_s...')
E_s = compute_local_energy(s,psi_s,psi)
# compute energy expectation and MC variance
E_mean=np.mean(E_s)
E_var=np.std(E_s)/np.sqrt(N_MC_points)
#
#### compute exact expectation value #####
# compute full basis: required to construct the exact Hamiltonian
basis.make(Ns_block_est=16000)
print(basis) # after the basis is made, printing the basis returns the states
# build Hamiltonian
H = hamiltonian(static,[],basis=basis,dtype=np.float64)
E_exact=H.expt_value(psi/np.linalg.norm(psi))
# compare results
print('mean energy: {0:.4f}, MC variance: {1:.4f}, exact energy {2:.4f}'.format(E_mean, E_var, E_exact) )
print("simulation took {0:.4f} sec".format(time()-ti))
