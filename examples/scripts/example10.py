from __future__ import print_function, division
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import tensor_basis,spinless_fermion_basis_1d,boson_basis_1d # bases
from quspin.tools.measurements import obs_vs_time # calculating dynamics
from quspin.tools.Floquet import Floquet_t_vec # period-spaced time vector
import numpy as np # general math functions
import matplotlib.pyplot as plt # plotting library
#
##### setting up parameters for simulation
# physical parameters
L = 6 # system size
Nf, Nb = L//2, L # number of fermions, bosons
N = Nf + Nb # total number of particles
Jb, Jf = 1.0, 1.0 # boson, fermon hopping strength
Uff, Ubb, Ubf = -2.0, 0.5, 5.0  # bb, ff, bf interaction
# define time-dependent perturbation
A = 2.0
Omega = 1.0
def drive(t,Omega):
	return np.sin(Omega*t)
drive_args=[Omega]
#
###### create the basis
# build the two bases to tensor together to a bose-fermi mixture
basis_b=boson_basis_1d(L,Nb=Nb,sps=3) # boson basis
basis_f=spinless_fermion_basis_1d(L,Nf=Nf) # fermion basis
basis=tensor_basis(basis_b,basis_f) # BFM
#
##### create model
# define site-coupling lists
hop_b = [[-Jb,i,(i+1)%L] for i in range(L)] # b hopping
int_list_bb = [[Ubb/2.0,i,i] for i in range(L)] # bb onsite interaction
int_list_bb_lin = [[-Ubb/2.0,i] for i in range(L)] # bb interaction, linear term
# 
hop_f_right = [[-Jf,i,(i+1)%L] for i in range(L)] # f hopping right
hop_f_left = [[Jf,i,(i+1)%L] for i in range(L)] # f hopping left
int_list_ff = [[Uff,i,(i+1)%L] for i in range(L)] # ff nearest-neighbour interaction
drive_f = [[A*(-1.0)**i,i] for i in range(L)] # density staggered drive
#
int_list_bf = [[Ubf,i,i] for i in range(L)] # bf onsite interaction
# create static lists
static = [	
			["+-|", hop_b], # bosons hop left
			["-+|", hop_b], # bosons hop right
			["n|", int_list_bb_lin], # bb onsite interaction
			["nn|", int_list_bb], # bb onsite interaction
			#
			["|+-", hop_f_left], # fermions hop left
			["|-+", hop_f_right], # fermions hop right
			["|nn", int_list_ff], # ff nn interaction
			#
			["n|n", int_list_bf], # bf onsite interaction
			]
dynamic = [["|n",drive_f,drive,drive_args]] # drive couples to fermions only
#
###### set up Hamiltonian and initial states
no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
H_BFM = hamiltonian(static,dynamic,basis=basis,**no_checks)
# define initial Fock state through strings
s_f = "".join("1" for i in range(Nf)) + "".join("0" for i in range(L-Nf))
s_b = "".join("1" for i in range(Nb))
# basis.index accepts strings and returns the index which corresponds to that state in the basis list
i_0 = basis.index(s_b,s_f) # find index of product state in basis
psi_0 = np.zeros(basis.Ns) # allocate space for state
psi_0[i_0] = 1.0 # set MB state to be the given product state
print("H-space size: {:d}, initial state: |{:s}>|{:s}>".format(basis.Ns,s_b,s_f))
#
###### time evolve initial state and measure entanglement between species
t=Floquet_t_vec(Omega,10,len_T=10) # t.vals=times, t.i=initial time, t.T=drive period
psi_t = H_BFM.evolve(psi_0,t.i,t.vals,iterate=True)
# measure observable
Sent_args=dict(basis=basis,sub_sys_A="left")
meas = obs_vs_time(psi_t,t.vals,{},Sent_args=Sent_args)
# read off measurements
Entropy_t = meas["Sent_time"]["Sent_A"]
#
######
# configuring plots
plt.plot(t/t.T, Entropy_t)
plt.xlabel("$\\mathrm{driving\\ cycle}$",fontsize=18)
plt.ylabel('$S_\\mathrm{ent}(t)$',fontsize=18)
plt.grid(True)
plt.tick_params(labelsize=16)
plt.tight_layout()
plt.savefig('BFM.pdf', bbox_inches='tight')
plt.show()