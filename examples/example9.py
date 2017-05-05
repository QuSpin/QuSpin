from __future__ import print_function, division

import sys,os
import argparse

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

import numpy as np
from numpy.random import uniform,choice
from quspin.basis import tensor_basis,fermion_basis_1d
from quspin.operators import hamiltonian,exp_op,ops_dict
from quspin.tools.measurements import obs_vs_time
from joblib import Parallel,delayed
import matplotlib.pyplot as plt
from time import time



# setting parameters for simulation
L = 12 # system size
N = L//2 # number of particles
w = 3.0 # disorder strength
J = 1.0 # hopping strength
U = 4.7 # interaction strength
k = 0.1 # trap stiffness

# range to do dynamics
start=0
stop=35
num=101

n_jobs = 2 # number of cores to use in calculating realizations
n_real = 30 # number of realizations
N_up = N//2 + N % 2 # number of fermions with spin up
N_down = N//2 # number of fermions with spin down
i_mid = (L//2+1 if L%2 else L//2+0.5) # mid point on lattice
# building the two basis to tensor together
basis_up = fermion_basis_1d(L,Nf=N_up) # up basis
basis_down = fermion_basis_1d(L,Nf=N_down) # down basis
# setting up full spinful fermion basis using tensor basis class
basis = tensor_basis(basis_up,basis_down)
print("H-space size: {:d}".format(basis.Ns))
# creating coupling lists
J_right = [[J,i,i+1] for i in range(L-1)] # hopping to the right
J_left = [[-J,i,i+1] for i in range(L-1)] # hopping to the left
U_list = [[U,i,i] for i in range(L)] # onsite interaction
trap_list = [[0.5*k*(i-i_mid)**2,i] for i in range(L)] # harmonic trap
# coupling list to create the sublattice imbalance observable
sublat_list = [[(-1)**i,i] for i in range(0,L)]
# create static lists
operator_list_0 = [	
			["+-|",J_left], # up hopping
			["-+|",J_right], 
			["|+-",J_left], # down hopping
			["|-+",J_right],
			["n|n",U_list], # onsite interaction
			["n|",trap_list], # trap potential
			["|n",trap_list],
		 ]
# create operator dictionary for ops_dict class
# creates a dictioanry with keys h0,h1,h2,...,hL for local potential
operator_dict = {"h"+str(i):[["n|",[[1.0,i]]],["|n",[[1.0,i]]]] for i in range(L)}
operator_dict["H0"]=operator_list_0
# set up hamiltonian dictionary and observable
no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
H_dict = ops_dict(operator_dict,basis=basis,**no_checks)
I = hamiltonian([["n|",sublat_list],["|n",sublat_list]],[],basis=basis,**no_checks)/N
# strings which represent the initial state
s_up = "".join("1000" for i in range(N_up))
s_down = "".join("0010" for i in range(N_down))
# basis.index accepts strings and returns the index 
# which corresponds to that state in the basis list
i_0 = basis.index(s_up,s_down)
psi_0 = np.zeros(basis.Ns)
psi_0[i_0] = 1.0
# set up times
times = np.linspace(start,stop,num=num,endpoint=True)
# define function to do dynamics for different disorder realizations.
def realization(H_dict,I,psi_0,disorder,start,stop,num,i):
	ti = time() # start timing function
	# create a parameter list which specifies the onsite potential with disorder
	parameters = {"h"+str(i):h for i,h in enumerate(disorder)}
	# using the parameters dictionary construct a hamiltonian object with those
	# parameters defined in the list
	H = H_dict.tohamiltonian(parameters)
	# use exp_op to get the evolution operator
	U = exp_op(H,a=-1j,start=start,stop=stop,num=num,endpoint=True)
	psi_t = U.dot(psi_0) # get generator of time evolution
	# use obs_vs_time to evaluate the dynamics
	obs_t = obs_vs_time(psi_t,U.grid,dict(I=I))
	# print reporting the computation time for realization
	print("realization {}/{} completed in {:.2f} s".format(i+1,n_real,time()-ti))
	# return observable values.
	return obs_t["I"]

# machinery for doing parallel realizations loop
I_data = np.vstack(Parallel(n_jobs=n_jobs)(delayed(realization)(H_dict,I,psi_0,uniform(-w,w,size=L),start,stop,num,i) for i in range(n_real)))
# calculating mean and error via bootstrap sampling
I = I_data.mean(axis=0) # get mean value of I
n_boot = 100*n_real
# generate bootstrap samples
bootstrap_gen = (I_data[choice(n_real,n_real)].mean(axis=0) for i in range(n_boot))
# generate the fluctuations about the mean of I
sq_fluc_gen = ((bootstrap-I)**2 for bootstrap in bootstrap_gen)
# error is calculated as the squareroot of mean fluctuations
dI = np.sqrt(sum(sq_fluc_gen)/n_boot)
# plot imbalance with error bars
fig = plt.figure()
plt.xlabel("$t/J$",fontsize=18)
plt.ylabel("$\mathcal{I}$",fontsize=18)
plt.grid(True)
plt.tick_params(labelsize=16)
plt.errorbar(times,I,dI,marker=".")
fig.savefig('example9.pdf', bbox_inches='tight')
plt.show()
