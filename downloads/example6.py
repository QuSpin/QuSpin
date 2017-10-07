from __future__ import print_function, division
from quspin.operators import hamiltonian,exp_op,ops_dict # operators
from quspin.basis import tensor_basis,fermion_basis_1d # Hilbert spaces
from quspin.tools.measurements import obs_vs_time # calculating dynamics
import numpy as np # general math functions
from numpy.random import uniform,choice # tools for doing random sampling
from time import time # tool for calculating computation time
import matplotlib.pyplot as plt # plotting library
#
##### setting parameters for simulation
# simulation parameters
n_real = 100 # number of realizations
n_boot = 100 # number of bootstrap samples to calculate error
# physical parameters
L = 8 # system size
N = L//2 # number of particles
N_up = N//2 + N % 2 # number of fermions with spin up
N_down = N//2 # number of fermions with spin down
w_list = [1.0,4.0,10.0] # disorder strength
J = 1.0 # hopping strength
U = 5.0 # interaction strength
# range in time to evolve system
start,stop,num=0.0,35.0,101
t = np.linspace(start,stop,num=num,endpoint=True)
#
###### create the basis
# build the two bases to tensor together to spinful fermions
basis_up = fermion_basis_1d(L,Nf=N_up) # up basis
basis_down = fermion_basis_1d(L,Nf=N_down) # down basis
basis = tensor_basis(basis_up,basis_down) # spinful fermions
#
##### create model
# define site-coupling lists
hop_right = [[-J,i,i+1] for i in range(L-1)] # hopping to the right OBC
hop_left = [[J,i,i+1] for i in range(L-1)] # hopping to the left OBC
int_list = [[U,i,i] for i in range(L)] # onsite interaction
# site-coupling list to create the sublattice imbalance observable
sublat_list = [[(-1.0)**i/N,i] for i in range(0,L)]
# create static lists
operator_list_0 = [	
			["+-|", hop_left], # up hop left
			["-+|", hop_right], # up hop right
			["|+-", hop_left], # down hop left
			["|-+", hop_right], # down hop right
			["n|n", int_list], # onsite interaction
			]
imbalance_list = [["n|",sublat_list],["|n",sublat_list]]
# create operator dictionary for ops_dict class
# add key for Hubbard hamiltonian
operator_dict=dict(H0=operator_list_0)
# add keys for local potential in each site
for i in range(L):
	# add to dictioanry keys h0,h1,h2,...,hL with local potential operator
	operator_dict["n"+str(i)] = [["n|",[[1.0,i]]],["|n",[[1.0,i]]]]
#
###### setting up operators	
# set up hamiltonian dictionary and observable (imbalance I)
no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
H_dict = ops_dict(operator_dict,basis=basis,**no_checks)
I = hamiltonian(imbalance_list,[],basis=basis,**no_checks)
# strings which represent the initial state
s_up = "".join("1000" for i in range(N_up))
s_down = "".join("0010" for i in range(N_down))
# basis.index accepts strings and returns the index 
# which corresponds to that state in the basis list
i_0 = basis.index(s_up,s_down) # find index of product state
psi_0 = np.zeros(basis.Ns) # allocate space for state
psi_0[i_0] = 1.0 # set MB state to be the given product state
print("H-space size: {:d}, initial state: |{:s}>(x)|{:s}>".format(basis.Ns,s_up,s_down))
#
# define function to do dynamics for different disorder realizations.
def real(H_dict,I,psi_0,w,t,i):
	# body of function goes below
	ti = time() # start timing function for duration of reach realisation
	# create a parameter list which specifies the onsite potential with disorder
	params_dict=dict(H0=1.0)
	for j in range(L):
		params_dict["n"+str(j)] = uniform(-w,w)
	# using the parameters dictionary construct a hamiltonian object with those
	# parameters defined in the list
	H = H_dict.tohamiltonian(params_dict)
	# use exp_op to get the evolution operator
	U = exp_op(H,a=-1j,start=t.min(),stop=t.max(),num=len(t),iterate=True)
	psi_t = U.dot(psi_0) # get generator psi_t for time evolved state
	# use obs_vs_time to evaluate the dynamics
	t = U.grid # extract time grid stored in U, and defined in exp_op
	obs_t = obs_vs_time(psi_t,t,dict(I=I))
	# print reporting the computation time for realization
	print("realization {}/{} completed in {:.2f} s".format(i+1,n_real,time()-ti))
	# return observable values
	return obs_t["I"]
#
###### looping over differnt disorder strengths
for w in w_list:	
	I_data = np.vstack([real(H_dict,I,psi_0,w,t,i) for i in range(n_real)])
	##### averaging and error estimation
	I_avg = I_data.mean(axis=0) # get mean value of I for all time points
	# generate bootstrap samples
	bootstrap_gen = (I_data[choice(n_real,size=n_real)].mean(axis=0) for i in range(n_boot)) 
	# generate the fluctuations about the mean of I
	sq_fluc_gen = ((bootstrap-I_avg)**2 for bootstrap in bootstrap_gen)
	I_error = np.sqrt(sum(sq_fluc_gen)/n_boot) 
	##### plotting results
	plt.errorbar(t,I_avg,I_error,marker=".",label="w={:.2f}".format(w))
# configuring plots
plt.xlabel("$t/J$",fontsize=18)
plt.ylabel("$\mathcal{I}$",fontsize=18)
plt.grid(True)
plt.tick_params(labelsize=16)
plt.legend(loc=0)
plt.tight_layout()
plt.savefig('fermion_MBL.pdf', bbox_inches='tight')
plt.show()