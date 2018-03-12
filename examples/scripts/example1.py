from __future__ import print_function, division
import sys,os
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
##################################################################
#                            example 1                           #
#    In this script we demonstrate how to use QuSpin's           #
#    functionality to solve the time-dependent Schroedinger      #
#    equation with a time-dependent operator. We also showcase   #
#    a function which is used to calculate the entanglement      #
#    entropy of a pure state.                                    #
##################################################################
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import ent_entropy, diag_ensemble # entropies
from numpy.random import ranf,seed # pseudo random numbers
from joblib import delayed,Parallel # parallelisation
import numpy as np # generic math functions
from time import time # timing package
#
##### define simulation parameters #####
n_real=20 # number of disorder realisations
n_jobs=2 # number of spawned processes used for parallelisation
#
##### define model parameters #####
L=10 # system size
Jxy=1.0 # xy interaction
Jzz_0=1.0 # zz interaction at time t=0
h_MBL=3.9 # MBL disorder strength
h_ETH=0.1 # delocalised disorder strength
vs=np.logspace(-2.0,0.0,num=20,base=10) # log_2-spaced vector of ramp speeds
#
##### set up Heisenberg Hamiltonian with linearly varying zz-interaction #####
# define linear ramp function
v = 1.0 # declare ramp speed variable
def ramp(t):
	return (0.5 + v*t)
ramp_args=[]
# compute basis in the 0-total magnetisation sector (requires L even)
basis = spin_basis_1d(L,Nup=L//2,pauli=False)
# define operators with OBC using site-coupling lists
J_zz = [[Jzz_0,i,i+1] for i in range(L-1)] # OBC
J_xy = [[Jxy/2.0,i,i+1] for i in range(L-1)] # OBC
# static and dynamic lists
static = [["+-",J_xy],["-+",J_xy]]
dynamic =[["zz",J_zz,ramp,ramp_args]]
# compute the time-dependent Heisenberg Hamiltonian
H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
#
##### calculate diagonal and entanglement entropies #####
def realization(vs,H_XXZ,basis,real):
	"""
	This function computes the entropies for a single disorder realisation.
	--- arguments ---
	vs: vector of ramp speeds
	H_XXZ: time-dep. Heisenberg Hamiltonian with driven zz-interactions
	basis: spin_basis_1d object containing the spin basis
	n_real: number of disorder realisations; used only for timing
	"""
	ti = time() # get start time
	#
	global v # declare ramp speed v a global variable
	#
	seed() # the random number needs to be seeded for each parallel process
	#
	# draw random field uniformly from [-1.0,1.0] for each lattice site
	unscaled_fields=-1+2*ranf((basis.L,))
	# define z-field operator site-coupling list
	h_z=[[unscaled_fields[i],i] for i in range(basis.L)]
	# static list
	disorder_field = [["z",h_z]]
	# compute disordered z-field Hamiltonian
	no_checks={"check_herm":False,"check_pcon":False,"check_symm":False}
	Hz=hamiltonian(disorder_field,[],basis=basis,dtype=np.float64,**no_checks)
	# compute the MBL and ETH Hamiltonians for the same disorder realisation
	H_MBL=H_XXZ+h_MBL*Hz
	H_ETH=H_XXZ+h_ETH*Hz
	#
	### ramp in MBL phase ###
	v=1.0 # reset ramp speed to unity
	# calculate the energy at infinite temperature for initial MBL Hamiltonian
	eigsh_args={"k":2,"which":"BE","maxiter":1E4,"return_eigenvectors":False}
	Emin,Emax=H_MBL.eigsh(time=0.0,**eigsh_args)
	E_inf_temp=(Emax+Emin)/2.0
	# calculate nearest eigenstate to energy at infinite temperature
	E,psi_0=H_MBL.eigsh(time=0.0,k=1,sigma=E_inf_temp,maxiter=1E4)
	psi_0=psi_0.reshape((-1,))
	# calculate the eigensystem of the final MBL hamiltonian
	E_final,V_final=H_MBL.eigh(time=(0.5/vs[-1]))
	# evolve states and calculate entropies in MBL phase
	run_MBL=[_do_ramp(psi_0,H_MBL,basis,v,E_final,V_final) for v in vs]
	run_MBL=np.vstack(run_MBL).T
	#
	###  ramp in ETH phase ###
	v=1.0 # reset ramp speed to unity
	# calculate the energy at infinite temperature for initial ETH hamiltonian
	Emin,Emax=H_ETH.eigsh(time=0.0,**eigsh_args)
	E_inf_temp=(Emax+Emin)/2.0
	# calculate nearest eigenstate to energy at infinite temperature
	E,psi_0=H_ETH.eigsh(time=0.0,k=1,sigma=E_inf_temp,maxiter=1E4)
	psi_0=psi_0.reshape((-1,))
	# calculate the eigensystem of the final ETH hamiltonian
	E_final,V_final=H_ETH.eigh(time=(0.5/vs[-1]))
	# evolve states and calculate entropies in ETH phase
	run_ETH=[_do_ramp(psi_0,H_ETH,basis,v,E_final,V_final) for v in vs]
	run_ETH=np.vstack(run_ETH).T # stack vertically elements of list run_ETH
	# show time taken
	print("realization {0}/{1} took {2:.2f} sec".format(real+1,n_real,time()-ti))
	#
	return run_MBL,run_ETH
#
##### evolve state and evaluate entropies #####
def _do_ramp(psi_0,H,basis,v,E_final,V_final):
	"""
	Auxiliary function to evolve the state and calculate the entropies after the
	ramp.
	--- arguments ---
	psi_0: initial state
	H: time-dependent Hamiltonian
	basis: spin_basis_1d object containing the spin basis (required for Sent)
	E_final, V_final: eigensystem of H(t_f) at the end of the ramp t_f=1/(2v)
	"""
	# determine total ramp time
	t_f = 0.5/v 
	# time-evolve state from time 0.0 to time t_f
	psi = H.evolve(psi_0,0.0,t_f)
	# calculate entanglement entropy
	subsys = range(basis.L//2) # define subsystem
	Sent = ent_entropy(psi,basis,chain_subsys=subsys)["Sent"]
	# calculate diagonal entropy in the basis of H(t_f)
	S_d = diag_ensemble(basis.L,psi,E_final,V_final,Sd_Renyi=True)["Sd_pure"]
	#
	return np.asarray([S_d,Sent])
#
##### produce data for n_real disorder realisations #####
# __name__ == '__main__' required to use joblib in Windows.
if __name__ == '__main__':
	"""
	# alternative way without parallelisation
	data = np.asarray([realization(vs,H_XXZ,basis,i) for i in range(n_real)])
	"""
	data = np.asarray(Parallel(n_jobs=n_jobs)(delayed(realization)(vs,H_XXZ,basis,i) for i in range(n_real)))
	#
	run_MBL,run_ETH = zip(*data) # extract MBL and data
	# average over disorder
	mean_MBL = np.mean(run_MBL,axis=0)
	mean_ETH = np.mean(run_ETH,axis=0)
	#
	##### plot results #####
	import matplotlib.pyplot as plt
	### MBL plot ###
	fig, pltarr1 = plt.subplots(2,sharex=True) # define subplot panel
	# subplot 1: diag enetropy vs ramp speed
	pltarr1[0].plot(vs,mean_MBL[0],label="MBL",marker=".",color="blue") # plot data
	pltarr1[0].set_ylabel("$s_d(t_f)$",fontsize=22) # label y-axis
	pltarr1[0].set_xlabel("$v/J_{zz}(0)$",fontsize=22) # label x-axis
	pltarr1[0].set_xscale("log") # set log scale on x-axis
	pltarr1[0].grid(True,which='both') # plot grid
	pltarr1[0].tick_params(labelsize=16)
	# subplot 2: entanglement entropy vs ramp speed
	pltarr1[1].plot(vs,mean_MBL[1],marker=".",color="blue") # plot data
	pltarr1[1].set_ylabel("$s_\mathrm{ent}(t_f)$",fontsize=22) # label y-axis
	pltarr1[1].set_xlabel("$v/J_{zz}(0)$",fontsize=22) # label x-axis
	pltarr1[1].set_xscale("log") # set log scale on x-axis
	pltarr1[1].grid(True,which='both') # plot grid
	pltarr1[1].tick_params(labelsize=16)
	# save figure
	plt.tight_layout()
	fig.savefig('example1_MBL.pdf', bbox_inches='tight')
	#
	### ETH plot ###
	fig, pltarr2 = plt.subplots(2,sharex=True) # define subplot panel
	# subplot 1: diag enetropy vs ramp speed
	pltarr2[0].plot(vs,mean_ETH[0],marker=".",color="green") # plot data
	pltarr2[0].set_ylabel("$s_d(t_f)$",fontsize=22) # label y-axis
	pltarr2[0].set_xlabel("$v/J_{zz}(0)$",fontsize=22) # label x-axis
	pltarr2[0].set_xscale("log") # set log scale on x-axis
	pltarr2[0].grid(True,which='both') # plot grid
	pltarr2[0].tick_params(labelsize=16)
	# subplot 2: entanglement entropy vs ramp speed
	pltarr2[1].plot(vs,mean_ETH[1],marker=".",color="green") # plot data
	pltarr2[1].set_ylabel("$s_\mathrm{ent}(t_f)$",fontsize=22) # label y-axis
	pltarr2[1].set_xlabel("$v/J_{zz}(0)$",fontsize=22) # label x-axis
	pltarr2[1].set_xscale("log") # set log scale on x-axis
	pltarr2[1].grid(True,which='both') # plot grid
	pltarr2[1].tick_params(labelsize=16)
	# save figure
	plt.tight_layout()
	fig.savefig('example1_ETH.pdf', bbox_inches='tight')
	#
	plt.show() # show plots
	#plt.close()
