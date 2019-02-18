from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']=str(int(sys.argv[1])) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']=str(int(sys.argv[2])) # set number of MKL threads to run in parallel
#
###########################################################################
#                            example 12                                   #
# In this script we show how to use QuSpin's OpenMP and MKL capabilities. #	
###########################################################################
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
from quspin.operators._make_hamiltonian import _consolidate_static
import numpy as np
from scipy.special import comb
np.random.seed(1) #fixes seed of rng
from time import time # timing package
def run_computation():
	#
	###### define model parameters ######
	J1=1.0 # spin=spin interaction
	J2=0.5 # magnetic field strength
	Omega=8.0 # drive frequency
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
	###### setting up bases ######
	basis_2d = spin_basis_general(N_2d,pauli=False) # making the basis sped up by OpenMP
	print('finished computing basis')
	#
	###### setting up hamiltonian ######
	# set up time-dependence
	def drive(t,Omega):
		return np.cos(Omega*t)
	drive_args=[Omega,]
	# setting up site-coupling lists
	J1_list=[[J1,i,T_x[i]] for i in range(N_2d)] + [[J1,i,T_y[i]] for i in range(N_2d)]
	J2_list=[[J2,i,T_d[i]] for i in range(N_2d)] + [[J2,i,T_a[i]] for i in range(N_2d)]
	#
	static =[ ["xx",J1_list],["yy",J1_list],["zz",J1_list] ]  
	dynamic=[ ["xx",J2_list,drive,drive_args],["yy",J2_list,drive,drive_args],["zz",J2_list,drive,drive_args] ]
	# build hamiltonian
	H=hamiltonian(static,[],basis=basis_2d,dtype=np.float64,check_symm=False,check_herm=False)
	# diagonalise H
	E,V=H.eigsh(time=0.0,k=50,which='LA') # H.eigsh sped up by MKL
	print('finished computing energies')
	psi_0=V[:,0]
	# evolve state
	t=np.linspace(0.0,20*2*np.pi/Omega,21)
	psi_t=H.evolve(psi_0,t[0],t,iterate=True) # H.evolve sped up by OpenMP
	for j,psi in enumerate(psi_t):
		E_t = H.expt_value(psi,time=t[j])
		print("finished evolving up to time step {:d}".format(j) )
# time computation
ti = time() # start timer
run_computation() 
print("single-threded simulation took {0:.4f} sec".format(time()-ti))
