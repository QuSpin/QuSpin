from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
##########################################################################
#                            example 12                                  #
# In this script we demonstrate how to use QuSpin's OpenMP capabilities. #	
##########################################################################
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
from quspin.operators._make_hamiltonian import _consolidate_static
import numpy as np
from scipy.special import comb
np.random.seed(1) #fixes seed of rng
from time import time # timing package
#
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
#
def run_computation():
	#
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
	###### setting up bases ######
	basis_2d = spin_basis_general(N_2d,pauli=False)
	#
	###### setting up hamiltonian ######
	# setting up site-coupling lists
	J1_list=[[J1,i,T_x[i]] for i in range(N_2d)] + [[J1,i,T_y[i]] for i in range(N_2d)]
	J2_list=[[J2,i,T_d[i]] for i in range(N_2d)] + [[J2,i,T_a[i]] for i in range(N_2d)]
	#
	static=[ ["xx",J1_list],["yy",J1_list],["zz",J1_list],  
			 ["xx",J2_list],["yy",J2_list],["zz",J2_list]
			]
	#
	# build hamiltonian
	H=hamiltonian(static,[],basis=basis_2d,dtype=np.float64)
	# diagonalise H
	E,V=H.eigsh(k=50,which='LA')
	print(E)
#
#####
#
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
#
ti = time() # start timer
run_computation()
print("single-threded simulation took {0:.4f} sec".format(time()-ti))
#
##### define number of OpenMP threads used in the simulation
#
os.environ['MKL_NUM_THREADS']='8' # set number of MKL threads to run in parallel
os.environ['OMP_NUM_THREADS']='8'	 # set number of OpenMP threads to run in parallel
#
ti = time() # start timer
run_computation()
print("single-threded simulation took {0:.4f} sec".format(time()-ti))

