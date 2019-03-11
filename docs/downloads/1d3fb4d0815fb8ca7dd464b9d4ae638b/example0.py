from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#####################################################################
#                            example 0                              #
#    In this script we demonstrate how to use QuSpin's exact        #
#    diagonlization routines to solve for the eigenstates and       #
#    energies of the XXZ chain.                                     #
#####################################################################
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
#
##### define model parameters #####
L=12 # system size
Jxy=np.sqrt(2.0) # xy interaction
Jzz_0=1.0 # zz interaction
hz=1.0/np.sqrt(3.0) # z external field
#
##### set up Heisenberg Hamiltonian in an external z-field #####
# compute spin-1/2 basis
#basis = spin_basis_1d(L,pauli=False)
#basis = spin_basis_1d(L,pauli=False,Nup=L//2) # zero magnetisation sector
basis = spin_basis_1d(L,pauli=False,Nup=L//2,pblock=1) # and positive parity sector
# define operators with OBC using site-coupling lists
J_zz = [[Jzz_0,i,i+1] for i in range(L-1)] # OBC
J_xy = [[Jxy/2.0,i,i+1] for i in range(L-1)] # OBC
h_z=[[hz,i] for i in range(L)]
# static and dynamic lists
static = [["+-",J_xy],["-+",J_xy],["zz",J_zz],["z",h_z]]
dynamic=[]
# compute the time-dependent Heisenberg Hamiltonian
H_XXZ = hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
#
##### various exact diagonalisation routines #####
# calculate entire spectrum only
E=H_XXZ.eigvalsh()
# calculate full eigensystem
E,V=H_XXZ.eigh()
# calculate minimum and maximum energy only
Emin,Emax=H_XXZ.eigsh(k=2,which="BE",maxiter=1E4,return_eigenvectors=False)
# calculate the eigenstate closest to energy E_star
E_star = 0.0
E,psi_0=H_XXZ.eigsh(k=1,sigma=E_star,maxiter=1E4)
psi_0=psi_0.reshape((-1,))