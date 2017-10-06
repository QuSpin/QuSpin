from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian # Hamiltonian and observables
from quspin.basis.photon import coherent_state # HO coherent state
import numpy as np # generic math functions
#
##### define model parameters #####
Nph_tot=60 # maximum photon occupation 
Nph=Nph_tot/2 # mean number of photons in initial coherent state
Omega=3.5 # drive frequency
A=0.8 # spin-photon coupling strength (drive amplitude)
Delta=1.0 # difference between atom energy levels
#
##### set up photon-atom Hamiltonian #####
# define operator site-coupling lists
ph_energy=[[Omega]] # photon energy
at_energy=[[Delta,0]] # atom energy
absorb=[[A/(2.0*np.sqrt(Nph)),0]] # absorption term	
emit=[[A/(2.0*np.sqrt(Nph)),0]] # emission term
# define static and dynamics lists
static=[["|n",ph_energy],["x|-",absorb],["x|+",emit],["z|",at_energy]]
dynamic=[]
# compute atom-photon basis
basis=photon_basis(spin_basis_1d,L=1,Nph=Nph_tot)
print(basis)
# compute atom-photon Hamiltonian H
H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
#
##### define initial state #####
# define atom ground state
psi_at_i=np.array([1.0,0.0]) # spin-down eigenstate of \sigma^z
# define photon coherent state with mean photon number Nph
psi_ph_i=coherent_state(np.sqrt(Nph),Nph_tot+1)
# compute atom-photon initial state as a tensor product
psi_i=np.kron(psi_at_i,psi_ph_i)