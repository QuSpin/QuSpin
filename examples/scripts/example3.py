from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
########################################################################################
#                                   example 3                                          #
#    In this example we show how to use the photon_basis class to study spin chains    #
#    coupled to a single photon mode. To demonstrate this we simulate a single spin    #
#    and show how the semi-classical limit emerges in the limit that the number of     #
#    photons goes to infinity.                                                         #
########################################################################################
from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian # Hamiltonian and observables
from quspin.tools.measurements import obs_vs_time # t_dep measurements
from quspin.tools.Floquet import Floquet,Floquet_t_vec # Floquet Hamiltonian
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
# compute atom-photon Hamiltonian H
H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
#
##### set up semi-classical Hamiltonian #####
# define operators
dipole_op=[[A,0]]
# define periodic drive and its parameters
def drive(t,Omega):
	return np.cos(Omega*t)
drive_args=[Omega]
# define semi-classical static and dynamic lists
static_sc=[["z",at_energy]]
dynamic_sc=[["x",dipole_op,drive,drive_args]]
# compute semi-classical basis
basis_sc=spin_basis_1d(L=1)
# compute semi-classical Hamiltonian H_{sc}(t)
H_sc=hamiltonian(static_sc,dynamic_sc,dtype=np.float64,basis=basis_sc)
#
##### define initial state #####
# define atom ground state
#psi_at_i=np.array([1.0,0.0]) # spin-down eigenstate of \sigma^z in QuSpin 0.2.3 or older
psi_at_i=np.array([0.0,1.0])  # spin-down eigenstate of \sigma^z in QuSpin 0.2.6 or newer
# define photon coherent state with mean photon number Nph
psi_ph_i=coherent_state(np.sqrt(Nph),Nph_tot+1)
# compute atom-photon initial state as a tensor product
psi_i=np.kron(psi_at_i,psi_ph_i)
#
##### calculate time evolution #####
# define time vector over 30 driving cycles with 100 points per period
t=Floquet_t_vec(Omega,30) # t.i = initial time, t.T = driving period
# evolve atom-photon state with Hamiltonian H
psi_t=H.evolve(psi_i,t.i,t.vals,iterate=True,rtol=1E-9,atol=1E-9) 
# evolve atom GS with semi-classical Hamiltonian H_sc
psi_sc_t=H_sc.evolve(psi_at_i,t.i,t.vals,iterate=True,rtol=1E-9,atol=1E-9)
#
##### define observables #####
# define observables parameters
obs_args={"basis":basis,"check_herm":False,"check_symm":False}
obs_args_sc={"basis":basis_sc,"check_herm":False,"check_symm":False}
# in atom-photon Hilbert space
n=hamiltonian([["|n", [[1.0  ]] ]],[],dtype=np.float64,**obs_args)
sz=hamiltonian([["z|",[[1.0,0]] ]],[],dtype=np.float64,**obs_args)
sy=hamiltonian([["y|",	[[1.0,0]] ]],[],dtype=np.complex128,**obs_args)
# in the semi-classical Hilbert space
sz_sc=hamiltonian([["z",[[1.0,0]] ]],[],dtype=np.float64,**obs_args_sc)
sy_sc=hamiltonian([["y",[[1.0,0]] ]],[],dtype=np.complex128,**obs_args_sc)
#
##### calculate expectation values #####
# in atom-photon Hilbert space
Obs_t = obs_vs_time(psi_t,t.vals,{"n":n,"sz":sz,"sy":sy})
O_n, O_sz, O_sy = Obs_t["n"], Obs_t["sz"], Obs_t["sy"]
# in the semi-classical Hilbert space
Obs_sc_t = obs_vs_time(psi_sc_t,t.vals,{"sz_sc":sz_sc,"sy_sc":sy_sc})
O_sz_sc, O_sy_sc = Obs_sc_t["sz_sc"], Obs_sc_t["sy_sc"]
##### plot results #####
import matplotlib.pyplot as plt
import pylab
# define legend labels
str_n = "$\\langle n\\rangle,$"
str_z = "$\\langle\\sigma^z\\rangle,$"
str_x = "$\\langle\\sigma^x\\rangle,$"
str_z_sc = "$\\langle\\sigma^z\\rangle_\\mathrm{sc},$"
str_x_sc = "$\\langle\\sigma^x\\rangle_\\mathrm{sc}$"
# plot spin-photon data
fig = plt.figure()
plt.plot(t.vals/t.T,O_n/Nph,"k",linewidth=1,label=str_n)
plt.plot(t.vals/t.T,O_sz,"c",linewidth=1,label=str_z)
plt.plot(t.vals/t.T,O_sy,"tan",linewidth=1,label=str_x)
# plot semi-classical data
plt.plot(t.vals/t.T,O_sz_sc,"b.",marker=".",markersize=1.8,label=str_z_sc)
plt.plot(t.vals/t.T,O_sy_sc,"r.",marker=".",markersize=2.0,label=str_x_sc)
# label axes
plt.xlabel("$t/T$",fontsize=18)
# set y axis limits
plt.ylim([-1.1,1.4])
# display legend horizontally
plt.legend(loc="upper right",ncol=5,columnspacing=0.6,numpoints=4)
# update axis font size
plt.tick_params(labelsize=16)
# turn on grid
plt.grid(True)
# save figure
plt.tight_layout()
plt.savefig('example3.pdf', bbox_inches='tight')
# show plot
#plt.show()
plt.close()