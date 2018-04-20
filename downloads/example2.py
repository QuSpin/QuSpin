from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
##################################################################################
#                            example 1                                           #
#     In this example we show how to use some of QuSpin's tools for studying     #
#     Floquet systems by analysing the heating in a periodically driven          #
#     spin chain. We also show how to construct more complicated multi-spin      #
#     interactions using QuSpin's interface.                                     #
##################################################################################
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import obs_vs_time, diag_ensemble # t_dep measurements
from quspin.tools.Floquet import Floquet, Floquet_t_vec # Floquet Hamiltonian
import numpy as np # generic math functions
#
##### define model parameters #####
L=14 # system size
J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field
Omega=4.5 # drive frequency
#
##### set up alternating Hamiltonians #####
# define time-reversal symmetric periodic step drive
def drive(t,Omega):
	return np.sign(np.cos(Omega*t))
drive_args=[Omega]
# compute basis in the 0-total momentum and +1-parity sector
basis=spin_basis_1d(L=L,a=1,kblock=0,pblock=1)
# define PBC site-coupling lists for operators
x_field_pos=[[+g,i]	for i in range(L)]
x_field_neg=[[-g,i]	for i in range(L)]
z_field=[[h,i]		for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
# static and dynamic lists
static=[["zz",J_nn],["z",z_field],["x",x_field_pos]]
dynamic=[["zz",J_nn,drive,drive_args],
		 ["z",z_field,drive,drive_args],["x",x_field_neg,drive,drive_args]]
# compute Hamiltonians
H=0.5*hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
#
##### set up second-order van Vleck Floquet Hamiltonian #####
# zeroth-order term
Heff_0=0.5*hamiltonian(static,[],dtype=np.float64,basis=basis)
# second-order term: site-coupling lists
Heff2_term_1=[[+J**2*g,i,(i+1)%L,(i+2)%L] for i in range(L)] # PBC
Heff2_term_2=[[+J*g*h, i,(i+1)%L] for i in range(L)] # PBC
Heff2_term_3=[[-J*g**2,i,(i+1)%L] for i in range(L)] # PBC
Heff2_term_4=[[+J**2*g+0.5*h**2*g,i] for i in range(L)]
Heff2_term_5=[[0.5*h*g**2,		  i] for i in range(L)]
# define static list
Heff_static=[["zxz",Heff2_term_1],
			 ["xz",Heff2_term_2],["zx",Heff2_term_2],
			 ["yy",Heff2_term_3],["zz",Heff2_term_2],
			 ["x",Heff2_term_4],
			 ["z",Heff2_term_5]							] 
# compute van Vleck Hamiltonian
Heff_2=hamiltonian(Heff_static,[],dtype=np.float64,basis=basis)
Heff_2*=-np.pi**2/(12.0*Omega**2)
# zeroth + second order van Vleck Floquet Hamiltonian
Heff_02=Heff_0+Heff_2
#
##### set up second-order van Vleck Kick operator #####
Keff2_term_1=[[J*g,i,(i+1)%L] for i in range(L)] # PBC
Keff2_term_2=[[h*g,i] for i in range(L)]
# define static list
Keff_static=[["zy",Keff2_term_1],["yz",Keff2_term_1],["y",Keff2_term_2]]
Keff_02=hamiltonian(Keff_static,[],dtype=np.complex128,basis=basis)
Keff_02*=np.pi**2/(8.0*Omega**2)
#
##### rotate Heff to stroboscopic basis #####
# e^{-1j*Keff_02} Heff_02 e^{+1j*Keff_02}
HF_02 = Heff_02.rotate_by(Keff_02,generator=True,a=1j) 
#
##### define time vector of stroboscopic times with 100 cycles #####
t=Floquet_t_vec(Omega,100,len_T=1) # t.vals=times, t.i=init. time, t.T=drive period
#
##### calculate exact Floquet eigensystem #####
t_list=np.array([0.0,t.T/4.0,3.0*t.T/4.0])+np.finfo(float).eps # times to evaluate H
dt_list=np.array([t.T/4.0,t.T/2.0,t.T/4.0]) # time step durations to apply H for
Floq=Floquet({'H':H,'t_list':t_list,'dt_list':dt_list},VF=True) # call Floquet class
VF=Floq.VF # read off Floquet states
EF=Floq.EF # read off quasienergies
#
##### calculate initial state (GS of HF_02) and its energy
EF_02, psi_i = HF_02.eigsh(k=1,which="SA",maxiter=1E4)
psi_i = psi_i.reshape((-1,))
#
##### time-dependent measurements
# calculate measurements
Sent_args = {"basis":basis,"chain_subsys":[j for j in range(L//2)]}
#meas = obs_vs_time((psi_i,EF,VF),t.vals,{"E_time":HF_02/L},Sent_args=Sent_args)
#"""
# alternative way by solving Schroedinger's eqn
psi_t = H.evolve(psi_i,t.i,t.vals,iterate=True,rtol=1E-9,atol=1E-9)
meas = obs_vs_time(psi_t,t.vals,{"E_time":HF_02/L},Sent_args=Sent_args)
#"""
# read off measurements
Energy_t = meas["E_time"]
Entropy_t = meas["Sent_time"]["Sent"]
#
##### calculate diagonal ensemble measurements
DE_args = {"Obs":HF_02,"Sd_Renyi":True,"Srdm_Renyi":True,"Srdm_args":Sent_args}
DE = diag_ensemble(L,psi_i,EF,VF,**DE_args)
Ed = DE["Obs_pure"]
Sd = DE["Sd_pure"]
Srdm=DE["Srdm_pure"]
#
##### plot results #####
import matplotlib.pyplot as plt
import pylab
# define legend labels
str_E_t = "$\\mathcal{E}(lT)$"
str_Sent_t = "$s_\mathrm{ent}(lT)$"
str_Ed = "$\\overline{\mathcal{E}}$"
str_Srdm = "$\\overline{s}_\mathrm{rdm}$"
str_Sd = "$s_d^F$"
# plot infinite-time data
fig = plt.figure()
plt.plot(t.vals/t.T,Ed*np.ones(t.vals.shape),"b--",linewidth=1,label=str_Ed)
plt.plot(t.vals/t.T,Srdm*np.ones(t.vals.shape),"r--",linewidth=1,label=str_Srdm)
plt.plot(t.vals/t.T,Sd*np.ones(t.vals.shape),"g--",linewidth=1,label=str_Sd)
# plot time-dependent data
plt.plot(t.vals/t.T,Energy_t,"b-o",linewidth=1,label=str_E_t,markersize=3.0)
plt.plot(t.vals/t.T,Entropy_t,"r-s",linewidth=1,label=str_Sent_t,markersize=3.0)
# label axes
plt.xlabel("$\\#\ \\mathrm{periods}\\ l$",fontsize=18)
# set y axis limits
plt.ylim([-0.6,0.7])
# display legend
plt.legend(loc="lower right",ncol=2,fontsize=18)
# update axis font size
plt.tick_params(labelsize=16)
# turn on grid
plt.grid(True)
# save figure
plt.tight_layout()
plt.savefig('example2.pdf', bbox_inches='tight')
# show plot
#plt.show() 
plt.close()