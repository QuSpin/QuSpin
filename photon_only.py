from exact_diag_py.tools import observables

from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis
import numpy as np
import scipy as sp
import scipy.sparse as sm
from numpy.linalg import norm
from numpy.random import random,seed

import matplotlib.pyplot as plt
import pylab

import time
import sys
import os

seed()


L = 1
Nph_tot = 100 #total number of photon states
Nph = Nph_tot/2 # nuber of photons in initial state


Omega = 9.0
T = 2.0*np.pi/Omega
A = 0.8

hz = 1.0
hx = np.sqrt(3.0)


all_fields=[[1.0,i] for i in xrange(L)]

x_field=[[hx,i] for i in xrange(L)]
z_field=[[hz,i] for i in xrange(L)]

drive_coupling=[[A,i] for i in xrange(L)]

def f(t,Omega):
	return 2.0*np.cos(Omega*t)
f_args = [Omega]
def g(t,Omega):
	return np.exp(1j*Omega*t)
def g_conj(t,Omega):
	return np.exp(-1j*Omega*t)

absorb=[[A/np.sqrt(Nph),i] for i in xrange(L)]
emit=[[np.conj(A)/np.sqrt(Nph),i] for i in xrange(L)]

ph_energy = [[Omega/L,i] for i in xrange(L)]


### build lists

#Pstatic = [["I|n",ph_energy], ["+|-",absorb], ["-|+",emit], ["z|",z_field], ["x|",x_field]]
Pstatic = [["I|n",ph_energy], ["x|-",absorb], ["x|+",emit], ["z|",z_field], ["x|",x_field]]

Pstatic_coupl = [["x|-",absorb], ["x|+",emit]]

Pstatic_rot = [["z|",z_field], ["x|",x_field]]
#Pdynamic = [["+|-",absorb,g,f_args], ["-|+",emit,g_conj,f_args]]
Pdynamic = [["x|-",absorb,g,f_args], ["x|+",emit,g_conj,f_args]]

Pstatic_n = [["I|n",[[1.0,0]] ] ]
Pstatic_d = [["I|-",[[1.0,0]] ] ]
Pstatic_c = [["I|+",[[1.0,0]] ] ]

Pstatic_x=[["x|",all_fields]]
Pstatic_y=[["y|",all_fields]]
Pstatic_z=[["z|",all_fields]]

###
Pbasis = photon_basis(spin_basis_1d,L=L,Nph=Nph_tot)

###
PH =hamiltonian(Pstatic,[],L=L,dtype=np.float64,basis=Pbasis)
PH_rot =hamiltonian(Pstatic_rot,Pdynamic,L=L,dtype=np.float64,basis=Pbasis)

PH_coupl=hamiltonian(Pstatic_coupl,[],L=L,dtype=np.float64,basis=Pbasis)
###
Pn = hamiltonian(Pstatic_n,[],L=L,dtype=np.float64,basis=Pbasis)
Pd = hamiltonian(Pstatic_d,[],L=L,dtype=np.float64,basis=Pbasis,check_herm=False)
Pc = hamiltonian(Pstatic_c,[],L=L,dtype=np.float64,basis=Pbasis,check_herm=False)

Psigma_x=hamiltonian(Pstatic_x,[],L=L,dtype=np.float64,basis=Pbasis)
Psigma_y=hamiltonian(Pstatic_y,[],L=L,dtype=np.complex128,basis=Pbasis)
Psigma_z=hamiltonian(Pstatic_z,[],L=L,dtype=np.float64,basis=Pbasis)


########
static = [["z",z_field], ["x",x_field]]
basis = spin_basis_1d(L=L)
H=hamiltonian(static,[],L=L,dtype=np.float64,basis=basis)
### diagonalise Hamiltonian
E, V = H.eigh()
### define initial state
psi0 = V[:,0]
psi0_ph = np.zeros((Nph_tot+1,))
psi0_ph[Nph] = 1.0
Ppsi0 = np.kron(psi0,psi0_ph)

# time-evolve
t = np.linspace(0.0,5.0,2001)

Ppsi = PH.evolve(Ppsi0,0.0,t)
Ppsi_rot = PH_rot.evolve(Ppsi0,0.0,t)

Ppsi_rot_V = np.zeros(Ppsi_rot.shape, dtype=np.complex128)
for i in xrange(len(t)):
	Ppsi_rot_V[i,:] = np.exp(-1j*Omega*t[i]*Pn.tocsr().diagonal() )*Ppsi_rot[i,:]



POz = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),Psigma_z.todense(),Ppsi) )
POz_rot = np.real( np.einsum("ij,jk,ik->i", Ppsi_rot.conj(),Psigma_z.todense(),Ppsi_rot) )

POx = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),Psigma_x.todense(),Ppsi) )
POx_rot = np.real( np.einsum("ij,jk,ik->i", Ppsi_rot.conj(),Psigma_x.todense(),Ppsi_rot) )

POy = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),Psigma_y.todense(),Ppsi) )
POy_rot = np.real( np.einsum("ij,jk,ik->i", Ppsi_rot.conj(),Psigma_y.todense(),Ppsi_rot) )

POn = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),Pn.todense(),Ppsi) )
POn_rot = np.real( np.einsum("ij,jk,ik->i", Ppsi_rot.conj(),Pn.todense(),Ppsi_rot) )

POd = np.einsum("ij,jk,ik->i", Ppsi.conj(),Pd.todense(),Ppsi) 
POd_rot = np.einsum("ij,jk,ik->i", Ppsi_rot.conj(),Pd.todense(),Ppsi_rot)
POd_rot_V = np.einsum("ij,jk,ik->i", Ppsi_rot_V.conj(),Pd.todense(),Ppsi_rot_V) 

POc = np.einsum("ij,jk,ik->i", Ppsi.conj(),Pc.todense(),Ppsi) 
POc_rot = np.einsum("ij,jk,ik->i", Ppsi_rot.conj(),Pc.todense(),Ppsi_rot) 
POc_rot_V = np.einsum("ij,jk,ik->i", Ppsi_rot_V.conj(),Pc.todense(),Ppsi_rot_V)

# observables
E_ph = Omega*POn
E_sp = hx*POx + hz*POz
E_ph_sp = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),PH_coupl.todense(),Ppsi) )
E_tot = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),PH.todense(),Ppsi) )


E_tot_rot = np.real( np.einsum("ij,jk,ik->i", Ppsi_rot_V.conj(),PH_rot.todense(),Ppsi_rot_V) )
E_ph_sp_rot = np.real( np.einsum("ij,jk,ik->i", Ppsi_rot_V.conj(),PH_coupl.todense(),Ppsi_rot_V) )

#####################
str1 = "lab"
str2 = "rot"
str3 = "lab-rot"

#plt.plot(t/T,POn,'b',linewidth=1,label=str1)
#plt.plot(t/T,POn_rot,'r--',linewidth=1,label=str2)
#plt.plot(t/T,POn-POn_rot,'b',linewidth=1,label=str3)

#plt.plot(t/T,POd,'b',linewidth=1,label=str1)
#plt.plot(t/T,POd_rot,'r--',linewidth=1,label=str2)
#plt.plot(t/T,POd_rot_V,'g.',linewidth=1,label=str2)

#plt.plot(t/T,np.imag(POc-POc_rot),'b',linewidth=1,label=str1)

plt.plot(t/T,(E_tot-E_tot[0]),'k--',linewidth=2,label=str1)
plt.plot(t/T,(E_ph-E_ph[0]),'b',linewidth=1,label=str1)
plt.plot(t/T,E_sp-E_sp[0],'r',linewidth=1,label=str1)
plt.plot(t/T,E_ph_sp-E_ph_sp[0],'m',linewidth=1,label=str1)
#plt.plot(t/T,E_ph_sp + E_sp - (E_ph_sp + E_sp)[0] ,'g',linewidth=1,label=str1)
plt.plot(t/T,E_ph_sp + E_sp + E_ph - (E_ph_sp + E_sp + E_ph)[0] ,'g',linewidth=1,label=str1)

#plt.plot(t/T,E_ph + E_ph_sp + E_sp - E_tot,'c',linewidth=2,label=str1)
#plt.plot(t/T,,'r--',linewidth=2,label=str1)

#plt.plot(t/T,E_ph_sp_rot,'g',linewidth=1,label=str1)
#plt.plot(t/T,E_sp+E_ph,'g',linewidth=1,label=str1)
#plt.plot(t/T,E_tot,'m',linewidth=1,label=str1)
#plt.plot(t/T,E_tot_rot,'c',linewidth=1,label=str1)

plt.legend(loc='upper right')
plt.grid(True)

plt.show()

