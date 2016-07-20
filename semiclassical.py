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
Nph_tot = 10 #total number of photon states
Nph = 6 #Nph_tot/2 # nuber of photons in initial state


Omega = 12.0
T = 2.0*np.pi/Omega
A = 0.8 #*Omega

hz = 1.0
hx = np.sqrt(3.0)


all_fields=[[1.0,i] for i in xrange(L)]

x_field=[[hx,i] for i in xrange(L)]
z_field=[[hz,i] for i in xrange(L)]

drive_coupling=[[A,i] for i in xrange(L)]

def f(t,Omega):
	return 2*np.cos(Omega*t)
def g(t,Omega):
	return np.exp(+1j*Omega*t)
def g_conj(t,Omega):
	return np.exp(-1j*Omega*t)

absorb=[[A/np.sqrt(Nph),i] for i in xrange(L)]
emit=[[np.conj(A)/np.sqrt(Nph),i] for i in xrange(L)]

ph_energy = [[Omega/L,i] for i in xrange(L)]


### build photon operator lists

Pstatic = [["z|",z_field], ["x|",x_field]]
Pdynamic = [["x|-",absorb,g,[Omega]], ["x|+",emit,g_conj,[Omega]]]

Pn = [["I|n",[[1.0,0]] ] ]

Pstatic_x=[["x|",all_fields]]
Pstatic_y=[["y|",all_fields]]
Pstatic_z=[["z|",all_fields]]

### build spin operator lists

static = [["z",z_field], ["x",x_field]]
dynamic = [["x",drive_coupling,f,[Omega]]]

static_x=[["x",all_fields]]
static_y=[["y",all_fields]]
static_z=[["z",all_fields]]

# build bases

Pbasis = photon_basis(spin_basis_1d,L=L,Nph=Nph_tot)
basis = spin_basis_1d(L=L)




# build photon Hamiltonian and operators
PH = hamiltonian(Pstatic,Pdynamic,L=L,dtype=np.float64,basis=Pbasis)

Pn = hamiltonian(Pn,[],L=L,dtype=np.float64,basis=Pbasis)

Psigma_x=hamiltonian(Pstatic_x,[],L=L,dtype=np.float64,basis=Pbasis)
Psigma_y=hamiltonian(Pstatic_y,[],L=L,dtype=np.complex128,basis=Pbasis)
Psigma_z=hamiltonian(Pstatic_z,[],L=L,dtype=np.float64,basis=Pbasis)

# build spin Hamiltonian and operators
H_driven=hamiltonian(static,dynamic,L=L,dtype=np.float64,basis=basis)
H=hamiltonian(static,[],L=L,dtype=np.float64,basis=basis)

sigma_x=hamiltonian(static_x,[],L=L,dtype=np.float64,basis=basis)
sigma_y=hamiltonian(static_y,[],L=L,dtype=np.complex128,basis=basis)
sigma_z=hamiltonian(static_z,[],L=L,dtype=np.float64,basis=basis)

### diagonalise Hamiltonian
E, V = H.eigh()
### define initial state
psi0 = V[:,0]

def coherent_state(a,n,dtype=np.float64):
	s1 = np.full((n,),-np.abs(a)**2/2.0,dtype=dtype)
	s2 = np.arange(n,dtype=np.float64)
	s3 = np.array(s2)
	s3[0] = 1
	np.log(s3,out=s3)
	s3[1:] = 0.5*np.cumsum(s3[1:])
	state = s1+np.log(a)*s2-s3
	return np.exp(state)
psi0_ph = coherent_state(np.sqrt(Nph),Nph_tot+1)

print np.linalg.norm(psi0_ph)
psi0_ph *= 1.0/np.linalg.norm(psi0_ph)



Ppsi0 = np.kron(psi0,psi0_ph)

# time-evolve
t = np.linspace(0.0,5.0,201)

Ppsi = PH.evolve(Ppsi0,0.0,t,rtol=1E-12,atol=1E-12)
psi = H_driven.evolve(psi0,0.0,t)

"""
Ppsi_rot_V = np.zeros(Ppsi_rot.shape, dtype=np.complex128)
for i in xrange(len(t)):
	Ppsi_rot_V[i,:] = np.exp(-1j*Omega*t[i]*Pn.tocsr().diagonal() )*Ppsi_rot[i,:]
"""



POz = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),Psigma_z.todense(),Ppsi) )
Oz = np.real( np.einsum("ij,jk,ik->i", psi.conj(),sigma_z.todense(),psi) )

POx = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),Psigma_x.todense(),Ppsi) )
Ox = np.real( np.einsum("ij,jk,ik->i", psi.conj(),sigma_x.todense(),psi) )

POy = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),Psigma_y.todense(),Ppsi) )
Oy = np.real( np.einsum("ij,jk,ik->i", psi.conj(),sigma_y.todense(),psi) )

n = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),Pn.todense(),Ppsi) )

# calculate energy of spin chain
Energy = np.real( np.einsum("ij,jk,ik->i", psi.conj(),H.todense(),psi) )

PEnergy = hx*POx + hz*POz 


# plot results

str1 = "quantum"
str2 = "semi-classical"
str3 = "\\langle n\\rangle"

title_params = tuple(np.around([hz,hx,A,Omega],2) ) + (Nph,Nph_tot+1)
titlestr = "$h_z=%s,\\ h_x=%s,\\ A=%s,\\ \\Omega=%s,\\ N_\\mathrm{ph}/N_\\mathrm{modes}=%s/%s$" %(title_params)

#'''
plt.plot(t/T,POz,'b',linewidth=1,label=str1)
plt.plot(t/T,Oz,'r-',linewidth=1,label=str2)
#'''
#plt.plot(t/T,POz-Oz,'k',linewidth=1,label=str1)

#plt.plot(t/T,n/Nph,'k',linewidth=1,label=str3)

#plt.plot(t/T,Energy,'k',linewidth=1,label=str3)
#plt.plot(t/T,PEnergy,'m',linewidth=1,label=str3)

plt.xlabel('$t/T$', fontsize=18)
plt.ylabel('$\\eta(t)$', fontsize=20)


plt.legend(loc='upper right')
plt.title(titlestr, fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(True)


plt.show()


			