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
Nph_tot = 1000 #total number of photon states
Nph = Nph_tot/2 # nuber of photons in initial state


Omega = 9.0


ph_energy = [[Omega/L,i] for i in xrange(L)]


### build lists
Pstatic = [["I|n",ph_energy]]

###
Pbasis = photon_basis(spin_basis_1d,L=L,Nph=Nph_tot)

###
PH =hamiltonian(Pstatic,[],L=L,dtype=np.float64,basis=Pbasis)


### define initial state
psi0 = np.array([1.0,0.0],dtype=np.complex128) #V[:,0]
psi0_ph = np.zeros((Nph_tot+1,))
psi0_ph[Nph] = 1.0
Ppsi0 = np.kron(psi0,psi0_ph)


# time-evolve
t = np.linspace(0.0,5.0,201)

Ppsi = PH.evolve(Ppsi0,0.0,t,rtol=1E-12,atol=1E-12)



norm = np.real( np.einsum("ij,ij->i", Ppsi.conj(),Ppsi) )
E_tot = np.real( np.einsum("ij,jk,ik->i", Ppsi.conj(),PH.todense(),Ppsi) )


'''
E_tot2 = np.zeros(t.shape,dtype=np.float64)
for i in xrange(len(t)):
	E_tot2[i] = reduce(np.dot, [Ppsi.conj()[i,:],PH.todense(),Ppsi[i,:] ])[0,0]

print np.linalg.norm(E_tot2 - E_tot)
'''

plt.plot(t,(E_tot-E_tot[0]),'k--',linewidth=1)
plt.plot(t,(norm-norm[0]),'r',linewidth=1)

plt.legend(loc='upper right')
plt.grid(True)

plt.show()

