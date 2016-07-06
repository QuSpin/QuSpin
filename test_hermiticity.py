from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis
import numpy as np
import scipy.sparse as sm
from numpy.linalg import norm
from numpy.random import random,seed

seed()



L=4

Jpm=   [[+1/2.0,i%L,(i+1)%L] for i in xrange(L)]
Jpm_cc=[[+1/2.0,i%L,(i+1)%L] for i in xrange(L)]

static=[["+-",Jpm],["-+",Jpm_cc],["+-",Jpm],["-+",Jpm_cc]]


b = photon_basis(spin_basis_1d,L,Ntot=L)
#print b
H=hamiltonian(static,[],L=L,dtype=np.complex64,pauli=False,pblock=1,zblock=1,kblock=0)

print np.round( np.linalg.norm(H.todense()-H.todense().T) ,4)





