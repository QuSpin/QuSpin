from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis
import numpy as np
import scipy.sparse as sm
from numpy.linalg import norm
from numpy.random import random,seed


seed()

L=3

Jpm=   [[+1j/2.0,i,(i+1)%L] for i in xrange(L)]
Jpm_cc=[[-1j/2.0,i,(i+1)%L] for i in xrange(L)]

static=[["+-",Jpm],["-+",Jpm_cc]]

H=hamiltonian(static,[],L=L,dtype=np.complex64,pauli=False,zblock=1)

print np.round( (H.todense()) ,4)