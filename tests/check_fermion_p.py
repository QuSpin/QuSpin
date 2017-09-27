from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spinless_fermion_basis_1d,photon_basis
import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
from numpy.random import random,seed

def eps(dtype):
	return 2*10.0**(-5)

L=2

dtype=np.float64

J_pp=[[+np.sqrt(2),i,(i+1)%L] for i in range(L-1)] # OBC
J_mm=[[-np.sqrt(2),i,(i+1)%L] for i in range(L-1)] # OBC

static=[["++",J_pp],["--",J_mm]]

basis=spinless_fermion_basis_1d(L=L)
H=hamiltonian(static,[],dtype=dtype,basis=basis)
Ns=H.Ns
E=H.eigvalsh()

basis1=spinless_fermion_basis_1d(L=L,pblock=1)
H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
basis2=spinless_fermion_basis_1d(L=L,pblock=-1)
H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

E1=H1.eigvalsh()
E2=H2.eigvalsh()

Ep=np.concatenate((E1,E2))
Ep.sort()

print(E)
print(Ep)

if norm(Ep-E) > Ns*eps(dtype):
	raise Exception( "test failed p symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype),norm(Ep-E)) )
