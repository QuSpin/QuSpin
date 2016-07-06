from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis
import numpy as np
import scipy.sparse as sm
from numpy.linalg import norm
from numpy.random import random,seed

seed()


L = 6
Ntot = 4


Omega = 10.0
A = 0.5

J = 1.0
hz = 0.9045
hx = 0.809

def f(t,Omega):
	return np.exp(1j*Omega*t)
drive_args = [Omega]


x_field=[[hx,i] for i in xrange(L)]
z_field=[[hz,i] for i in xrange(L)]
J_nn=[[J,i,(i+1)%L] for i in xrange(L)]

sp=[[A,i] for i in xrange(L)]
sm=[[np.conj(A),i] for i in xrange(L)]


static = [["zz|I",J_nn], ["z|I",z_field], ["x|I",x_field], ["+|-",sp], ["-|+",sm]]
dynamic = []

basis = photon_basis(spin_basis_1d,L=L,Ntot=Ntot)

H=hamiltonian(static,dynamic,L=L,dtype=np.complex64,basis=basis)

E = H.eigvalsh()

print E

			