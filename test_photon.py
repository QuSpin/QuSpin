from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis
import numpy as np
import scipy as sp
import scipy.sparse as sm
from numpy.linalg import norm
from numpy.random import random,seed

seed()


L = 8
Ntot = L


Omega = 10.0
T = 2*np.pi/Omega
A = 0.5

J = 1.0
hz = 0.9045
hx = 0.809

def f(t,Omega):
	return np.exp(1j*Omega*t)
def f_conj(t,Omega):
	return np.exp(-1j*Omega*t)
drive_args = [Omega]


x_field=[[hx,i] for i in xrange(L)]
z_field=[[hz,i] for i in xrange(L)]
J_nn=[[J,i,(i+1)%L] for i in xrange(L)]

absorb=[[A,i,0] for i in xrange(L)]
emit=[[np.conj(A),i,0] for i in xrange(L)]

ph_energy = [[Omega,0]]


static_sp_ph = [["zz|",J_nn], ["z|",z_field], ["x|",x_field], ["+|-",absorb], ["-|+",emit], ["|n",ph_energy]]
dynamic_sp_ph = []

static_sp = [["zz",J_nn], ["z",z_field], ["x",x_field]]
dynamic_sp = [["+",absorb,f,drive_args], ["-",emit,f_conj,drive_args]]

basis_sp_ph = photon_basis(spin_basis_1d,L=L,Ntot=Ntot,kblock=0,pblock=1)
basis_sp = spin_basis_1d(L=L,kblock=0,pblock=1)

H_sp_ph=hamiltonian(static_sp_ph,dynamic_sp_ph,L=L,dtype=np.float64,basis=basis_sp_ph)
H_sp=hamiltonian(static_sp,dynamic_sp_ph,L=L,dtype=np.float64,basis=basis_sp)

print "spin-photon H-space size is {}".format(H_sp_ph.Ns)
print "spin H-space size is {}".format(H_sp.Ns)

E_sp_ph = H_sp_ph.eigvalsh()
E_sp = H_sp.eigvalsh()


# calculate Floquet Hamiltonian
identity = np.ones(H_sp._shape, dtype=np.float64)
U_F = np.zeros(H_sp._shape, dtype=np.complex128)
for i in xrange(H_sp.Ns):
	solver = sp.integrate.complex_ode(H_sp._hamiltonian__SO)
	print solver
	U_F[:,i] = 1




print E_sp 
print E_sp_ph

			