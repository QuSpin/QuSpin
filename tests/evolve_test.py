from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
try:
	from itertools import izip as zip
except ImportError:
	pass

L = 8
dtype = np.float64
T=1

tol = 1e-6

def drive(t):
	return np.cos(2*np.pi*t/T)
	#return np.sin(t**2)

basis = spin_basis_1d(L,Nup=L//2,pzblock=1,kblock=0,a=2)
print(basis.Ns)

times = np.arange(0,6,0.01)


J1=[[1.0,i,(i+1)%L] for i in range(L)]
J2=[[(-1)**i,i] for i in range(L)]

static = [
			["xx",J1],
			["yy",J1],
			["zz",J1],
			]

dynamic = [
			["z",J2,drive,()],
			]

H_0 = hamiltonian(static,[],basis=basis,dtype=dtype)
V = hamiltonian([],dynamic,basis=basis,dtype=dtype)
O = hamiltonian([["z",J2]],[],basis=basis,dtype=dtype).todense()


E,psi_0 = H_0.eigsh(k=1,which="SA",maxiter=10000)

H = H_0 + V



psi_0 = psi_0.ravel()
rho_0 = np.outer(psi_0.conj(),psi_0)


psi_t = H.evolve(psi_0,0,times,iterate=False,atol=1e-10,rtol=1e-10)
rho_t = H.evolve(rho_0,0,times,eom="LvNE",iterate=False,atol=1e-10,rtol=1e-10)
O_t   = H.evolve(O,0,times,eom="HE",iterate=False,atol=1e-10,rtol=1e-10)

for i,t,psi,rho,o in zip(range(len(times)),times,psi_t,rho_t,O_t):
	E_SE = np.einsum("i,ij,j->",psi.conj(),O,psi).real
	E_HE = np.einsum("i,ij,j->",psi_0,o,psi_0).real
	E_LvNE = np.einsum("ij,ji->",O,rho).real

	#if np.abs(E_HE-E_SE) > tol:
	#	raise Exception("test failed by 'HE' at t={}".format(t))
	if np.abs(E_SE-E_LvNE) > tol:
		raise Exception("test failed by 'E_LvNE' at t={}".format(t))


	

psi_t = H.evolve(psi_0,0,times,iterate=True,atol=1e-10,rtol=1e-10)
rho_t = H.evolve(rho_0,0,times,eom="LvNE",iterate=True,atol=1e-10,rtol=1e-10)
O_t = H.evolve(O,0,times,eom="HE",iterate=True,atol=1e-10,rtol=1e-10)

for i,t,psi,rho,o in zip(range(len(times)),times,psi_t,rho_t,O_t):
	E_SE = np.einsum("i,ij,j->",psi.conj(),O,psi).real
	E_HE = np.einsum("i,ij,j->",psi_0,o,psi_0).real
	E_LvNE = np.einsum("ij,ji->",O,rho).real
	#if np.abs(E_HE-E_SE) > tol:
	#	raise Exception("test failed by 'HE' at t={}".format(t))
	if np.abs(E_SE-E_LvNE) > tol:
		raise Exception("test failed by 'E_LvNE' at t={}".format(t))
