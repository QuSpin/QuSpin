from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
try:
	from itertools import izip as zip
except ImportError:
	pass

L = 10
dtype = np.float64
T=0.5
n=10

tol = 1e-6

def drive(t):
	return t*np.cos(2*np.pi*t/T)

basis = spin_basis_1d(L,Nup=L//2,kblock=0,a=2)
times = np.arange(0,(n+0.1)*T,T/4)


J1=[[1.0,i,(i+1)%L] for i in range(L)]
J2=[[(-1)**i,i] for i in range(L)]
Sz = [[1.0/L,i] for i in range(L)]

static = [
			["xx",J1],
			["yy",J1],
			["zz",J1],
			]

dynamic = [
			["z",J2,drive,()],
			]

H_0 = hamiltonian(static,[],basis=basis,dtype=dtype)
V_t = hamiltonian([],dynamic,basis=basis,dtype=dtype)
O_0 = H_0.todense()


E,psi_0 = H_0.eigsh(k=1,which="SA",maxiter=10000)

H = H_0 + V_t



psi_0 = psi_0.ravel()
rho_0 = np.outer(psi_0.conj(),psi_0).astype(np.complex128)


psi_t = H.evolve(psi_0,0,times,eom="SE",iterate=True,atol=1e-10,rtol=1e-10)
O_expt = obs_vs_time(psi_t,times,dict(O=O_0))["O"]

rho_t = H.evolve(rho_0,0,times,eom="LvNE",iterate=False,atol=1e-10,rtol=1e-10)
O_expt_2 = np.einsum("ij...,ji->...",rho_t,O_0).real
np.testing.assert_allclose(O_expt,O_expt_2)


rho_t = H.evolve(rho_0,0,times,eom="LvNE",iterate=True,atol=1e-10,rtol=1e-10)
for t,O_SE,rho in zip(times,O_expt,rho_t):
	O_LvNE = np.einsum("ij,ji->",rho,O_0).real
	if np.abs(O_SE-O_LvNE) > tol:
		raise Exception("test failed by 'E_LvNE' at t={}, diff of {}".format(t,np.abs(E_SE-E_LvNE)))


print("evolve checks passed")