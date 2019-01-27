from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_1d,spin_basis_general
from quspin.operators import hamiltonian,quantum_operator,quantum_LinearOperator
from itertools import product
import cProfile,os
import numpy as np
import scipy.sparse as sp



dtypes = [np.float32,np.float64,np.complex64,np.complex128]

def eps(dtype1,dtype2):
	return 100*max(np.finfo(dtype1).eps,np.finfo(dtype2).eps)


Lx = 2
Ly = 4
N = Lx*Ly
m = None

i = np.arange(N)
x = i%Lx
y = i//Lx
tx = (x+1)%Lx + y*Lx
ty = x + ((y+1)%Ly)*Lx
px = x[::-1] + y*Lx
py = x + y[::-1]*Lx
z = -(1+i)


basis_full = spin_basis_general(N)
basis_pcon = spin_basis_general(N,m=0.0)
basis_pcon_symm = spin_basis_general(N,m=0.0,tx=(tx,0),ty=(ty,0),px=(px,0),py=(py,0),z=(z,0))

Jzz_list = [[1.0,i,tx[i]] for i in range(N)]+[[1.0,i,ty[i]] for i in range(N)]
Jxy_list = [[0.5,i,tx[i]] for i in range(N)]+[[0.5,i,ty[i]] for i in range(N)]
static = [["+-",Jxy_list],["-+",Jxy_list],["zz",Jzz_list]]


for b in [basis_full,basis_pcon,basis_pcon_symm]:
	for dtype1,dtype2 in product(dtypes,dtypes):
		H = hamiltonian(static,[],basis=b,dtype=dtype1)
		H_op = quantum_LinearOperator(static,basis=b,dtype=dtype1)
		
		for i in range(10):
			v = np.random.uniform(-1,1,size=(b.Ns,)) + 1j*np.random.uniform(-1,1,size=(b.Ns,))
			v /= np.linalg.norm(v)


			v1 = H.dot(v)
			v2 = H_op.dot(v)


			atol = eps(dtype1,dtype2)
			np.testing.assert_allclose(v1,v2,atol=atol)

			v = np.random.uniform(-1,1,size=(b.Ns,10)) + 1j*np.random.uniform(-1,1,size=(b.Ns,10))
			v /= np.linalg.norm(v)


			v1 = H.dot(v)
			v2 = H_op.dot(v)


			atol = eps(dtype1,dtype2)
			np.testing.assert_allclose(v1,v2,atol=atol)



basis_full = spin_basis_1d(N)
basis_pcon = spin_basis_1d(N,m=0.0)
basis_pcon_symm = spin_basis_1d(N,m=0.0,kblock=0,pblock=1,zblock=1)

Jzz_list = [[1.0,i,(i+1)%N] for i in range(N)]
Jxy_list = [[0.5,i,(i+1)%N] for i in range(N)]
static = [["+-",Jxy_list],["-+",Jxy_list],["zz",Jzz_list]]


for b in [basis_full,basis_pcon,basis_pcon_symm]:
	for dtype1,dtype2 in product(dtypes,dtypes):
		H = hamiltonian(static,[],basis=b,dtype=dtype1)
		H_op = quantum_LinearOperator(static,basis=b,dtype=dtype1)
		
		for i in range(10):
			v = np.random.uniform(-1,1,size=(b.Ns,)) + 1j*np.random.uniform(-1,1,size=(b.Ns,))
			v /= np.linalg.norm(v)


			v1 = H.dot(v)
			v2 = H_op.dot(v)


			atol = eps(dtype1,dtype2)
			np.testing.assert_allclose(v1,v2,atol=atol)

			v = np.random.uniform(-1,1,size=(b.Ns,10)) + 1j*np.random.uniform(-1,1,size=(b.Ns,10))
			v /= np.linalg.norm(v)


			v1 = H.dot(v)
			v2 = H_op.dot(v)


			atol = eps(dtype1,dtype2)
			np.testing.assert_allclose(v1,v2,atol=atol)
