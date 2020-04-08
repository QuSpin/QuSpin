from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
from numpy.random import random,seed

seed(0)

def test(basis,pcon=False):

	P = basis.get_proj(np.complex128,pcon=pcon)

	Ns_full, Ns = P.shape

	v=np.random.normal(size=Ns)
	v/=np.linalg.norm(v)

	v_full=np.random.normal(size=Ns_full)
	v_full/=np.linalg.norm(v_full)

	err_msg = "get_vec/get_vec_inv test failed for L={0}".format(basis.__class__)
	
	np.testing.assert_allclose(P.dot(v)       , basis.get_vec(v, sparse=False)         ,atol=1e-10,err_msg=err_msg)
	np.testing.assert_allclose(P.H.dot(v_full), basis.get_vec_inv(v_full, sparse=False),atol=1e-10,err_msg=err_msg)



L=4
z = -(np.arange(L)+1)
p = np.arange(L)[::-1]

bases=[spin_basis_general(L),
	   spin_basis_general(L,Nup=L//2),
	   spin_basis_general(L,zb=(z,0)),
	   spin_basis_general(L,zb=(z,1)),
	   spin_basis_general(L,pb=(p,0)),
	   spin_basis_general(L,pb=(p,1)),
	   spin_basis_general(L,zb=(z,0),pb=(p,0)),
	   spin_basis_general(L,zb=(z,0),pb=(p,1)),
	   spin_basis_general(L,zb=(z,1),pb=(p,0)),
	   spin_basis_general(L,zb=(z,1),pb=(p,1)),
	   ] 

for basis in bases:
	test(basis)


