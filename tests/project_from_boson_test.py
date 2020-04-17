from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
from numpy.random import random,seed

seed()
dtypes=[np.float32,np.float64,np.complex64,np.complex128]

def J(L,jb,l):
	blist=[]
	for i,j in jb:
		b=[j]
		b.extend([(i+j)%L for j in range(l)])
		blist.append(b)

	return blist

def Jnn(L,jb,l):
	blist=[]
	for i,j in jb:
		b=[j]
		b.extend([(i+j)%L for j in range(0,l+1,2)])
		blist.append(b)

	return blist


def getvec_boson(L,H1,static,sps=2,Nb=None,kblock=None,pblock=None,a=1,sparse=True):
	jb=[[i,1.0] for i in range(L)]
	dtype=np.complex128

	b = boson_basis_1d(L,sps=sps,Nb=Nb,kblock=kblock,pblock=pblock,a=a)
	
	Ns = b.Ns
	if Ns == 0:
		return 

	H2 = hamiltonian(static,[],basis=b,dtype=dtype)

	E,v0 = H2.eigh()
	v = b.get_vec(v0,sparse=sparse)
	P = b.get_proj(dtype=np.complex128)

	if sp.issparse(v):
		v = v.todense()


	H2 = H2.todense()
	H2 = v0.T.conj() * (H2 * v0)
	H1 = v.T.conj().dot(H1.dot(v))
	err_msg = "get_vec() symmetries failed for L={0} {1}".format(b.N,b.blocks)
	np.testing.assert_allclose(H1,H2,atol=1e-10,err_msg=err_msg)


def check_getvec_boson(L,sps=2,a=1,sparse=True):
	dtype=np.complex128
	jb=[[i,1.0] for i in range(L)]
	static = [
						['nn',J(L,jb,2)],
						['+-',J(L,jb,2)],
						['-+',J(L,jb,2)],
						['nnnn',J(L,jb,4)],
						['+nn-',J(L,jb,4)],
						['-nn+',J(L,jb,4)],
						['++--',J(L,jb,4)],
						['--++',J(L,jb,4)],
						['+-+-',J(L,jb,4)],
						['-+-+',J(L,jb,4)],
					]
	b_full = boson_basis_1d(L,sps=sps)
	H1 = hamiltonian(static,[],basis=b_full,dtype=dtype)

	H1 = H1.todense()

	for k in range(-L//a,L//a):
		getvec_boson(L,H1,static,sps=sps,kblock=k,a=a,sparse=sparse)


	for j in range(-1,2,2):
		getvec_boson(L,H1,static,sps=sps,pblock=j,a=a,sparse=sparse)
		for k in range(-L//a,L//a):
			getvec_boson(L,H1,static,sps=sps,kblock=k,pblock=j,a=a,sparse=sparse)

	for Nb in range(L+1):
		for k in range(-L//a,L//a):
				getvec_boson(L,H1,static,sps=sps,Nb=Nb,kblock=k,a=a,sparse=sparse)

	for Nb in range(0,L+1):
		for j in range(-1,2,2):
			getvec_boson(L,H1,static,sps=sps,Nb=Nb,pblock=j,a=a,sparse=sparse)
			for k in range(-L//a,L//a):
				getvec_boson(L,H1,static,sps=sps,kblock=k,Nb=Nb,pblock=j,a=a,sparse=sparse)


	




check_getvec_boson(6,sps=2,sparse=True)
check_getvec_boson(6,sps=2,sparse=False)
check_getvec_boson(6,sps=3,sparse=True)
check_getvec_boson(6,sps=3,sparse=False)



