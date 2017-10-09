from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spinless_fermion_basis_1d
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


def getvec_spinless_fermion(L,H1,static,Nf=None,kblock=None,pblock=None,a=1,sparse=True):
	jb=[[i,1.0] for i in range(L)]
	jbhc=[[i,-1.0] for i in range(L)]
	dtype=np.complex128

	b = spinless_fermion_basis_1d(L,Nf=Nf,kblock=kblock,pblock=pblock,a=a)
	
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
	H2 = np.asarray(v0.T.conj().dot(H2.dot(v0)))
	H1 = np.asarray(v.T.conj().dot(H1.dot(v)))
	err_msg = "get_vec() symmetries failed for L={0} {1}".format(b.N,b.blocks)
	np.testing.assert_allclose(H1,H2,atol=1e-10,err_msg=err_msg)
	


def check_getvec_spinless_fermion(L,a=1,sparse=True):
	dtype=np.complex128
	jb=[[i,1.0] for i in range(L)]
	jbhc=[[i,-1.0] for i in range(L)]
	static = [
						['nn',J(L,jb,2)],
						['+-',J(L,jb,2)],
						['-+',J(L,jbhc,2)],
						['nnnn',J(L,jb,4)],
						['+nn-',J(L,jb,4)],
						['-nn+',J(L,jbhc,4)],
						['++--',J(L,jb,4)],
						['--++',J(L,jb,4)],
						['+-+-',J(L,jb,4)],
						['-+-+',J(L,jb,4)],
					]

	b_full = spinless_fermion_basis_1d(L)
	H1 = hamiltonian(static,[],basis=b_full,dtype=dtype)

	H1 = H1.todense()

	for k in range(-L//a,L//a):
		getvec_spinless_fermion(L,H1,static,kblock=k,a=a,sparse=sparse)


	for j in range(-1,2,2):
		getvec_spinless_fermion(L,H1,static,pblock=j,a=a,sparse=sparse)
		for k in range(-L//a,L//a):
			getvec_spinless_fermion(L,H1,static,kblock=k,pblock=j,a=a,sparse=sparse)

	for Nf in range(L+1):
		for k in range(-L//a,L//a):
				getvec_spinless_fermion(L,H1,static,Nf=Nf,kblock=k,a=a,sparse=sparse)

	for Nf in range(0,L+1):
		for j in range(-1,2,2):
			getvec_spinless_fermion(L,H1,static,Nf=Nf,pblock=j,a=a,sparse=sparse)
			for k in range(-L//a,L//a):
				getvec_spinless_fermion(L,H1,static,kblock=k,Nf=Nf,pblock=j,a=a,sparse=sparse)


	




check_getvec_spinless_fermion(6,sparse=True)
check_getvec_spinless_fermion(6,sparse=False)



