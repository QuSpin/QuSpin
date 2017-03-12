from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d,photon_basis
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


def getvec(L,Nup=None,kblock=None,pblock=None,zblock=None,pzblock=None,a=1,sparse=True):
	jb=[[i,1.0] for i in range(L)]
	dtype=np.complex128

	b = spin_basis_1d(L,Nup=Nup,kblock=kblock,pblock=pblock,zblock=zblock,pzblock=pzblock,a=a)
	
	Ns = b.Ns
	

	static = [
						['xx',J(L,jb,2)],
						['yy',J(L,jb,2)],
						['zz',J(L,jb,2)],
						['+-',J(L,jb,2)],
						['-+',J(L,jb,2)],
						['zzzz',J(L,jb,4)],
						['xxxx',J(L,jb,4)],
						['yyyy',J(L,jb,4)],
						['xxzz',J(L,jb,4)],
						['zzxx',J(L,jb,4)],
						['yyzz',J(L,jb,4)],
						['zzyy',J(L,jb,4)],
						['yyxx',J(L,jb,4)],
						['xxyy',J(L,jb,4)],
						['+zz-',J(L,jb,4)],
						['-zz+',J(L,jb,4)],
						['+xx-',J(L,jb,4)],
						['-xx+',J(L,jb,4)],
						['+yy-',J(L,jb,4)],
						['-yy+',J(L,jb,4)],
						['++--',J(L,jb,4)],
						['--++',J(L,jb,4)],
						['+-+-',J(L,jb,4)],
						['-+-+',J(L,jb,4)],
					]


	H1 = hamiltonian(static,[],N=L,dtype=dtype)
	H2 = hamiltonian(static,[],basis=b,dtype=dtype)

	E,v0 = H2.eigh()
	v = b.get_vec(v0,sparse=sparse)
	P = b.get_proj(dtype=np.complex128)

	if sp.issparse(v):
		v = v.todense()


	if v.shape[0] != 0:
		H1 = H1.todense()
		H2 = H2.todense()
		H2 = v0.T.conj() * (H2 * v0)
		H1 = v.T.conj().dot(H1.dot(v))
		if np.abs(np.linalg.norm(H1-H2)) > 10**(-10):
			raise Exception("get_vec() TPZ symmetries failed for L={0} {1}".format(b.N,b.blocks))
	else: 
		pass

	
def getvec_zA_zB(L,kblock=None,zAblock=None,zBblock=None,a=2,sparse=True):
	jb=[[i,1.0] for i in range(L)]
	dtype=np.complex128

	b_zA_zB = spin_basis_1d(L,kblock=kblock,zAblock=zAblock,zBblock=zBblock,a=a)
	
	Ns_zA_zB = b_zA_zB.Ns

	static_zA_zB = [	['xx',J(L,jb,2)],
						['zz',Jnn(L,jb,2)],
						['zxz',J(L,jb,3)],
						['zxzx',J(L,jb,4)],
						['xxxx',J(L,jb,4)],
						['yyyy',J(L,jb,4)],
						['xzxz',J(L,jb,4)],
						['yzyz',J(L,jb,4)],
						['zyzy',J(L,jb,4)],
						['z+z-',J(L,jb,4)],
						['z-z+',J(L,jb,4)],
					]
	
	H1 = hamiltonian(static_zA_zB,[],N=L,dtype=dtype)
	H2_zA_zB = hamiltonian(static_zA_zB,[],N=L,basis=b_zA_zB,dtype=dtype)
	

	E_zA_zB,v0_zA_zB=H2_zA_zB.eigh()
	v_zA_zB = b_zA_zB.get_vec(v0_zA_zB,sparse=sparse)

	if sp.issparse(v_zA_zB):
		v_zA_zB = v_zA_zB.todense()

	if v_zA_zB.shape[0] != 0:
		H1 = H1.todense()
		H2_zA_zB = H2_zA_zB.todense()
		H2_zA_zB = v0_zA_zB.T.conj() * (H2_zA_zB * v0_zA_zB)
		H1 = v_zA_zB.T.conj() * ( H1 * v_zA_zB)
		if np.abs(np.linalg.norm(H1-H2_zA_zB)) > 10**(-10):
			raise Exception("get_vec() zA_zB Symmetries failed for L={0} {1}".format(b.N,b.blocks))
	else: 
		pass	


def check_getvec(L,a=1,sparse=True):
	for k in range(-L//a,L//a):
		getvec(L,kblock=k,a=a,sparse=sparse)


	for j in range(-1,2,2):
		getvec(L,pblock=j,a=a,sparse=sparse)
		for k in range(-L//a,L//a):
			getvec(L,kblock=k,pblock=j,a=a,sparse=sparse)

	Nup=None

	for i in range(-1,2,2):
		for j in range(-1,2,2):
			getvec(L,Nup=Nup,pblock=i,zblock=j,a=a,sparse=sparse)
			for k in range(-L//a,L//a):
				getvec(L,kblock=k,Nup=Nup,pblock=i,zblock=j,a=a,sparse=sparse)

	for j in range(-1,2,2):
			getvec(L,Nup=Nup,pzblock=j,a=a,sparse=sparse)
			for k in range(-L//a,L//a):
				 getvec(L,kblock=k,Nup=Nup,pzblock=j,a=a,sparse=sparse)

	for j in range(-1,2,2):
		getvec(L,Nup=Nup,zblock=j,a=a)
		for k in range(-L//a,L//a):
			getvec(L,kblock=k,Nup=Nup,zblock=j,a=a,sparse=sparse)

	for Nup in range(L+1):
		for k in range(-L//a,L//a):
				getvec(L,Nup=Nup,kblock=k,a=a,sparse=sparse)

	for Nup in range(0,L+1):
		for j in range(-1,2,2):
			getvec(L,Nup=Nup,pblock=j,a=a,sparse=sparse)
			for k in range(-L//a,L//a):
				getvec(L,kblock=k,Nup=Nup,pblock=j,a=a,sparse=sparse)

	if (L%2)==0:
		Nup=L//2
		for i in range(-1,2,2):
			for j in range(-1,2,2):
				getvec(L,Nup=Nup,pblock=i,zblock=j,a=a,sparse=sparse)
				for k in range(-L//a,L//a):
					getvec(L,kblock=k,Nup=Nup,pblock=i,zblock=j,a=a,sparse=sparse)

		for j in range(-1,2,2):
				getvec(L,Nup=Nup,pzblock=j,a=a,sparse=sparse)
				for k in range(-L//a,L//a):
					getvec(L,kblock=k,Nup=Nup,pzblock=j,a=a,sparse=sparse)

		for j in range(-1,2,2):
			getvec(L,Nup=Nup,zblock=j,a=a,sparse=sparse)
			for k in range(-L//a,L//a):
				getvec(L,kblock=k,Nup=Nup,zblock=j,a=a,sparse=sparse)

def check_getvec_zA_zB(L,a=2,sparse=True):

	for k in range(-L//a,L//a):
		getvec_zA_zB(L,kblock=k,a=a,sparse=sparse)
	
	for j in range(-1,2,2):
		getvec_zA_zB(L,zAblock=j,a=a,sparse=sparse)
		for k in range(-L//a,L//a):
			getvec_zA_zB(L,kblock=k,zAblock=j,a=a,sparse=sparse)
	
	for j in range(-1,2,2):
		getvec_zA_zB(L,zBblock=j,a=a,sparse=sparse)
		for k in range(-L//a,L//a):
			getvec_zA_zB(L,kblock=k,zBblock=j,a=a,sparse=sparse)

	for i in range(-1,2,2):
		for j in range(-1,2,2):
			getvec_zA_zB(L,zAblock=i,zBblock=j,a=a,sparse=sparse)
			for k in range(-L//a,L//a):
				getvec_zA_zB(L,kblock=k,zAblock=i,zBblock=j,a=a,sparse=sparse)
	





for L in range(4,9):
	check_getvec(L,sparse=True)
	check_getvec(L,sparse=False)
	if L%2 == 0:
		check_getvec_zA_zB(L,sparse=True)
		check_getvec_zA_zB(L,sparse=False)
















