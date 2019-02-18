from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
import numpy as np


def lte(r,r0,n):
	return 0 <= (r-r0).dot(n)


def lt(r,r0,n):
	return 0 < (r-r0).dot(n)


def tilted_square_transformations(n,m,a_1=None,a_2=None):
	if n < 1 or m < 1:
		raise ValueError("n and m must be >= 1")

	def test_r(r):
		return lte(r,r0,L2) and lte(r,r0,L1) and lt(r,r1,-L2) and lt(r,r1,-L1)

	L1=np.array([n,m])
	L2=np.array([m,-n])

	r0 = np.array([0,0])
	r1 = L1+L2

	x = np.arange(0,n+m+1,1)
	x = np.kron(x,np.ones_like(x))
	y = np.arange(-n,m+1,1)
	y = np.kron(np.ones_like(y),y)
	r_list = np.vstack((x,y)).T

	r_lat = np.array([r for r in r_list[:] if test_r(r)])

	arg = np.argsort(r_lat[:,0]+1j*r_lat[:,1])

	r_lat = r_lat[arg].copy()


	if a_1 is None:
		a_1 = np.array([0,1])

	if a_2 is None:
		a_2 = np.array([1,0])


	Pr = np.array([[0,-1],[1,0]])

	angle = -np.arctan2(L2[1],L2[0])

	R = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
	L = np.sqrt(n**2+m**2)

	r_lat_Pr = R.dot(Pr.dot(r_lat.T))+1e-15
	r_lat_Pr = R.T.dot(r_lat_Pr%L).T
	r_lat_Pr = np.round(r_lat_Pr).astype(int)
	Pr = []
	for r in r_lat_Pr[:]:
		[[i]] = np.argwhere((r[0]==r_lat[:,0])*(r[1]==r_lat[:,1]))
		Pr.append(i)

	Pr = np.array(Pr)

	r_lat_T1 = R.dot((r_lat+a_1).T)+1e-15
	r_lat_T1 = R.T.dot(r_lat_T1%L).T
	r_lat_T1 = np.round(r_lat_T1).astype(int)
	T1 = []
	for r in r_lat_T1[:]:
		[[i]] = np.argwhere((r[0]==r_lat[:,0])*(r[1]==r_lat[:,1]))
		T1.append(i)

	T1 = np.array(T1)

	r_lat_T2 = R.dot((r_lat+a_2).T)+1.1e-15
	r_lat_T2 = R.T.dot(r_lat_T2%L).T
	r_lat_T2 = np.round(r_lat_T2).astype(int)
	T2 = []
	for r in r_lat_T2[:]:
		[[i]] = np.argwhere((r[0]==r_lat[:,0])*(r[1]==r_lat[:,1]))
		T2.append(i)

	T2 = np.array(T2)

	a = np.arange(n**2+m**2)
	b = a[T1].copy()
	t1=1
	while not np.array_equal(a,b):
		b = b[T1]
		t1 += 1


	a = np.arange(n**2+m**2)
	b = a[T2].copy()
	t2=1
	while not np.array_equal(a,b):
		b = b[T2]
		t2 += 1

	a = np.arange(n**2+m**2)
	b = a[Pr].copy()
	pr=1
	while not np.array_equal(a,b):
		b = b[Pr]
		pr += 1

	r_lat_Tx = R.dot((r_lat+np.array([1,0])).T)+1.1e-15
	r_lat_Tx = R.T.dot(r_lat_Tx%L).T
	r_lat_Tx = np.round(r_lat_Tx).astype(int)
	Tx = []
	for r in r_lat_Tx[:]:
		[[i]] = np.argwhere((r[0]==r_lat[:,0])*(r[1]==r_lat[:,1]))
		Tx.append(i)

	Tx = np.array(Tx)

	r_lat_Ty = R.dot((r_lat+np.array([0,1])).T)+1.1e-15
	r_lat_Ty = R.T.dot(r_lat_Ty%L).T
	r_lat_Ty = np.round(r_lat_Ty).astype(int)

	Ty = []
	for r in r_lat_Ty[:]:
		[[i]] = np.argwhere((r[0]==r_lat[:,0])*(r[1]==r_lat[:,1]))
		Ty.append(i)

	Ty = np.array(Ty)


	return T1,t1,T2,t2,Pr,pr,Tx,Ty


def get_blocks(T1,t1,T2,t2,Pr,pr):
	for i1 in range(t1):
		for i2 in range(t2):
			if i1*2==t1 and i2*2==t2:
				for ip in range(pr):
					blocks = dict(tb1=(T1,i1),tb2=(T2,i2),prb=(Pr,ip))
					yield blocks
			elif i1==0 and i2 == 0:
				for ip in range(pr):
					blocks = dict(tb1=(T1,i1),tb2=(T2,i2),prb=(Pr,ip))
					yield blocks
			else:
				blocks = dict(tb1=(T1,i1),tb2=(T2,i2))
				yield blocks


def test_Nup(n,m,S):
	nmax = int(eval("2*"+S))
	N = n**2 + m**2
	Nups=range(nmax*N+1)

	a_1 = np.array([0,1])
	a_2 = np.array([1,0])
	T1,t1,T2,t2,Pr,pr,Tx,Ty = tilted_square_transformations(n,m,a_1,a_2)

	Jzz = [[-1.0,i,Tx[i]] for i in range(N)]
	Jzz.extend([-1.0,i,Ty[i]] for i in range(N))
	hx = [[-1.0,i] for i in range(N)]

	fzz = lambda x:1-x
	fx = lambda x:x
	dynamic=[["zz",Jzz,fzz,()],["+-",Jzz,fx,()],["-+",Jzz,fx,()]]
	ss = np.linspace(0,1,11)

	for Nup in Nups:

		basis_full = spin_basis_general(N,S=S,Nup=Nup)
		H_full = hamiltonian([],dynamic,basis=basis_full,dtype=np.float64)

		E_full = []
		for s in ss:
			E_full.append(H_full.eigvalsh(time=s))

		E_full = np.vstack(E_full)

		E_symm = np.zeros((E_full.shape[0],0),dtype=E_full.dtype)
		no_checks = dict(check_symm=False,check_pcon=False,check_herm=False)

		for blocks in get_blocks(T1,t1,T2,t2,Pr,pr):
			basis = spin_basis_general(N,S=S,Nup=Nup,**blocks)
			H = hamiltonian([],dynamic,basis=basis,dtype=np.complex128,**no_checks)
			if H.Ns == 0:
				continue

			block,values = zip(*blocks.items())
			Tr,qs = zip(*values)
			print(basis.Ns,Nup,zip(block,qs))

			for Hd in H._dynamic.values():
				dH=(Hd-Hd.T.conj())
				n = np.linalg.norm(dH.data)
				np.testing.assert_allclose(dH.data,0,atol=1e-7,err_msg=str(n))

			E_list = []
			for i,s in enumerate(ss):
				E = H.eigvalsh(time=s)
				E_list.append(E)

			E_list = np.vstack(E_list)
			E_symm = np.hstack((E_symm,E_list))

		E_symm.sort(axis=1)

		np.testing.assert_allclose(E_symm,E_full,atol=1e-13)


test_Nup(2,1,"1/2")
test_Nup(2,2,"1/2")
test_Nup(2,3,"1/2")
test_Nup(2,1,"1")
test_Nup(2,2,"1")

