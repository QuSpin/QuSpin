from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spinless_fermion_basis_1d,photon_basis
import numpy as np
import scipy.sparse as sp
from numpy.linalg import norm
from numpy.random import random,seed

seed()
dtypes=[np.float32,np.float64,np.complex64,np.complex128]
# dtypes=[np.complex128]


def eps(dtype):
	return 2*10.0**(-5)


def check_m(Lmax):
	for dtype in dtypes:
		for L in range(2,Lmax+1):
			h=[[2.0*random()-1.0,i] for i in range(L)]
			J1=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L)]
			J2_p=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L)]
			J2_m=[[-J2_p[i][0],i,(i+1)%L] for i in range(L)]

			static=[["zz",J1],["+-",J2_p],["-+",J2_m],["z",h]]
			
			basis=spinless_fermion_basis_1d(L=L)
			H=hamiltonian(static,[],dtype=dtype,basis=basis)
			Ns=H.Ns
			E=H.eigvalsh()

			Em=[]
			for Nf in range(L+1):
				basis=spinless_fermion_basis_1d(L=L,Nf=Nf)
				H=hamiltonian(static,[],dtype=dtype,basis=basis)
				Etemp=H.eigvalsh()
				Em.append(Etemp)

			Em=np.concatenate(Em)
			Em.sort()
			

			if norm(Em-E) > Ns*eps(dtype):
				raise Exception( "test failed m symmetry at L={0:3d} with dtype {1} {2}".format(L,dtype,norm(Em-E) ) )

def check_p(L,dtype,Nf=None):
	L_2=int(L/2)
	hr=[2.0*random()-1.0 for i in range(L_2)]
	hi=[hr[i] for i in range(L_2)]
	hi.reverse()
	hi.extend(hr)
	
	h=[[hi[i],i] for i in range(L)]
	J=[[1.0,i,(i+1)%L] for i in range(L-1)]
	J_p=[[1.0,i,(i+1)%L] for i in range(L-1)]
	J_m=[[-1.0,i,(i+1)%L] for i in range(L-1)]

	J_pp=[[np.sqrt(2),i,(i+1)%L] for i in range(L-1)] # OBC
	J_mm=[[-np.sqrt(2),i,(i+1)%L] for i in range(L-1)] # OBC

	static=[["+-",J_p],["-+",J_m],["n",h],["nn",J]]

	basis=spinless_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spinless_fermion_basis_1d(L=L,Nf=Nf,pblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spinless_fermion_basis_1d(L=L,Nf=Nf,pblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ep=np.concatenate((E1,E2))
	Ep.sort()

	if norm(Ep-E) > Ns*eps(dtype):
		raise Exception( "test failed p symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf,norm(Ep-E)) )


def check_obc(Lmax):

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_p(L,dtype,Nf=int(L/2))
			check_p(L,dtype)


def check_t(L,dtype,Nf=None):
	hx=random()
	J=random()
	Delta=random()
	h=[[hx,i] for i in range(L)]
	J1  =[[+J,i,(i+1)%L] for i in range(L)]

	J1_p=[[+J,i,(i+1)%L] for i in range(L)]
	J1_m=[[-J,i,(i+1)%L] for i in range(L)]

	J_pp=[[+Delta,i,(i+1)%L] for i in range(L)] # PBC
	J_mm=[[-Delta,i,(i+1)%L] for i in range(L)] # PBC

	if type(Nf) is int:
		static=[["+-",J1_p],["-+",J1_m],["zz",J1],["z",h]]
	else:
		static=[["zz",J1],["++",J_pp],["--",J_mm],["z",h]]


	basis=spinless_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	Et=np.array([])
	for kblock in range(0,L):

		basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk)
		Et=np.append(Et,Hk.eigvalsh())
		
	
	Et.sort()
	
	if norm(Et-E) > Ns*eps(dtype):
		raise Exception( "test failed t symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf,norm(Et-E)) )


def check_t_p(L,dtype,Nf=None):
	hx=random()
	Jnn=random()
	h=[[hx,i] for i in range(L)]
	J=[[Jnn,i,(i+1)%L] for i in range(L)]
	J1_p=[[Jnn,i,(i+1)%L] for i in range(L)]
	J1_n=[[-Jnn,i,(i+1)%L] for i in range(L)]

	Delta=random()
	J_pp=[[+Delta,i,(i+1)%L] for i in range(L)] # PBC
	J_mm=[[-Delta,i,(i+1)%L] for i in range(L)] # PBC

	if type(Nf) is int:
		static=[["+-",J1_p],["-+",J1_n],["z",h],["zz",J]]
	else:
		static=[["zz",J],["+-",J1_p],["-+",J1_n],["+",h],["-",h]]


	L_2=int(L/2)

	if dtype is np.float32:
		kdtype = np.complex64
	elif dtype is np.float64:
		kdtype = np.complex128
	else:
		kdtype = dtype
		

	for kblock in range(-L_2+1,0):

		basisk=spinless_fermion_basis_1d(L=L,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinless_fermion_basis_1d(L=L,kblock=kblock,pblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
		basisk2=spinless_fermion_basis_1d(L=L,kblock=kblock,pblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		
		if norm(Ek-Ek1) > Ns*eps(dtype):
			raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )

		if norm(Ek-Ek2) > Ns*eps(dtype):
			raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )


	basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=0)
	Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=0,pblock=+1)
	Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
	basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=0,pblock=-1)
	Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)

	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps(dtype):
			raise Exception( "test failed t p symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,0,np.dtype(dtype),Nf,norm(Ek-Ekp)) )


	if L%2 == 0:	
		for kblock in range(1,L_2):

			basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
			basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )

		basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,pblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,pblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps(dtype):
				raise Exception( "test failed t pc symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,int(L/2),np.dtype(dtype),Nf,norm(Ek-Ekp)) )

	else:
		for kblock in range(1,L_2+1):
			
			basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
			basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )


def check_pbc(Lmax):

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,1):
			check_t(L,dtype)
			for Nf in range(L+1):
				check_t(L,dtype,Nf=Nf)
	
	for dtype in dtypes:
		for L in range(2,Lmax+1,1):
			check_t_p(L,dtype)
			for Nf in range(L+1):
				check_t_p(L,dtype,Nf=Nf)
	
		

check_m(4)
check_obc(8)
check_pbc(8)







