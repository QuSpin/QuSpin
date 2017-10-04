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
# dtypes=[np.float32,np.float64,np.complex64,np.complex128]
dtypes=[np.complex128]


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


def check_c(L,dtype,Nf=None):

	h=[[2.0*random()-1.0,i] for i in range(L)]
	J1=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
	J2_p=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
	J2_m=[[-J2_p[i][0],i,(i+1)%L] for i in range(L-1)]

	J_pp=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
	J_mm=[[-J_pp[i][0],i,(i+1)%L] for i in range(L-1)]

	if type(Nf) is int:
		static=[["+-",J2_p],["-+",J2_m],["zz",J1]]
	else:
		static=[["+-",J2_p],["-+",J2_m],["zz",J1],["++",J_pp],["--",J_mm]]

	basis=spinless_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spinless_fermion_basis_1d(L=L,Nf=Nf,cblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spinless_fermion_basis_1d(L=L,Nf=Nf,cblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()

	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed z symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf, norm(Ez-E)))


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



def check_pc(L,dtype,Nf=None):
	L_2=int(L/2)
	hr=[(i+0.1)**2/float(L**2) for i in range(L_2)]
	hi=[-(i+0.1)**2/float(L**2) for i in range(L_2)]
	hi.reverse()
	hi.extend(hr)
	h=[[hi[i],i] for i in range(L)]
	J=[[1.0,i,(i+1)%L] for i in range(L-1)]
	J_p=[[+1.0,i,(i+1)%L] for i in range(L-1)]
	J_n=[[-1.0,i,(i+1)%L] for i in range(L-1)]

	static=[["zz",J],["+-",J_n],["-+",J_p],["z",h]]

	basis=spinless_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spinless_fermion_basis_1d(L=L,Nf=Nf,pcblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spinless_fermion_basis_1d(L=L,Nf=Nf,pcblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Epz=np.concatenate((E1,E2))
	Epz.sort()

	if norm(Epz-E) > Ns*eps(dtype):
		raise Exception( "test failed pc symmetry at L={0:3d} with dtype {1} and Nf={2:2d} {3}".format(L,np.dtype(dtype),Nf,norm(Epz-E)) )



def check_p_c(L,dtype,Nf=None):
	J=[[1.0,i,(i+1)%L] for i in range(L-1)]
	J_p=[[1.0,i,(i+1)%L] for i in range(L-1)]
	J_m=[[-1.0,i,(i+1)%L] for i in range(L-1)]

	static=[["+-",J_p],["-+",J_m],["zz",J]]

	basis=spinless_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spinless_fermion_basis_1d(L=L,Nf=Nf,pblock=1,cblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spinless_fermion_basis_1d(L=L,Nf=Nf,pblock=-1,cblock=1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)
	basis3=spinless_fermion_basis_1d(L=L,Nf=Nf,pblock=1,cblock=-1)
	H3=hamiltonian(static,[],dtype=dtype,basis=basis3)
	basis4=spinless_fermion_basis_1d(L=L,Nf=Nf,pblock=-1,cblock=-1)
	H4=hamiltonian(static,[],dtype=dtype,basis=basis4)
	
	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	E3=H3.eigvalsh()
	E4=H4.eigvalsh()

	
	Epz=np.concatenate((E1,E2,E3,E4))
	Epz.sort()

	if norm(Epz-E) > Ns*eps(dtype):
		raise Exception( "test failed p z symmetry at L={0:} with dtype {1} and Nf {2:} {3}".format(L,np.dtype(dtype),Nf,norm(Epz-E)) )




def check_obc(Lmax):

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_c(L,dtype,Nf=int(L/2))
			check_c(L,dtype)
	
	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_p(L,dtype,Nf=int(L/2))
			check_p(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_p_c(L,dtype,Nf=int(L/2))
			check_p_c(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_pc(L,dtype,Nf=int(L/2))
			check_pc(L,dtype) 



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


def check_t_c(L,dtype,Nf=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+1)%L] for i in range(L)]
	J1_p=[[+J,i,(i+1)%L] for i in range(L)]
	J1_n=[[-J,i,(i+1)%L] for i in range(L)]
	
	Delta=random()
	J_pp=[[+Delta,i,(i+1)%L] for i in range(L)] # PBC
	J_mm=[[-Delta,i,(i+1)%L] for i in range(L)] # PBC

	if type(Nf) is int:
		static=[["+-",J1_p],["-+",J1_n],["z",h],["zz",J1]]
	else:
		static=[["zz",J1],["++",J_pp],["--",J_mm]]

	L_2=int(L/2)

	for kblock in range(-L_2+1,L_2+1):

		basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,cblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,cblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()


		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t z symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf,norm(Ek-Ekz)) )

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
		static=[["zz",J],["++",J_pp],["--",J_mm]]


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


def check_t_pc(L,dtype,Nf=None):
	hx=random()*0.0
	hz=random()*0.0
	J=random()
	h1=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+1)%L] for i in range(L)]
	J1_p=[[J,i,(i+1)%L] for i in range(L)]
	J1_n=[[-J,i,(i+1)%L] for i in range(L)]
	h2=[[hz*(-1)**i,i] for i in range(L)]

	Delta=random()
	J_pp=[[+Delta,i,(i+1)%L] for i in range(L)] # PBC
	J_mm=[[-Delta,i,(i+1)%L] for i in range(L)] # PBC

	if type(Nf) is int:
		static=[["+-",J1_p],["-+",J1_n],["z",h2],["zz",J1]]
	else:
		static=[["zz",J1],["++",J_pp],["--",J_pp]]	

	if dtype is np.float32:
		kdtype = np.complex64
	elif dtype is np.float64:
		kdtype = np.complex128
	else:
		kdtype = dtype

	
	a=2
	L_2=int(L/(a*2))
	for kblock in range(-L_2+1,0):

		basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,pcblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,pcblock=-1) 
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		if norm(Ek-Ek1) > Ns*eps(dtype):
			raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )

		if norm(Ek-Ek2) > Ns*eps(dtype):
			raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )

	basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=0,a=a)
	Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=0,a=a,pcblock=+1)
	Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
	basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=0,a=a,pcblock=-1)
	Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)
	
	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps(dtype):
			raise Exception( "test failed t pc symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,0,np.dtype(dtype),Nf,norm(Ek-Ekp)) )

	if((L/a)%2 == 0):
		for kblock in range(1,L_2):

			basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,pcblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
			
			basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,pcblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )

		basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,a=a)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,a=a,pcblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,a=a,pcblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps(dtype):
				raise Exception( "test failed t pc symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,int(L/2),np.dtype(dtype),Nf,norm(Ek-Ekp)) )
	else:
		for kblock in range(1,L_2+1):

			basisk=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,pcblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
			basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,pcblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )


def check_t_p_c(L,dtype,Nf=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+1)%L] for i in range(L)]
	J1_p=[[J,i,(i+1)%L] for i in range(L)]
	J1_n=[[-J,i,(i+1)%L] for i in range(L)]
	
	Delta=random()
	J_pp=[[+Delta,i,(i+1)%L] for i in range(L)] # PBC
	J_mm=[[-Delta,i,(i+1)%L] for i in range(L)] # PBC

	if type(Nf) is int:
		static=[["+-",J1_p],["-+",J1_n],["z",h],["zz",J1]]
	else:
		static=[["zz",J1],["++",J_pp],["--",J_mm]]	

	L_2=int(L/2)
	for kblock in range(-L_2+1,L_2+1):

		basisk1=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1)
		Hkp1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1)
		Hkp2=hamiltonian(static,[],dtype=dtype,basis=basisk2)
		Ns=Hkp1.Ns
		Ekp1=Hkp1.eigvalsh()
		Ekp2=Hkp2.eigvalsh()

		basisk11=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1,cblock=+1)
		Hkpz11=hamiltonian(static,[],dtype=dtype,basis=basisk11)
		basisk12=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1,cblock=-1) 
		Hkpz12=hamiltonian(static,[],dtype=dtype,basis=basisk12)	
		Ekpz11=Hkpz11.eigvalsh()
		Ekpz12=Hkpz12.eigvalsh()

		Ekpz1=np.concatenate((Ekpz11,Ekpz12))
		Ekpz1.sort()

		basisk21=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1,cblock=+1)
		Hkpz21=hamiltonian(static,[],dtype=dtype,basis=basisk21)
		basisk22=spinless_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1,cblock=-1) 
		Hkpz22=hamiltonian(static,[],dtype=dtype,basis=basisk22)	
		Ekpz21=Hkpz21.eigvalsh()
		Ekpz22=Hkpz22.eigvalsh()

		Ekpz2=np.concatenate((Ekpz21,Ekpz22))
		Ekpz2.sort()
			
		if norm(Ekp1-Ekpz1) > Ns*eps(dtype):
			raise Exception( "test failed t z p+  symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ekp1-Ekpz1)) )

		if norm(Ekp2-Ekpz2) > Ns*eps(dtype):
			raise Exception( "test failed t z p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ekp2-Ekpz2)) )

		if(kblock not in [0,L_2]):
			if norm(Ekp2-Ekpz1) > Ns*eps(dtype):
				raise Exception( "test failed t z p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ekp2-Ekpz1)) )

			if norm(Ekp1-Ekpz2) > Ns*eps(dtype):
				raise Exception( "test failed t z p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ekp1-Ekpz2)) )


def check_pbc(Lmax):

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,1):
			check_t(L,dtype)
			for Nf in range(L+1):
				check_t(L,dtype,Nf=Nf)
	
	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,2):
			check_t_c(L,dtype,Nf=int(L/2))
			check_t_c(L,dtype)
	
	for dtype in dtypes:
		for L in range(2,Lmax+1,1):
			check_t_p(L,dtype)
			for Nf in range(L+1):
				check_t_p(L,dtype,Nf=Nf)
	
	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_t_pc(L,dtype,Nf=int(L/2))
			check_t_pc(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_t_p_c(L,dtype,Nf=int(L/2))
			check_t_p_c(L,dtype) 


		

check_m(4)
check_obc(8)
check_pbc(8)







