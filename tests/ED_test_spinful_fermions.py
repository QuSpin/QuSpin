from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spinful_fermion_basis_1d # Hilbert spaces
import numpy as np # general math functions
from itertools import product
import scipy.sparse as sp
from numpy.linalg import norm
from numpy.random import random,seed


seed(0)

#no_checks = dict()
no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)


dtypes=[np.float32,np.float64,np.complex64,np.complex128]
	
def eps(dtype):
	return 2*10.0**(-5)


def check_m(Lmax):
	for dtype in dtypes:
		for L in range(2,Lmax+1):
			h1=[[2.0*random()-1.0,i] for i in range(L)]
			h2=[[2.0*random()-1.0,i] for i in range(L)]
			J1=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L)]
			J0=random()
			J2p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
			J2m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]
			J0=random()
			J1p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
			J1m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]

			static=[["z|n",J1],["+-|",J2p],["-+|",J2m],["|+-",J1p],["|-+",J1m],["z|",h1],["|n",h2]]
			
			basis=spinful_fermion_basis_1d(L=L)
			H=hamiltonian(static,[],dtype=dtype,basis=basis)
			Ns=H.Ns
			E=H.eigvalsh()

			Em=[]
			for Nf,Ndown in product(range(L+1),range(L+1)):
				basis=spinful_fermion_basis_1d(L=L,Nf=(Nf,Ndown))
				H=hamiltonian(static,[],dtype=dtype,basis=basis)
				Etemp=H.eigvalsh()
				Em.append(Etemp)

			Em=np.concatenate(Em)
			Em.sort()
			

			if norm(Em-E) > Ns*eps(dtype):
				raise Exception( "test failed m symmetry at L={0:3d} with dtype {1} {2}".format(L,dtype,norm(Em-E) ) )



def check_z(L,dtype,Nf=None):
	
	J1=[[2.0*random()-1.0,i,i] for i in range(L)]
	J0=random()
	J2p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
	J2m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]
	J1p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
	J1m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]

	static=[["z|z",J1],["+-|",J2p],["-+|",J2m],["|+-",J1p],["|-+",J1m]]

	
	basis=spinful_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()
	
	basis1=spinful_fermion_basis_1d(L=L,Nf=Nf,sblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1,**no_checks)
	basis2=spinful_fermion_basis_1d(L=L,Nf=Nf,sblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2,**no_checks)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()

	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed z symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf, norm(Ez-E)))

#check_z(4,np.float64,Nf=(2,2))
#check_z(4,np.complex128)


def check_p(L,dtype,Nf=None):

	""" 

	1) correct checks for spinful fermions and go back to OBC
	
	"""


	L_2=int(L/2)
	hr=[2.0*random()-1.0 for i in range(L_2)]
	hi=[hr[i] for i in range(L_2)]
	hi.reverse()
	hi.extend(hr)
	
	h=[[hi[i],i] for i in range(L)]

	J=[[1.0,i,i] for i in range(L)]

	J0=random()
	J2p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L-1)]
	J2m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L-1)]
	J0=random()
	J1p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L-1)]
	J1m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L-1)]


	if type(Nf) is tuple:
		if type(Nf[0]) is int and type(Nf[1]) is int:
			static=[["z|z",J],["+-|",J1p],["-+|",J1m],["|+-",J2p],["|-+",J2m],["z|",h]]
			#static=[["z|z",J],["+-|",J2p],["-+|",J2m],["|+-",J1p],["|-+",J1m]]
	else:
		static=[["z|z",J],["|+",h],["-|",h]]

	
	basis=spinful_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis,**no_checks)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spinful_fermion_basis_1d(L=L,Nf=Nf,pblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1,**no_checks)
	basis2=spinful_fermion_basis_1d(L=L,Nf=Nf,pblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2,**no_checks)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ep=np.concatenate((E1,E2))
	Ep.sort()

	if norm(Ep-E) > Ns*eps(dtype):
		raise Exception( "test failed p symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf,norm(Ep-E)) )


#check_p(4,np.float64,Nf=(1,3))
#check_p(4,np.float64)



def check_pz(L,dtype,Nf=None):
	L_2=int(L/2)
	hr=[2.0*random()-1.0 for i in range(L_2)]
	hi=[hr[i] for i in range(L_2)]
	hi.reverse()
	hi.extend(hr)

	h=[[hi[i],i] for i in range(L)]

	J=[[1.0,i,i] for i in range(L)]

	J0=random()
	Jp=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L-1)]
	Jm=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L-1)]
	
	static=[["z|z",J],["+-|",Jp],["-+|",Jm],["|+-",Jp],["|-+",Jm],["z|",h],["|z",h]]
	
	basis=spinful_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis,**no_checks)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spinful_fermion_basis_1d(L=L,Nf=Nf,psblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1,**no_checks)
	basis2=spinful_fermion_basis_1d(L=L,Nf=Nf,psblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2,**no_checks)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Epz=np.concatenate((E1,E2))
	Epz.sort()

	if norm(Epz-E) > Ns*eps(dtype):
		raise Exception( "test failed pz symmetry at L={0:3d} with dtype {1} and Nf={2:2d} {3}".format(L,np.dtype(dtype),Nf,norm(Epz-E)) )


#check_pz(4,np.float64,Nf=(2,2))
#exit()

def check_p_z(L,dtype,Nf=None):
	L_2=int(L/2)
	hr=[2.0*random()-1.0 for i in range(L_2)]
	hi=[hr[i] for i in range(L_2)]
	hi.reverse()
	hi.extend(hr)

	h=[[hi[i],i] for i in range(L)]

	J=[[1.0,i,i] for i in range(L)]

	J0=random()
	Jp=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L-1)]
	Jm=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L-1)]
	
	if type(Nf) is tuple:
		if type(Nf[0]) is int and type(Nf[1]) is int:
			static=[["z|z",J],["+-|",Jp],["-+|",Jm],["|+-",Jp],["|-+",Jm],["z|",h],["|z",h]]
	else:
		static=[["z|z",J],["+|",h],["-|",h],["|+",h],["|-",h]]

	basis=spinful_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis,**no_checks)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spinful_fermion_basis_1d(L=L,Nf=Nf,pblock=1,sblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1,**no_checks)
	basis2=spinful_fermion_basis_1d(L=L,Nf=Nf,pblock=-1,sblock=1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2,**no_checks)
	basis3=spinful_fermion_basis_1d(L=L,Nf=Nf,pblock=1,sblock=-1)
	H3=hamiltonian(static,[],dtype=dtype,basis=basis3,**no_checks)
	basis4=spinful_fermion_basis_1d(L=L,Nf=Nf,pblock=-1,sblock=-1)
	H4=hamiltonian(static,[],dtype=dtype,basis=basis4,**no_checks)
	
	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	E3=H3.eigvalsh()
	E4=H4.eigvalsh()

	
	Epz=np.concatenate((E1,E2,E3,E4))
	Epz.sort()

	if norm(Epz-E) > Ns*eps(dtype):
		raise Exception( "test failed pz symmetry at L={0:3d} with dtype {1} and Nf={2:2d} {3}".format(L,np.dtype(dtype),Nf,norm(Epz-E)) )


#check_p_z(4,np.float64,Nf=(2,2))
#check_p_z(4,np.complex128)


def check_obc(Lmax):
	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_z(L,dtype,Nf=(L//2,L//2))
			check_z(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			for Nup in range(L+1):
				check_t_p(L,dtype,Nf=(Nup,L-Nup))
				check_p(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_pz(L,dtype,Nf=(L//2,L//2))
			check_pz(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_p_z(L,dtype,Nf=(L//2,L//2))
			check_p_z(L,dtype) 



################################################



def check_t(L,dtype,Nf=None):
	hx=random()
	h=[[hx,i] for i in range(L)]

	J=random()
	J=[[J,i,(i+1)%L] for i in range(L)]

	J0=random()
	J2p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
	J2m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]
	J0=random()
	J1p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
	J1m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]


	if type(Nf) is tuple:
		if type(Nf[0]) is int and type(Nf[1]) is int:
			static=[["z|z",J],["+-|",J1p],["-+|",J1m],["|+-",J2p],["|-+",J2m],["z|",h]]
	else:
		static=[["z|z",J],["|+",h],["-|",h]]

	basis=spinful_fermion_basis_1d(L=L,Nf=Nf)
	H=hamiltonian(static,[],dtype=dtype,basis=basis,**no_checks)
	Ns=H.Ns
	E=H.eigvalsh()

	Et=np.array([])
	for kblock in range(0,L):

		basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk,**no_checks)
		Et=np.append(Et,Hk.eigvalsh())
		
	
	Et.sort()
	
	if norm(Et-E) > Ns*eps(dtype):
		raise Exception( "test failed t symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf,norm(Et-E)) )


#check_t(4,np.complex128,Nf=(1,3))
#check_t(4,np.complex128)


def check_t_z(L,dtype,Nf=None):

	h0=random()
	h=[[h0,i] for i in range(L)]

	J0=random()
	J=[[2.0*J0-1.0,i,i] for i in range(L)]

	J0=random()
	Jp=[[  2.0*J0-1.0 ,i,(i+1)%L] for i in range(L)]
	Jm=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]
	

	if type(Nf) is tuple:
		if type(Nf[0]) is int and type(Nf[1]) is int:
			static=[["z|z",J],["+-|",Jp],["-+|",Jm],["|+-",Jp],["|-+",Jm]]
	else:
		static=[["z|z",J],["+|",h],["-|",h],["|+",h],["|-",h]]


	L_2=int(L/2)

	for kblock in range(-L_2+1,L_2+1):

		basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk,**no_checks)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,sblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks)
		basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,sblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()


		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t z symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf,norm(Ek-Ekz)) )

#check_t_z(4,np.complex128,Nf=(2,2))
#check_t_z(4,np.complex128)


def check_t_p(L,dtype,Nf=None):

	hx=random()
	h=[[hx,i] for i in range(L)]

	J=random()
	J=[[J,i,i] for i in range(L)]

	J0=random()
	J2p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
	J2m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]
	J0=random()
	J1p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
	J1m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]


	if type(Nf) is tuple:
		if type(Nf[0]) is int and type(Nf[1]) is int:
			static=[["z|z",J],["+-|",J1p],["-+|",J1m],["|+-",J2p],["|-+",J2m],["z|",h]]
	else:
		static=[["z|z",J],["|+",h],["-|",h]]

	L_2=int(L/2)

	if dtype is np.float32:
		kdtype = np.complex64
	elif dtype is np.float64:
		kdtype = np.complex128
	else:
		kdtype = dtype
		

	for kblock in range(-L_2+1,0):

		basisk=spinful_fermion_basis_1d(L=L,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinful_fermion_basis_1d(L=L,kblock=kblock,pblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks) 
		basisk2=spinful_fermion_basis_1d(L=L,kblock=kblock,pblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		
		if norm(Ek-Ek1) > Ns*eps(dtype):
			raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )

		if norm(Ek-Ek2) > Ns*eps(dtype):
			raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )


	basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=0)
	Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=0,pblock=+1)
	Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks)
	basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=0,pblock=-1)
	Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)

	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps(dtype):
			raise Exception( "test failed t p symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,0,np.dtype(dtype),Nf,norm(Ek-Ekp)) )


	if L%2 == 0:	
		for kblock in range(1,L_2):

			basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks) 
			basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )

		basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,pblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks)
		basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,pblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps(dtype):
				raise Exception( "test failed t p symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,int(L/2),np.dtype(dtype),Nf,norm(Ek-Ekp)) )

	else:
		for kblock in range(1,L_2+1):
			
			basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks) 
			basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )
	
#check_t_p(4,np.complex128,Nf=(3,4))
#check_t_p(4,np.complex128)


def check_t_pz(L,dtype,Nf=None):

	h0=random()
	h=[[h0,i] for i in range(L)]

	J=[[1.0,i,i] for i in range(L)]

	J0=random()
	Jp=[[  2.0*J0-1.0 ,i,(i+1)%L] for i in range(L)]
	Jm=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]
	
	
	static=[["z|z",J],["+-|",Jp],["-+|",Jm],["|+-",Jp],["|-+",Jm],["z|",h],["|z",h]]

	if dtype is np.float32:
		kdtype = np.complex64
	elif dtype is np.float64:
		kdtype = np.complex128
	else:
		kdtype = dtype

	
	a=2
	L_2=int(L/(a*2))
	for kblock in range(-L_2+1,0):

		basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,psblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks)
		basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,psblock=-1) 
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		if norm(Ek-Ek1) > Ns*eps(dtype):
			raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )

		if norm(Ek-Ek2) > Ns*eps(dtype):
			raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )


	basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=0,a=a)
	Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=0,a=a,psblock=+1)
	Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks)
	basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=0,a=a,psblock=-1)
	Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)
	
	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps(dtype):
			raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,0,np.dtype(dtype),Nf,norm(Ek-Ekp)) )

	if((L/a)%2 == 0):
		for kblock in range(1,L_2):

			basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,psblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks) 
			
			basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,psblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )

		basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,a=a)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,a=a,psblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks)
		basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=L_2,a=a,psblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps(dtype):
				raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,int(L/2),np.dtype(dtype),Nup,norm(Ek-Ekp)) )
	else:
		for kblock in range(1,L_2+1):

			basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk,**no_checks)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,psblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks) 
			basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,a=a,psblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ek-Ek2)) )


#check_t_pz(8,np.complex128,Nf=(4,4))
#check_t_pz(6,np.complex128)

def check_t_p_z(L,dtype,Nf=None):
	h0=random()
	h=[[h0,i] for i in range(L)]

	J=[[1.0,i,i] for i in range(L)]

	J0=random()
	Jp=[[  2.0*J0-1.0 ,i,(i+1)%L] for i in range(L)]
	Jm=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]
	
	if type(Nf) is tuple:
		if type(Nf[0]) is int and type(Nf[1]) is int:
			static=[["z|z",J],["+-|",Jp],["-+|",Jm],["|+-",Jp],["|-+",Jm],["z|",h],["|z",h]]
	else:
		static=[["z|z",J],["+|",h],["-|",h],["|+",h],["|-",h]]

	
	L_2=int(L/2)
	for kblock in range(-L_2+1,L_2+1):

		print(kblock)

		basisk1=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1)
		Hkp1=hamiltonian(static,[],dtype=dtype,basis=basisk1,**no_checks)
		basisk2=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1)
		Hkp2=hamiltonian(static,[],dtype=dtype,basis=basisk2,**no_checks)
		Ns=Hkp1.Ns
		Ekp1=Hkp1.eigvalsh()
		Ekp2=Hkp2.eigvalsh()

		basisk11=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1,sblock=+1)
		Hkpz11=hamiltonian(static,[],dtype=dtype,basis=basisk11,**no_checks)
		basisk12=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=+1,sblock=-1) 
		Hkpz12=hamiltonian(static,[],dtype=dtype,basis=basisk12,**no_checks)	
		Ekpz11=Hkpz11.eigvalsh()
		Ekpz12=Hkpz12.eigvalsh()

		Ekpz1=np.concatenate((Ekpz11,Ekpz12))
		Ekpz1.sort()

		basisk21=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1,sblock=+1)
		Hkpz21=hamiltonian(static,[],dtype=dtype,basis=basisk21,**no_checks)
		basisk22=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock,pblock=-1,sblock=-1) 
		Hkpz22=hamiltonian(static,[],dtype=dtype,basis=basisk22,**no_checks)	
		Ekpz21=Hkpz21.eigvalsh()
		Ekpz22=Hkpz22.eigvalsh()

		Ekpz2=np.concatenate((Ekpz21,Ekpz22))
		Ekpz2.sort()

		print(Ekp1)
		print(Ekpz1)
		#exit()
			
		if norm(Ekp1-Ekpz1) > Ns*eps(dtype):
			raise Exception( "test failed t z p+  symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ekp1-Ekpz1)) )

		if norm(Ekp2-Ekpz2) > Ns*eps(dtype):
			raise Exception( "test failed t z p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ekp2-Ekpz2)) )

		if(kblock not in [0,L_2]):
			if norm(Ekp2-Ekpz1) > Ns*eps(dtype):
				raise Exception( "test failed t z p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ekp2-Ekpz1)) )

			if norm(Ekp1-Ekpz2) > Ns*eps(dtype):
				raise Exception( "test failed t z p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nf={3} {4}".format(L,kblock,np.dtype(dtype),Nf,norm(Ekp1-Ekpz2)) )


#check_t_p_z(2,np.complex128,Nf=(1,1))
check_t_p_z(2,np.complex128)
exit()


def check_pbc(Lmax):

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,1):
			check_t(L,dtype)
			for Nup in range(L+1):
				N_down=L=Nup
				check_t(L,dtype,Nf=(Nup,Ndown))

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,2):
			check_t_z(L,dtype,Nf=(L//2,L//2))
			check_t_z(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,1):
			check_t_p(L,dtype)
			for Nup in range(L+1):
				check_t_p(L,dtype,Nf=(Nup,L-Nup))

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_t_pz(L,dtype,Nf=(L//2,L//2))
			check_t_pz(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_t_p_z(L,dtype,Nf=(L//2,L//2))
			check_t_p_z(L,dtype)



check_m(4)
check_obc(4)
check_obc(8)

print('GET RID OF NO_CHECKS')
print('MAKE SURE OBC COUPLINGS ARE NOT TRANSL INV')
print('RELEASE SEED')	
