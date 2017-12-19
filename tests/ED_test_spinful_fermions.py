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


'''
def check_z(L,dtype,Nf=None):
	
	J1=[[2.0*random()-1.0,i,i] for i in range(L)]
	J0=random()
	J2p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
	J2m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]
	J1p=[[2.0*J0-1.0,i,(i+1)%L] for i in range(L)]
	J1m=[[-(2.0*J0-1.0),i,(i+1)%L] for i in range(L)]

	static=[["z|z",J1],["+-|",J2p],["-+|",J2m],["|+-",J1p],["|-+",J1m]]


	basis=spinful_fermion_basis_1d(L=L,Nf=Nf,check_z_symm=False)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spinful_fermion_basis_1d(L=L,Nf=Nf,sblock=1,check_z_symm=False)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spinful_fermion_basis_1d(L=L,Nf=Nf,sblock=-1,check_z_symm=False)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()

	print(basis)
	print(basis1)
	print(basis2)
	exit()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()

	print(E)
	print(Ez)

	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed z symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf, norm(Ez-E)))

check_z(2,np.float64,Nf=(1,1))
'''




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
	else:
		static=[["z|z",J],["|+",h],["-|",h]]

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

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
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	Et=np.array([])
	for kblock in range(0,L):

		basisk=spinful_fermion_basis_1d(L=L,Nf=Nf,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk)
		Et=np.append(Et,Hk.eigvalsh())
		
	
	Et.sort()
	
	if norm(Et-E) > Ns*eps(dtype):
		raise Exception( "test failed t symmetry at L={0:3d} with dtype {1} and Nf={2} {3}".format(L,np.dtype(dtype),Nf,norm(Et-E)) )


#check_t(4,np.complex128,Nf=(1,3))
#check_p(4,np.complex128)
#exit()


def check_t_p(L,dtype,Nf=None):

	"""

	1) get rid of no_checks

	"""

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

	no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)

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




check_m(4)



print('GET RID OF NO_CHECKS')
print('RELEASE SEED')	
