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


def eps(dtype):
	return 2*10.0**(-5)





def check_opstr(Lmax):
	for dtype in dtypes:
		for L in range(2,Lmax+1):
			h=[[2.0*random()-1.0,i] for i in range(L)]
			J1=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L)]
			J2=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L)]
			J3=[[J2[i][0]*0.5,i,(i+1)%L] for i in range(L)]

			static1=[["zz",J1],["yy",J2],["xx",J2],["x",h]]
			static2=[["zz",J1],["+-",J3],["-+",J3],["x",h]]

			basis=spin_basis_1d(L=L,pauli=False)

			H1=hamiltonian(static1,[],dtype=dtype,basis=basis)
			H2=hamiltonian(static2,[],dtype=dtype,basis=basis)


			if norm(H1.todense()-H2.todense()) > eps(dtype):
				raise Exception( "test failed opstr at L={0:3d} with dtype {1} {2}".format(L,np.dtype(dtype)), norm(H1.todense()-H2.todense()) )






def check_m(Lmax):
	for dtype in dtypes:
		for L in range(2,Lmax+1):
			h=[[2.0*random()-1.0,i] for i in range(L)]
			J1=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L)]
			J2=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L)]

			static=[["zz",J1],["yy",J2],["xx",J2],["z",h]]
			
			basis=spin_basis_1d(L=L,pauli=False)
			H=hamiltonian(static,[],dtype=dtype,basis=basis)
			Ns=H.Ns
			E=H.eigvalsh()

			Em=[]
			for Nup in range(L+1):
				basis=spin_basis_1d(L=L,pauli=False,Nup=Nup)
				H=hamiltonian(static,[],dtype=dtype,basis=basis)
				Etemp=H.eigvalsh()
				Em.append(Etemp)

			Em=np.concatenate(Em)
			Em.sort()
			

			if norm(Em-E) > Ns*eps(dtype):
				raise Exception( "test failed m symmetry at L={0:3d} with dtype {1} {2}".format(L,dtype,norm(Em-E) ) )


def check_z(L,dtype,Nup=None):
	if type(Nup) is int:
		J1=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
		J2=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
		static=[["zz",J1],["yy",J2],["xx",J2]]
	else:
		h=[[2.0*random()-1.0,i] for i in range(L)]
		J1=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
		J2=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
		static=[["zz",J1],["x",h]]

	basis=spin_basis_1d(L=L,Nup=Nup)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spin_basis_1d(L=L,Nup=Nup,zblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spin_basis_1d(L=L,Nup=Nup,zblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()

	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed z symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup, norm(Ez-E)))

def check_zA(L,dtype):
	
	h=[[2.0*random()-1.0,i] for i in range(L)]
	J1=[[2.0*random()-1.0,i,(i+2)%L] for i in range(L-1)]
	J2=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
	static=[["zz",J1],["xx",J2],["x",h]]

	basis=spin_basis_1d(L=L)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spin_basis_1d(L=L,zAblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spin_basis_1d(L=L,zAblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()


	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed zA symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype), norm(Ez-E)))


def check_zB(L,dtype):
	
	h=[[2.0*random()-1.0,i] for i in range(L)]
	J1=[[2.0*random()-1.0,i,(i+2)%L] for i in range(L-1)]
	J2=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
	static=[["zz",J1],["xx",J2],["x",h]]

	basis=spin_basis_1d(L=L)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spin_basis_1d(L=L,zBblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spin_basis_1d(L=L,zBblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()


	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed zB symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype), norm(Ez-E)))


def check_zA_zB(L,dtype):
	
	h=[[2.0*random()-1.0,i] for i in range(L)]
	J1=[[2.0*random()-1.0,i,(i+2)%L] for i in range(L-1)]
	J2=[[2.0*random()-1.0,i,(i+1)%L] for i in range(L-1)]
	static=[["zz",J1],["xx",J2],["x",h]]

	basis=spin_basis_1d(L=L)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spin_basis_1d(L=L,zAblock=+1,zBblock=+1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spin_basis_1d(L=L,zAblock=+1,zBblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)
	basis3=spin_basis_1d(L=L,zAblock=-1,zBblock=+1)
	H3=hamiltonian(static,[],dtype=dtype,basis=basis3)
	basis4=spin_basis_1d(L=L,zAblock=-1,zBblock=-1)
	H4=hamiltonian(static,[],dtype=dtype,basis=basis4)	
	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	E3=H3.eigvalsh()
	E4=H4.eigvalsh()

	Ez=np.concatenate((E1,E2,E3,E4))
	Ez.sort()

	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed zA zB symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype), norm(Ez-E)))



def check_p(L,dtype,Nup=None):
	L_2=int(L/2)
	hr=[2.0*random()-1.0 for i in range(L_2)]
	hi=[hr[i] for i in range(L_2)]
	hi.reverse()
	hi.extend(hr)
	
	h=[[hi[i],i] for i in range(L)]
	J=[[1.0,i,(i+1)%L] for i in range(L-1)]

	if type(Nup) is int:
		static=[["zz",J],["yy",J],["xx",J],["z",h]]
	else:
		static=[["zz",J],["x",h]]

	basis=spin_basis_1d(L=L,Nup=Nup)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spin_basis_1d(L=L,Nup=Nup,pblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spin_basis_1d(L=L,Nup=Nup,pblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ep=np.concatenate((E1,E2))
	Ep.sort()

	if norm(Ep-E) > Ns*eps(dtype):
		raise Exception( "test failed p symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Ep-E)) )




def check_pz(L,dtype,Nup=None):
	L_2=int(L/2)
	hr=[(i+0.1)**2/float(L**2) for i in range(L_2)]
	hi=[-(i+0.1)**2/float(L**2) for i in range(L_2)]
	hi.reverse()
	hi.extend(hr)

	h=[[hi[i],i] for i in range(L)]
	J=[[1.0,i,(i+1)%L] for i in range(L-1)]

	static=[["zz",J],["yy",J],["xx",J],["z",h]]

	basis=spin_basis_1d(L=L,Nup=Nup)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spin_basis_1d(L=L,Nup=Nup,pzblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spin_basis_1d(L=L,Nup=Nup,pzblock=-1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Epz=np.concatenate((E1,E2))
	Epz.sort()


	if norm(Epz-E) > Ns*eps(dtype):
		raise Exception( "test failed pz symmetry at L={0:3d} with dtype {1} and Nup={2:2d} {3}".format(L,np.dtype(dtype),Nup,norm(Epz-E)) )




def check_p_z(L,dtype,Nup=None):
	h=[[1.0,i] for i in range(L)]
	J=[[1.0,i,(i+1)%L] for i in range(L-1)]

	if type(Nup) is int:
		static=[["zz",J],["yy",J],["xx",J]]
	else:
		static=[["zz",J],["x",h]]

	basis=spin_basis_1d(L=L,Nup=Nup)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	basis1=spin_basis_1d(L=L,Nup=Nup,pblock=1,zblock=1)
	H1=hamiltonian(static,[],dtype=dtype,basis=basis1)
	basis2=spin_basis_1d(L=L,Nup=Nup,pblock=-1,zblock=1)
	H2=hamiltonian(static,[],dtype=dtype,basis=basis2)
	basis3=spin_basis_1d(L=L,Nup=Nup,pblock=1,zblock=-1)
	H3=hamiltonian(static,[],dtype=dtype,basis=basis3)
	basis4=spin_basis_1d(L=L,Nup=Nup,pblock=-1,zblock=-1)
	H4=hamiltonian(static,[],dtype=dtype,basis=basis4)
	
	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	E3=H3.eigvalsh()
	E4=H4.eigvalsh()

	
	Epz=np.concatenate((E1,E2,E3,E4))
	Epz.sort()

	if norm(Epz-E) > Ns*eps(dtype):
		raise Exception( "test failed p z symmetry at L={0:3d} with dtype {1} and Nup {2:2d} {3}".format(L,np.dtype(dtype),Nup,norm(Epz-E)) )










def check_obc(Lmax):
	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_z(L,dtype,Nup=int(L/2))
			check_z(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_zA(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_zB(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_zA_zB(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_p(L,dtype,Nup=int(L/2))
			check_p(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_pz(L,dtype,Nup=int(L/2))
			check_pz(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_p_z(L,dtype,Nup=int(L/2))
			check_p_z(L,dtype) 



def check_t(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+1)%L] for i in range(L)]
	if type(Nup) is int:
		static=[["zz",J1],["yy",J1],["xx",J1]]
	else:
		static=[["zz",J1],["x",h]]


	basis=spin_basis_1d(L=L,Nup=Nup)
	H=hamiltonian(static,[],dtype=dtype,basis=basis)
	Ns=H.Ns
	E=H.eigvalsh()

	Et=np.array([])
	for kblock in range(0,L):

		basisk=spin_basis_1d(L=L,Nup=Nup,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk)
		Et=np.append(Et,Hk.eigvalsh())
		
	
	Et.sort()
	
	if norm(Et-E) > Ns*eps(dtype):
		raise Exception( "test failed t symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Et-E)) )





def check_t_z(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+1)%L] for i in range(L)]
	if type(Nup) is int:
		static=[["zz",J1],["yy",J1],["xx",J1]]
	else:
		static=[["zz",J1],["x",h]]

	L_2=int(L/2)

	for kblock in range(-L_2+1,L_2+1):

		basisk=spin_basis_1d(L=L,Nup=Nup,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,zblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,zblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()


		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t z symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Ek-Ekz)) )


def check_t_zA(L,dtype,a=2):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+2)%L] for i in range(L)]

	static=[["zz",J1],["x",h]]

	L_2=int(L/a)

	for kblock in range(-L_2+2,L_2+2):

		basisk=spin_basis_1d(L=L,kblock=kblock,a=a)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spin_basis_1d(L=L,kblock=kblock,a=a,zAblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spin_basis_1d(L=L,kblock=kblock,a=a,zAblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()

		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t zA symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype),norm(Ek-Ekz)) )


def check_t_zB(L,dtype,a=2):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+2)%L] for i in range(L)]

	static=[["zz",J1],["x",h]]

	L_2=int(L/a)

	for kblock in range(-L_2+2,L_2+2):

		basisk=spin_basis_1d(L=L,kblock=kblock,a=a)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spin_basis_1d(L=L,kblock=kblock,a=a,zBblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spin_basis_1d(L=L,kblock=kblock,a=a,zBblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()

		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t zB symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype),norm(Ek-Ekz)) )


def check_t_zA_zB(L,dtype,a=2):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+2)%L] for i in range(L)]

	static=[["zz",J1],["x",h]]
	
	L_2=int(L/a)

	for kblock in range(0,L_2):

		basisk=spin_basis_1d(L=L,kblock=kblock,a=a)
		Hk=hamiltonian(static,[],dtype=dtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spin_basis_1d(L=L,kblock=kblock,a=a,zAblock=+1,zBblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spin_basis_1d(L=L,kblock=kblock,a=a,zAblock=+1,zBblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)
		basisk3=spin_basis_1d(L=L,kblock=kblock,a=a,zAblock=-1,zBblock=+1)
		Hk3=hamiltonian(static,[],dtype=dtype,basis=basisk3)
		basisk4=spin_basis_1d(L=L,kblock=kblock,a=a,zAblock=-1,zBblock=-1)
		Hk4=hamiltonian(static,[],dtype=dtype,basis=basisk4)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ek3=Hk3.eigvalsh()
		Ek4=Hk4.eigvalsh()
		Ekz=np.concatenate((Ek1,Ek2,Ek3,Ek4))
		Ekz.sort()


		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t zA zB symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype),norm(Ek-Ekz)) )

def check_t_p(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+1)%L] for i in range(L)]
	if type(Nup) is int:
		static=[["zz",J1],["xx",J1],["yy",J1]] # 
	else:
		static=[["zz",J1],["x",h]]

	L_2=int(L/2)

	if dtype is np.float32:
		kdtype = np.complex64
	elif dtype is np.float64:
		kdtype = np.complex128
	else:
		kdtype = dtype
		

	for kblock in range(-L_2+1,0):

		basisk=spin_basis_1d(L=L,kblock=kblock)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spin_basis_1d(L=L,kblock=kblock,pblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
		basisk2=spin_basis_1d(L=L,kblock=kblock,pblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		
		if norm(Ek-Ek1) > Ns*eps(dtype):
			raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )

		if norm(Ek-Ek2) > Ns*eps(dtype):
			raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )


	basisk=spin_basis_1d(L=L,Nup=Nup,kblock=0)
	Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=0,pblock=+1)
	Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
	basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=0,pblock=-1)
	Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)

	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps(dtype):
			raise Exception( "test failed t p symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,0,np.dtype(dtype),Nup,norm(Ek-Ekp)) )


	if L%2 == 0:	
		for kblock in range(1,L_2):

			basisk=spin_basis_1d(L=L,Nup=Nup,kblock=kblock)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
			basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )

		basisk=spin_basis_1d(L=L,Nup=Nup,kblock=L_2)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=L_2,pblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=L_2,pblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps(dtype):
				raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,int(L/2),np.dtype(dtype),Nup,norm(Ek-Ekp)) )

	else:
		for kblock in range(1,L_2+1):
			
			basisk=spin_basis_1d(L=L,Nup=Nup,kblock=kblock)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
			basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )
	








def check_t_pz(L,dtype,Nup=None):
	hx=random()*0.0
	hz=random()*0.0
	J=random()
	h1=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+1)%L] for i in range(L)]
	h2=[[hz*(-1)**i,i] for i in range(L)]

	if type(Nup) is int:
		static=[["zz",J1],["xx",J1],["yy",J1],['z',h2]] 
	else:
		static=[["x",h1],['z',h2],['zz',J1]]

	if dtype is np.float32:
		kdtype = np.complex64
	elif dtype is np.float64:
		kdtype = np.complex128
	else:
		kdtype = dtype

	
	a=2
	L_2=int(L/(a*2))
	for kblock in range(-L_2+1,0):

		basisk=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a,pzblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a,pzblock=-1) 
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		if norm(Ek-Ek1) > Ns*eps(dtype):
			raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )

		if norm(Ek-Ek2) > Ns*eps(dtype):
			raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )

	basisk=spin_basis_1d(L=L,Nup=Nup,kblock=0,a=a)
	Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=0,a=a,pzblock=+1)
	Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
	basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=0,a=a,pzblock=-1)
	Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)
	
	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps(dtype):
			raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,0,np.dtype(dtype),Nup,norm(Ek-Ekp)) )

	if((L/a)%2 == 0):
		for kblock in range(1,L_2):

			basisk=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a,pzblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
			
			basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a,pzblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )

		basisk=spin_basis_1d(L=L,Nup=Nup,kblock=L_2,a=a)
		Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=L_2,a=a,pzblock=+1)
		Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=L_2,a=a,pzblock=-1)
		Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps(dtype):
				raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,int(L/2),np.dtype(dtype),Nup,norm(Ek-Ekp)) )
	else:
		for kblock in range(1,L_2+1):

			basisk=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a)
			Hk=hamiltonian(static,[],dtype=kdtype,basis=basisk)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a,pzblock=+1)
			Hk1=hamiltonian(static,[],dtype=dtype,basis=basisk1) 
			basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,a=a,pzblock=-1)
			Hk2=hamiltonian(static,[],dtype=dtype,basis=basisk2)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )





def check_t_p_z(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in range(L)]
	J1=[[J,i,(i+1)%L] for i in range(L)]
	if type(Nup) is int:
		static=[["zz",J1],["xx",J1],["yy",J1]] 
	else:
		static=[["zz",J1],["x",h]]

	
	L_2=int(L/2)
	for kblock in range(-L_2+1,L_2+1):

		basisk1=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=+1)
		Hkp1=hamiltonian(static,[],dtype=dtype,basis=basisk1)
		basisk2=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=-1)
		Hkp2=hamiltonian(static,[],dtype=dtype,basis=basisk2)
		Ns=Hkp1.Ns
		Ekp1=Hkp1.eigvalsh()
		Ekp2=Hkp2.eigvalsh()

		basisk11=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=+1,zblock=+1)
		Hkpz11=hamiltonian(static,[],dtype=dtype,basis=basisk11)
		basisk12=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=+1,zblock=-1) 
		Hkpz12=hamiltonian(static,[],dtype=dtype,basis=basisk12)	
		Ekpz11=Hkpz11.eigvalsh()
		Ekpz12=Hkpz12.eigvalsh()

		Ekpz1=np.concatenate((Ekpz11,Ekpz12))
		Ekpz1.sort()

		basisk21=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=-1,zblock=+1)
		Hkpz21=hamiltonian(static,[],dtype=dtype,basis=basisk21)
		basisk22=spin_basis_1d(L=L,Nup=Nup,kblock=kblock,pblock=-1,zblock=-1) 
		Hkpz22=hamiltonian(static,[],dtype=dtype,basis=basisk22)	
		Ekpz21=Hkpz21.eigvalsh()
		Ekpz22=Hkpz22.eigvalsh()

		Ekpz2=np.concatenate((Ekpz21,Ekpz22))
		Ekpz2.sort()
			
		if norm(Ekp1-Ekpz1) > Ns*eps(dtype):
			raise Exception( "test failed t z p+  symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ekp1-Ekpz1)) )

		if norm(Ekp2-Ekpz2) > Ns*eps(dtype):
			raise Exception( "test failed t z p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ekp2-Ekpz2)) )

		if(kblock not in [0,L_2]):
			if norm(Ekp2-Ekpz1) > Ns*eps(dtype):
				raise Exception( "test failed t z p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ekp2-Ekpz1)) )

			if norm(Ekp1-Ekpz2) > Ns*eps(dtype):
				raise Exception( "test failed t z p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ekp1-Ekpz2)) )


def check_pbc(Lmax):

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,1):
			check_t(L,dtype)
			for Nup in range(L+1):
				check_t(L,dtype,Nup=Nup)

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,2):
			check_t_z(L,dtype,Nup=int(L/2))
			check_t_z(L,dtype)

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,2):
			check_t_zA(L,dtype)

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,2):
			check_t_zB(L,dtype)

	for dtype in (np.complex64,np.complex128):
		for L in range(2,Lmax+1,2):
			check_t_zA_zB(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,1):
			check_t_p(L,dtype)
			for Nup in range(L+1):
				check_t_p(L,dtype,Nup=Nup)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_t_pz(L,dtype,Nup=int(L/2))
			check_t_pz(L,dtype)

	for dtype in dtypes:
		for L in range(2,Lmax+1,2):
			check_t_p_z(L,dtype,Nup=int(L/2))
			check_t_p_z(L,dtype) 


		

check_m(4)
check_opstr(4)
check_obc(8)
check_pbc(8)





