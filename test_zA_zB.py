from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.hamiltonian import supported_dtypes
from exact_diag_py.basis import basis1d
import numpy as np
import scipy.sparse as sm
from numpy.linalg import norm
from numpy.random import random,seed

seed()

def eps(dtype):
	return 1.15*10.0**(-5)

def check_zA(L,dtype):
	
	h=[[2.0*random()-1.0,i] for i in xrange(L)]
	J1=[[2.0*random()-1.0,i,(i+2)%L] for i in xrange(L-1)]
	J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L-1)]
	static=[["zz",J1],["xx",J2],["x",h]]


	H=hamiltonian(static,[],L=L,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L=L,zAblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L=L,zAblock=-1,dtype=dtype)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()


	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed zA symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup, norm(Ez-E)))


def check_zB(L,dtype):
	
	h=[[2.0*random()-1.0,i] for i in xrange(L)]
	J1=[[2.0*random()-1.0,i,(i+2)%L] for i in xrange(L-1)]
	J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L-1)]
	static=[["zz",J1],["xx",J2],["x",h]]


	H=hamiltonian(static,[],L=L,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L=L,zBblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L=L,zBblock=-1,dtype=dtype)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()


	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed zB symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup, norm(Ez-E)))


def check_zA_zB(L,dtype):
	
	h=[[2.0*random()-1.0,i] for i in xrange(L)]
	J1=[[2.0*random()-1.0,i,(i+2)%L] for i in xrange(L-1)]
	J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L-1)]
	static=[["zz",J1],["xx",J2],["x",h]]


	H=hamiltonian(static,[],L=L,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L=L,dtype=dtype,zAblock=+1,zBblock=+1)
	H2=hamiltonian(static,[],L=L,dtype=dtype,zAblock=+1,zBblock=-1)
	H3=hamiltonian(static,[],L=L,dtype=dtype,zAblock=-1,zBblock=+1)
	H4=hamiltonian(static,[],L=L,dtype=dtype,zAblock=-1,zBblock=-1)	
	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	E3=H3.eigvalsh()
	E4=H4.eigvalsh()

	Ez=np.concatenate((E1,E2,E3,E4))
	Ez.sort()

	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed zA zB symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup, norm(Ez-E)))

def check_t_zA(L,dtype,a=2):
	hx=random()
	J=random()
	h=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+2)%L] for i in xrange(L)]

	static=[["zz",J1],["x",h]]

	L_2=int(L/a)

	for kblock in xrange(-L_2+2,L_2+2):

		Hk=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,a=2)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,zAblock=+1,a=a)
		Hk2=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,zAblock=-1,a=a)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()

		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t zA symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Ek-Ekz)) )


def check_t_zB(L,dtype,a=2):
	hx=random()
	J=random()
	h=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+2)%L] for i in xrange(L)]

	static=[["zz",J1],["x",h]]

	L_2=int(L/a)

	for kblock in xrange(-L_2+2,L_2+2):
		#print kblock
		Hk=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,a=2)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,zBblock=+1,a=a)
		Hk2=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,zBblock=-1,a=a)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()

		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t zB symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Ek-Ekz)) )


def check_t_zA_zB(L,dtype,a=2):
	hx=random()
	J=random()
	h=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+2)%L] for i in xrange(L)]

	static=[["zz",J1],["x",h]]
	
	L_2=int(L/a)

	for kblock in xrange(0,L_2):

		Hk=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,a=a)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,zAblock=+1,zBblock=+1,a=a)
		Hk2=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,zAblock=+1,zBblock=-1,a=a)
		Hk3=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,zAblock=-1,zBblock=+1,a=a)
		Hk4=hamiltonian(static,[],L=L,dtype=dtype,kblock=kblock,zAblock=-1,zBblock=-1,a=a)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ek3=Hk3.eigvalsh()
		Ek4=Hk4.eigvalsh()
		Ekz=np.concatenate((Ek1,Ek2,Ek3,Ek4))
		Ekz.sort()


		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t zA zB symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Ek-Ekz)) )


check_zA(8,np.complex128)
check_zA_zB(8,np.complex128)
check_zB(8,np.complex128)
check_t_zA(8,np.complex128)
check_t_zB(8,np.complex128)
check_t_zA_zB(8,np.complex128)

