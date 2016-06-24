from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import basis1d
import numpy as np
import scipy.sparse as sm
from numpy.linalg import norm
from numpy.random import random,seed

seed()
dtypes=[np.float32,np.float64,np.complex64,np.complex128]


def eps(dtype):
	return 2*10.0**(-5)


def J(L,jb,l):
	blist=[]
	for i,j in jb:
		b=[j]
		b.extend([(i+j)%L for j in xrange(l)])
		blist.append(b)

	return blist


def check_opstr(Lmax):
	for dtype in dtypes:
		for L in xrange(2,Lmax+1):
			h=[[2.0*random()-1.0,i] for i in xrange(L)]
			J1=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]
			J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]
			J3=[[J2[i][0]*0.5,i,(i+1)%L] for i in xrange(L)]

			static1=[["zz",J1],["yy",J2],["xx",J2],["x",h]]
			static2=[["zz",J1],["+-",J3],["-+",J3],["x",h]]

			H1=hamiltonian(static1,[],L=L,dtype=dtype,pauli=False)
			H2=hamiltonian(static2,[],L=L,dtype=dtype,pauli=False)


			if norm(H1.todense()-H2.todense()) > eps(dtype):
				raise Exception( "test failed opstr at L={0:3d} with dtype {1} {2}".format(L,np.dtype(dtype)), norm(H1.todense()-H2.todense()) )






def check_m(Lmax):
	for dtype in dtypes:
		for L in xrange(2,Lmax+1):
			h=[[2.0*random()-1.0,i] for i in xrange(L)]
			J1=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]
			J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]

			static=[["zz",J1],["yy",J2],["xx",J2],["z",h]]

			H=hamiltonian(static,[],L=L,dtype=dtype,pauli=False)
			Ns=H.Ns
			E=H.eigvalsh()

			Em=[]
			for Nup in xrange(L+1):
				H=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,pauli=False)
				Etemp=H.eigvalsh()
				Em.append(Etemp)

			Em=np.concatenate(Em)
			Em.sort()
			

			if norm(Em-E) > Ns*eps(dtype):
				raise Exception( "test failed m symmetry at L={0:3d} with dtype {1} {2}".format(L,dtype,norm(Em-E) ) )


def check_z(L,dtype,Nup=None):
	if type(Nup) is int:
		J1=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L-1)]
		J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L-1)]
		static=[["zz",J1],["yy",J2],["xx",J2]]
	else:
		h=[[2.0*random()-1.0,i] for i in xrange(L)]
		J1=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L-1)]
		J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L-1)]
		static=[["zz",J1],["x",h]]


	

	H=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L=L,Nup=Nup,zblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L=L,Nup=Nup,zblock=-1,dtype=dtype)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()


	if norm(Ez-E) > Ns*eps(dtype):
		raise Exception( "test failed z symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup, norm(Ez-E)))

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
		raise Exception( "test failed zA symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype), norm(Ez-E)))


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
		raise Exception( "test failed zB symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype), norm(Ez-E)))


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
		raise Exception( "test failed zA zB symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype), norm(Ez-E)))



def check_p(L,dtype,Nup=None):
	L_2=int(L/2)
	hr=[2.0*random()-1.0 for i in xrange(L_2)]
	hi=[hr[i] for i in xrange(L_2)]
	hi.reverse()
	hi.extend(hr)
	
	h=[[hi[i],i] for i in xrange(L)]
	J=[[1.0,i,(i+1)%L] for i in xrange(L-1)]

	if type(Nup) is int:
		static=[["zz",J],["yy",J],["xx",J],["z",h]]
	else:
		static=[["zz",J],["x",h]]

	H=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L=L,Nup=Nup,pblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L=L,Nup=Nup,pblock=-1,dtype=dtype)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ep=np.concatenate((E1,E2))
	Ep.sort()

	if norm(Ep-E) > Ns*eps(dtype):
		raise Exception( "test failed p symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Ep-E)) )




def check_pz(L,dtype,Nup=None):
	L_2=int(L/2)
	hr=[(i+0.1)**2/float(L**2) for i in xrange(L_2)]
	hi=[-(i+0.1)**2/float(L**2) for i in xrange(L_2)]
	hi.reverse()
	hi.extend(hr)

	h=[[hi[i],i] for i in xrange(L)]
	J=[[1.0,i,(i+1)%L] for i in xrange(L-1)]

	static=[["zz",J],["yy",J],["xx",J],["z",h]]

	H=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L=L,Nup=Nup,pzblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L=L,Nup=Nup,pzblock=-1,dtype=dtype)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Epz=np.concatenate((E1,E2))
	Epz.sort()


	if norm(Epz-E) > Ns*eps(dtype):
		raise Exception( "test failed pz symmetry at L={0:3d} with dtype {1} and Nup={2:2d} {3}".format(L,np.dtype(dtype),Nup,norm(Epz-E)) )




def check_p_z(L,dtype,Nup=None):
	h=[[1.0,i] for i in xrange(L)]
	J=[[1.0,i,(i+1)%L] for i in xrange(L-1)]

	if type(Nup) is int:
		static=[["zz",J],["yy",J],["xx",J],["z",h]]
	else:
		static=[["zz",J],["x",h]]

	H=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L=L,Nup=Nup,pblock=1,zblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L=L,Nup=Nup,pblock=-1,zblock=1,dtype=dtype)
	H3=hamiltonian(static,[],L=L,Nup=Nup,pblock=1,zblock=-1,dtype=dtype)
	H4=hamiltonian(static,[],L=L,Nup=Nup,pblock=-1,zblock=-1,dtype=dtype)
	
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
		for L in xrange(2,Lmax+1,2):
			check_z(L,dtype,Nup=int(L/2))
			check_z(L,dtype)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,2):
			check_zA(L,dtype)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,2):
			check_zB(L,dtype)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,2):
			check_zA_zB(L,dtype)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,2):
			check_p(L,dtype,Nup=int(L/2))
			check_p(L,dtype)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,2):
			check_pz(L,dtype,Nup=int(L/2))
			check_pz(L,dtype)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,2):
			check_p_z(L,dtype,Nup=int(L/2))
			check_p_z(L,dtype) 



def check_t(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+1)%L] for i in xrange(L)]
	if type(Nup) is int:
		static=[["zz",J1],["yy",J1],["xx",J1]]
	else:
		static=[["zz",J1],["x",h]]


	
	H=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	Et=np.array([])
	for kblock in xrange(0,L):
		Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock)
		Et=np.append(Et,Hk.eigvalsh())
		
	
	Et.sort()
	
	if norm(Et-E) > Ns*eps(dtype):
		raise Exception( "test failed t symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Et-E)) )





def check_t_z(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+1)%L] for i in xrange(L)]
	if type(Nup) is int:
		static=[["zz",J1],["yy",J1],["xx",J1]]
	else:
		static=[["zz",J1],["x",h]]

	L_2=int(L/2)

	for kblock in xrange(-L_2+1,L_2+1):
		Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,zblock=+1)
		Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,zblock=-1)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()


		if norm(Ek-Ekz) > Ns*eps(dtype):
			raise Exception( "test failed t z symmetry at L={0:3d} with dtype {1} and Nup={2} {3}".format(L,np.dtype(dtype),Nup,norm(Ek-Ekz)) )


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
			raise Exception( "test failed t zA symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype),norm(Ek-Ekz)) )


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
			raise Exception( "test failed t zB symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype),norm(Ek-Ekz)) )


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
			raise Exception( "test failed t zA zB symmetry at L={0:3d} with dtype {1} and {2}".format(L,np.dtype(dtype),norm(Ek-Ekz)) )

def check_t_p(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+1)%L] for i in xrange(L)]
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
		

	for kblock in xrange(-L_2+1,0):
		Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=kblock)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1) 

		Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		
		if norm(Ek-Ek1) > Ns*eps(dtype):
			raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )

		if norm(Ek-Ek2) > Ns*eps(dtype):
			raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )

	Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=0)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=0,pblock=+1)
	Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=0,pblock=-1)	
	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps(dtype):
			raise Exception( "test failed t p symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,0,np.dtype(dtype),Nup,norm(Ek-Ekp)) )


	if L%2 == 0:	
		for kblock in xrange(1,L_2):
			Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=kblock)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1) 
	
			Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )

	
		Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=L_2)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=L_2,pblock=+1)
		Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=L_2,pblock=-1)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps(dtype):
				raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,int(L/2),np.dtype(dtype),Nup,norm(Ek-Ekp)) )

	else:
		for kblock in xrange(1,L_2+1):
			Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=kblock)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1) 
	
			Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1)	
	
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
	h1=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+1)%L] for i in xrange(L)]
	h2=[[hz*(-1)**i,i] for i in xrange(L)]

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
	for kblock in xrange(-L_2+1,0):
		Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=kblock,a=a)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=+1,a=a) 
		Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=-1,a=a)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		if norm(Ek-Ek1) > Ns*eps(dtype):
			raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )

		if norm(Ek-Ek2) > Ns*eps(dtype):
			raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )

	Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=0,a=a)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=0,pzblock=+1,a=a)
	Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=0,pzblock=-1,a=a)
	
	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps(dtype):
			raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,0,np.dtype(dtype),Nup,norm(Ek-Ekp)) )

	if((L/a)%2 == 0):
		for kblock in xrange(1,L_2):
			Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=kblock,a=a)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=+1,a=a) 
	
			Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=-1,a=a)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )

		Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=L_2,a=a)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=L_2,pzblock=+1,a=a)
		Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=L_2,pzblock=-1,a=a)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps(dtype):
				raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,int(L/2),np.dtype(dtype),Nup,norm(Ek-Ekp)) )
	else:
		for kblock in xrange(1,L_2+1):
			Hk=hamiltonian(static,[],L=L,Nup=Nup,dtype=kdtype,kblock=kblock,a=a)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			Hk1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=+1,a=a) 
	
			Hk2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=-1,a=a)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps(dtype):
				raise Exception( "test failed t pz+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek1)) )
	
			if norm(Ek-Ek2) > Ns*eps(dtype):
				raise Exception( "test failed t pz- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3} {4}".format(L,kblock,np.dtype(dtype),Nup,norm(Ek-Ek2)) )





def check_t_p_z(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+1)%L] for i in xrange(L)]
	if type(Nup) is int:
		static=[["zz",J1],["xx",J1],["yy",J1]] 
	else:
		static=[["zz",J1],["x",h]]

	
	L_2=int(L/2)
	for kblock in xrange(-L_2+1,L_2+1):
		Hkp1=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1)
		Hkp2=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1)
		Ns=Hkp1.Ns
		Ekp1=Hkp1.eigvalsh()
		Ekp2=Hkp2.eigvalsh()

		Hkpz11=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1,zblock=+1) 
		Hkpz12=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1,zblock=-1)	
		Ekpz11=Hkpz11.eigvalsh()
		Ekpz12=Hkpz12.eigvalsh()

		Ekpz1=np.concatenate((Ekpz11,Ekpz12))
		Ekpz1.sort()

		Hkpz21=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1,zblock=+1) 
		Hkpz22=hamiltonian(static,[],L=L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1,zblock=-1)	
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
		for L in xrange(2,Lmax+1,1):
			check_t(L,dtype)
			for Nup in xrange(L+1):
				check_t(L,dtype,Nup=Nup)

	for dtype in (np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_t_z(L,dtype,Nup=int(L/2))
			check_t_z(L,dtype)

	for dtype in (np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_t_zA(L,dtype)

	for dtype in (np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_t_zB(L,dtype)

	for dtype in (np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_t_zA_zB(L,dtype)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,1):
			check_t_p(L,dtype)
			for Nup in xrange(L+1):
				check_t_p(L,dtype,Nup=Nup)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,2):
			check_t_pz(L,dtype,Nup=int(L/2))
			check_t_pz(L,dtype)

	for dtype in dtypes:
		for L in xrange(2,Lmax+1,2):
			check_t_p_z(L,dtype,Nup=int(L/2))
			check_t_p_z(L,dtype) 


		

def getvec(L,Nup=None,kblock=None,pblock=None,zblock=None,pzblock=None,a=1,sparse=True):
	jb=[[i,1.0] for i in xrange(L)]
	dtype=np.complex128

	b = basis1d(L,Nup=Nup,kblock=kblock,pblock=pblock,zblock=zblock,pzblock=pzblock,a=a)
	Ns = b.Ns

#	bits=" ".join(["{"+str(i)+":0"+str(L)+"b}" for i in xrange(len(b.basis))])
#	norms = b.get_norms(np.float64)
	
#	static = [['xxzz',J3],['zzxx',J3],['yyzz',J3],['zzyy',J3],['xxyy',J3],['yyxx',J3]]

	static = [['xx',J(L,jb,2)],
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

	H1 = hamiltonian(static,[],L=L,dtype=dtype)
	H2 = hamiltonian(static,[],L=L,basis=b,dtype=dtype)


	E,v0=H2.eigh()
	v = b.get_vec(v0,sparse=sparse)

	if sm.issparse(v):
		v = v.todense()

	H1 = H1.todense()
	H2 = H2.todense()
	H2 = v0.T.conj() * (H2 * v0)

	if v.shape[0] != 0:
		H1 = v.T.conj() * ( H1 * v)
		if np.abs(np.linalg.norm(H1-H2)) > 10**(-10):
#			print "failed"
			raise Exception("get_vec() failed {0}")
	else: 
		pass	



def check_getvec(L,a=1,sparse=True):
	for k in xrange(-L/a,L/a):
			getvec(L,kblock=k,a=a,sparse=sparse)

	for j in xrange(-1,2,2):
		getvec(L,pblock=j,a=a,sparse=sparse)
		for k in xrange(-L/a,L/a):
			getvec(L,kblock=k,pblock=j,a=a,sparse=sparse)

	Nup=None

	for i in xrange(-1,2,2):
		for j in xrange(-1,2,2):
			getvec(L,Nup=Nup,pblock=i,zblock=j,a=a,sparse=sparse)
			for k in xrange(-L/a,L/a):
				getvec(L,kblock=k,Nup=Nup,pblock=i,zblock=j,a=a,sparse=sparse)

	for j in xrange(-1,2,2):
			getvec(L,Nup=Nup,pzblock=j,a=a,sparse=sparse)
			for k in xrange(-L/a,L/a):
				 getvec(L,kblock=k,Nup=Nup,pzblock=j,a=a,sparse=sparse)

	for j in xrange(-1,2,2):
		getvec(L,Nup=Nup,zblock=j,a=a)
		for k in xrange(-L/a,L/a):
			getvec(L,kblock=k,Nup=Nup,zblock=j,a=a,sparse=sparse)

	for Nup in xrange(L+1):
		for k in xrange(-L/a,L/a):
				getvec(L,Nup=Nup,kblock=k,a=a,sparse=sparse)

	for Nup in xrange(0,L+1):
		for j in xrange(-1,2,2):
			getvec(L,Nup=Nup,pblock=j,a=a,sparse=sparse)
			for k in xrange(-L/a,L/a):
				getvec(L,kblock=k,Nup=Nup,pblock=j,a=a,sparse=sparse)

	if (L%2)==0:
		Nup=L/2
		for i in xrange(-1,2,2):
			for j in xrange(-1,2,2):
				getvec(L,Nup=Nup,pblock=i,zblock=j,a=a,sparse=sparse)
				for k in xrange(-L/a,L/a):
					getvec(L,kblock=k,Nup=Nup,pblock=i,zblock=j,a=a,sparse=sparse)

		for j in xrange(-1,2,2):
				getvec(L,Nup=Nup,pzblock=j,a=a,sparse=sparse)
				for k in xrange(-L/a,L/a):
					getvec(L,kblock=k,Nup=Nup,pzblock=j,a=a,sparse=sparse)

		for j in xrange(-1,2,2):
			getvec(L,Nup=Nup,zblock=j,a=a,sparse=sparse)
			for k in xrange(-L/a,L/a):
				getvec(L,kblock=k,Nup=Nup,zblock=j,a=a,sparse=sparse)




#check_m(4)
#check_opstr(4)
#check_obc(8)
#check_pbc(8)
#for L in xrange(4,8):
#	check_getvec(L,sparse=True)
#	check_getvec(L,sparse=False)








