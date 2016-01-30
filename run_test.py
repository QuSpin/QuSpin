from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import basis1d
import numpy as np
from numpy.linalg import norm
from numpy.random import random,seed

seed()


L=32

print "making basis"
b = basis1d(L,Nup=L/2,kblock=0,pblock=+1,zblock=+1)

h=[[2.0*random()-1.0,i] for i in xrange(L)]
J1=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]
J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]
J3=[[J2[i][0]*0.5,i,(i+1)%L] for i in xrange(L)]

static1=[["zz",J1],["yy",J2],["xx",J2],["x",h]]
print "making hamiltonian"
H = hamiltonian(static1,[],L,dtype=np.float32,basis=b)





def check_opstr(Lmax):
	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,Lmax+1):
			h=[[2.0*random()-1.0,i] for i in xrange(L)]
			J1=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]
			J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]
			J3=[[J2[i][0]*0.5,i,(i+1)%L] for i in xrange(L)]

			static1=[["zz",J1],["yy",J2],["xx",J2],["x",h]]
			static2=[["zz",J1],["+-",J3],["-+",J3],["x",h]]

			eps=np.finfo(dtype).eps

			H1=hamiltonian(static1,[],L,dtype=dtype,pauli=False)
			H2=hamiltonian(static2,[],L,dtype=dtype,pauli=False)


			if norm(H1.todense()-H2.todense()) > eps:
				raise Exception( "test failed opstr at L={0:3d} with dtype {1}".format(L,np.dtype(dtype)) )






def check_m(Lmax):
	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,Lmax+1):
			h=[[2.0*random()-1.0,i] for i in xrange(L)]
			J1=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]
			J2=[[2.0*random()-1.0,i,(i+1)%L] for i in xrange(L)]

			static=[["zz",J1],["yy",J2],["xx",J2],["z",h]]

			H=hamiltonian(static,[],L,dtype=dtype,pauli=False)
			Ns=H.Ns
			E=H.eigvalsh()

			Em=[]
			for Nup in xrange(L+1):
				H=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,pauli=False)
				Etemp=H.eigvalsh()
				Em.append(Etemp)

			Em=np.concatenate(Em)
			Em.sort()
			
			eps=np.finfo(dtype).eps
			if norm(Em-E) > Ns*100*eps:
				raise Exception( "test failed m symmetry at L={0:3d} with dtype {1}".format(L,np.dtype(dtype)) )


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


	

	H=hamiltonian(static,[],L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L,Nup=Nup,zblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L,Nup=Nup,zblock=-1,dtype=dtype)


	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ez=np.concatenate((E1,E2))
	Ez.sort()

	eps=np.finfo(dtype).eps
	if norm(Ez-E) > Ns*100*eps:
		raise Exception( "test failed z symmetry at L={0:3d} with dtype {1} and Nup={2}".format(L,np.dtype(dtype),Nup) )



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

	H=hamiltonian(static,[],L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L,Nup=Nup,pblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L,Nup=Nup,pblock=-1,dtype=dtype)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Ep=np.concatenate((E1,E2))
	Ep.sort()

	eps=np.finfo(dtype).eps
	if norm(Ep-E) > Ns*100*eps:
		raise Exception( "test failed p symmetry at L={0:3d} with dtype {1} and Nup={2}".format(L,np.dtype(dtype),Nup) )




def check_pz(L,dtype,Nup=None):
	L_2=int(L/2)
	hr=[(i+0.1)**2/float(L**2) for i in xrange(L_2)]
	hi=[-(i+0.1)**2/float(L**2) for i in xrange(L_2)]
	hi.reverse()
	hi.extend(hr)

	h=[[hi[i],i] for i in xrange(L)]
	J=[[1.0,i,(i+1)%L] for i in xrange(L-1)]

	static=[["zz",J],["yy",J],["xx",J],["z",h]]

	H=hamiltonian(static,[],L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L,Nup=Nup,pzblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L,Nup=Nup,pzblock=-1,dtype=dtype)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	
	Epz=np.concatenate((E1,E2))
	Epz.sort()


	eps=np.finfo(dtype).eps
	if norm(Epz-E) > Ns*100*eps:
		raise Exception( "test failed pz symmetry at L={0:3d} with dtype {1} and Nup={2:2d}".format(L,np.dtype(dtype),Nup) )




def check_p_z(L,dtype,Nup=None):
	h=[[1.0,i] for i in xrange(L)]
	J=[[1.0,i,(i+1)%L] for i in xrange(L-1)]

	if type(Nup) is int:
		static=[["zz",J],["yy",J],["xx",J],["z",h]]
	else:
		static=[["zz",J],["x",h]]

	H=hamiltonian(static,[],L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	H1=hamiltonian(static,[],L,Nup=Nup,pblock=1,zblock=1,dtype=dtype)
	H2=hamiltonian(static,[],L,Nup=Nup,pblock=-1,zblock=1,dtype=dtype)
	H3=hamiltonian(static,[],L,Nup=Nup,pblock=1,zblock=-1,dtype=dtype)
	H4=hamiltonian(static,[],L,Nup=Nup,pblock=-1,zblock=-1,dtype=dtype)

	E1=H1.eigvalsh()
	E2=H2.eigvalsh()
	E3=H3.eigvalsh()
	E4=H4.eigvalsh()

	
	Epz=np.concatenate((E1,E2,E3,E4))
	Epz.sort()

	eps=np.finfo(dtype).eps

	if norm(Epz-E) > Ns*100*eps:
		raise Exception( "test failed p z symmetry at L={0:3d} with dtype {1} and Nup {2:2d}".format(L,np.dtype(dtype),Nup) )










def check_obc(Lmax):
	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_z(L,dtype,Nup=int(L/2))
			check_z(L,dtype)

	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_p(L,dtype,Nup=int(L/2))
			check_p(L,dtype)

	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_pz(L,dtype,Nup=int(L/2))
			check_pz(L,dtype)

	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
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


	
	H=hamiltonian(static,[],L,Nup=Nup,dtype=dtype)
	Ns=H.Ns
	E=H.eigvalsh()

	Et=np.array([])
	for kblock in xrange(0,L):
		Hk=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock)
		Et=np.append(Et,Hk.eigvalsh())
		
	
	Et.sort()
	

	eps=np.finfo(dtype).eps
	if norm(Et-E) > Ns*10*eps:
		raise Exception( "test failed t symmetry at L={0:3d} with dtype {1} and Nup={2}".format(L,np.dtype(dtype),Nup) )





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
	eps=np.finfo(dtype).eps

	for kblock in xrange(-L_2+1,L_2+1):
		Hk=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,zblock=+1)
		Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,zblock=-1)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekz=np.append(Ek1,Ek2)
		Ekz.sort()


		if norm(Ek-Ekz) > Ns*10*eps:
			raise Exception( "test failed t z symmetry at L={0:3d} with dtype {1} and Nup={2}".format(L,np.dtype(dtype),Nup) )




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
	eps=20*np.finfo(dtype).eps

	if dtype is np.float32:
		kdtype = np.complex64
	elif dtype is np.float64:
		kdtype = np.complex128
	else:
		kdtype = dtype
		

	for kblock in xrange(-L_2+1,0):
		Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=kblock)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1) 

		Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1)	

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		
		if norm(Ek-Ek1) > Ns*eps:
			raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

		if norm(Ek-Ek2) > Ns*eps:
			raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

	Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=0)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=0,pblock=+1)
	Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=0,pblock=-1)	
	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps:
			raise Exception( "test failed t p symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,0,np.dtype(dtype),Nup) )


	if L%2 == 0:	
		for kblock in xrange(1,L_2):
			Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=kblock)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1) 
	
			Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps:
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )
	
			if norm(Ek-Ek2) > Ns*eps:
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

	
		Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=L_2)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=L_2,pblock=+1)
		Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=L_2,pblock=-1)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if norm(Ek-Ekp) > Ns*eps:
				raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,int(L/2),np.dtype(dtype),Nup) )

	else:
		for kblock in xrange(1,L_2+1):
			Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=kblock)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1) 
	
			Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps:
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )
	
			if norm(Ek-Ek2) > Ns*eps:
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )
	








def check_t_pz(L,dtype,Nup=None):
	hx=random()
	hz=random()
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

	eps=20*np.finfo(dtype).eps
	
	a=2
	L_2=int(L/(a*2))
	for kblock in xrange(-L_2+1,0):
		Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=kblock,a=a)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=+1,a=a) 
		Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=-1,a=a)

		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		if norm(Ek-Ek1) > Ns*eps:
			raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

		if norm(Ek-Ek2) > Ns*eps:
			raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

	Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=0,a=a)
	Ns=Hk.Ns
	Ek=Hk.eigvalsh()

	Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=0,pzblock=+1,a=a)
	Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=0,pzblock=-1,a=a)
	
	Ek1=Hk1.eigvalsh()
	Ek2=Hk2.eigvalsh()
	Ekp=np.append(Ek1,Ek2)
	Ekp.sort()


	if norm(Ek-Ekp) > Ns*eps:
			raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,0,np.dtype(dtype),Nup) )

	if((L/a)%2 == 0):
		for kblock in xrange(1,L_2):
			Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=kblock,a=a)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=+1,a=a) 
	
			Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=-1,a=a)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps:
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )
	
			if norm(Ek-Ek2) > Ns*eps:
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

		Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=L_2,a=a)
		Ns=Hk.Ns
		Ek=Hk.eigvalsh()

		Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=L_2,pzblock=+1,a=a)
		Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=L_2,pzblock=-1,a=a)	
		Ek1=Hk1.eigvalsh()
		Ek2=Hk2.eigvalsh()
		Ekp=np.append(Ek1,Ek2)
		Ekp.sort()

		if np.sum(np.abs(Ek-Ekp)) > Ns*eps:
				raise Exception( "test failed t pz symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,int(L/2),np.dtype(dtype),Nup) )
	else:
		for kblock in xrange(1,L_2+1):
			Hk=hamiltonian(static,[],L,Nup=Nup,dtype=kdtype,kblock=kblock,a=a)
			Ns=Hk.Ns
			Ek=Hk.eigvalsh()
	
			Hk1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=+1,a=a) 
	
			Hk2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pzblock=-1,a=a)	
	
			Ek1=Hk1.eigvalsh()
			Ek2=Hk2.eigvalsh()
	
			if norm(Ek-Ek1) > Ns*eps:
				raise Exception( "test failed t p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )
	
			if norm(Ek-Ek2) > Ns*eps:
				raise Exception( "test failed t p- symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )





def check_t_p_z(L,dtype,Nup=None):
	hx=random()
	J=random()
	h=[[hx,i] for i in xrange(L)]
	J1=[[J,i,(i+1)%L] for i in xrange(L)]
	if type(Nup) is int:
		static=[["zz",J1],["xx",J1],["yy",J1]] 
	else:
		static=[["zz",J1],["x",h]]

	eps=20*np.finfo(dtype).eps
	L_2=int(L/2)
	for kblock in xrange(-L_2+1,L_2+1):
		Hkp1=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1)
		Hkp2=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1)
		Ns=Hkp1.Ns
		Ekp1=Hkp1.eigvalsh()
		Ekp2=Hkp2.eigvalsh()

		Hkpz11=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1,zblock=+1) 
		Hkpz12=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=+1,zblock=-1)	
		Ekpz11=Hkpz11.eigvalsh()
		Ekpz12=Hkpz12.eigvalsh()

		Ekpz1=np.concatenate((Ekpz11,Ekpz12))
		Ekpz1.sort()

		Hkpz21=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1,zblock=+1) 
		Hkpz22=hamiltonian(static,[],L,Nup=Nup,dtype=dtype,kblock=kblock,pblock=-1,zblock=-1)	
		Ekpz21=Hkpz21.eigvalsh()
		Ekpz22=Hkpz22.eigvalsh()

		Ekpz2=np.concatenate((Ekpz21,Ekpz22))
		Ekpz2.sort()
			
		if norm(Ekp1-Ekpz1) > Ns*eps:
			raise Exception( "test failed t z p+  symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

		if norm(Ekp2-Ekpz2) > Ns*eps:
			raise Exception( "test failed t z p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

		if(kblock not in [0,L_2]):
			if norm(Ekp2-Ekpz1) > Ns*eps:
				raise Exception( "test failed t z p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )

			if norm(Ekp1-Ekpz2) > Ns*eps:
				raise Exception( "test failed t z p+ symmetry at L={0:3d} kblock={1:3d} with dtype {2} and Nup={3}".format(L,kblock,np.dtype(dtype),Nup) )


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


	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,1):
			check_t_p(L,dtype)
			for Nup in xrange(L+1):
				check_t_p(L,dtype,Nup=Nup)

	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_t_pz(L,dtype,Nup=int(L/2))
			check_t_pz(L,dtype)

	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,Lmax+1,2):
			check_p_z(L,dtype,Nup=int(L/2))
			check_p_z(L,dtype) 





#check_m(4)
#check_opstr(4)
#check_obc(8)
#check_pbc(8)


