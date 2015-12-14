from exact_diag_py.spins import hamiltonian
from exact_diag_py.basis import basis1d
import numpy as np
from numpy.random import random,seed

seed()


def check_opstr():
	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,8):
			J1=[[random(),i,(i+1)%L] for i in xrange(L)]
			J2=[[J1[i][0]/2,i,(i+1)%L] for i in xrange(L)]

			static1=[['zz',J1],['yy',J1],['xx',J1]]
			static2=[['zz',J1],['+-',J2],['-+',J2]]

			eps=np.finfo(dtype).eps

			H1=hamiltonian(static1,[],L,dtype=dtype)
			H2=hamiltonian(static2,[],L,dtype=dtype)
			Ns=H1.Ns
			E1=H1.eigvalsh()
			E2=H2.eigvalsh()

			if np.sum(E1-E2)/Ns > eps:
				return False
			
			for Nup in xrange(L+1):
				H1=hamiltonian(static1,[],L,Nup=Nup,dtype=dtype)
				H2=hamiltonian(static2,[],L,Nup=Nup,dtype=dtype)
				Ns=H1.Ns
				E1=H1.eigvalsh()
				E2=H2.eigvalsh()
				if np.sum(np.abs(E1-E2))/Ns > 10*eps:
					raise Exception( "test failed opstr at L={0} with dtype {1}".format(L,str(np.dtype(dtype))) )
	return True





def check_m():
	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,8):
			J=[[random(),i,(i+1)%L] for i in xrange(L)]

			static=[['zz',J],['yy',J],['xx',J]]

			H=hamiltonian(static,[],L,dtype=dtype)
			Ns=H.Ns
			E=H.eigvalsh()

			Em=[]
			for Nup in xrange(L+1):
				H=hamiltonian(static,[],L,Nup=Nup,dtype=dtype)
				Etemp=H.eigvalsh()
				Em.append(Etemp)

			Em=np.concatenate(Em)
			Em.sort()
			
			eps=np.finfo(dtype).eps
			if np.sum(np.abs(Em-E))/Ns > 10*eps:
				raise Exception( "test failed m symmetry at L={0} with dtype {1}".format(L,str(np.dtype(dtype))) )


	return True


def check_z(L,dtype):
	Nup=L/2

	J=[[1.0,i,(i+1)%L] for i in xrange(L)]

	static=[['zz',J],['yy',J],['xx',J]]

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
	if np.sum(np.abs(Ez-E))/Ns > 10*eps:
		raise Exception( "test failed z symmetry at L={0} with dtype {1}".format(L,str(np.dtype(dtype))) )


def check_p(L,dtype):
	Nup=L/2

	J=[[1.0,i,(i+1)%L] for i in xrange(L)]

	static=[['zz',J],['yy',J],['xx',J]]

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
	if np.sum(np.abs(Ep-E)/Ns) > 10*eps:
		raise Exception( "test failed p symmetry at L={0} with dtype {1}".format(L,str(np.dtype(dtype))) )




def check_pz(L,dtype):
	Nup=L/2

	J=[[1.0,i,(i+1)%L] for i in xrange(L)]

	static=[['zz',J],['yy',J],['xx',J]]

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
	if np.sum(np.abs(Epz-E)/Ns) > 10*eps:
		raise Exception( "test failed pz symmetry at L={0} with dtype {1}".format(L,str(np.dtype(dtype))) )




def check_p_z(L,dtype):
	Nup=L/2

	J=[[1.0,i,(i+1)%L] for i in xrange(L)]

	static=[['zz',J],['yy',J],['xx',J]]

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
	if np.sum(np.abs(Epz-E)/Ns) > 10*eps:
		raise Exception( "test failed p z symmetry at L={0} with dtype {1}".format(L,str(np.dtype(dtype))) )




def check_obc():
	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,11,2):
			check_z(L,dtype)
			check_p(L,dtype)	
			check_pz(L,dtype)
			check_p_z(L,dtype)





check_obc()
check_m()
check_opstr()





