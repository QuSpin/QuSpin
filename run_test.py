from exact_diag_py.spins import hamiltonian
from exact_diag_py.basis import basis1d
import numpy as np


def check_m():
	for dtype in (np.float32,np.float64,np.complex64,np.complex128):
		for L in xrange(2,10):
			J=[[1.0,i,(i+1)%L] for i in xrange(L)]

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

			if np.sum(Em-E)/Ns > eps:
				return False


	return True


print check_m()


