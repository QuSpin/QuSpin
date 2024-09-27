import numpy as np
from six import iteritems
from quspin.operators import hamiltonian
from quspin.tools.matvec import _get_matvec_function


def test():

	M = np.arange(16).reshape(4,4).astype(np.float64) 
	M = (M.T.conj() + M)/2.0
	M = np.asarray(M)

	H = hamiltonian([],[[M,np.cos,()]])


	psi0 = np.ones((4,),dtype=np.complex128)
	psi0 /= np.linalg.norm(psi0)


	V_out = np.zeros_like(psi0)
	V_out2 = np.zeros_like(psi0)


	time=np.pi/np.sqrt(0.5)

	for func, Hd in iteritems(H._dynamic):
		#
		matvec = _get_matvec_function(Hd)
		matvec(Hd,psi0,a=func(time),out=V_out,overwrite_out=False)
		#
		V_out2+=func(time)*Hd@psi0

	np.testing.assert_allclose(V_out-V_out2,0,atol=1e-10)



#test()


