from qspin.operators import HamiltonianOperator,hamiltonian
from qspin.basis import spin_basis_1d
import numpy as np


def dot_test(L,dtype):
	J = [[np.random.ranf(),i,(i+1)] for i in xrange(L-1)]
	operator_list = [["zz",J],["xx",J],["yy",J]]
	H_op = HamiltonianOperator(operator_list,L,Nup=L/2,dtype=dtype,check_symm=False,check_herm=False,check_pcon=False)
	LO = H_op.LinearOperator
	H = hamiltonian(operator_list,[],N=L,Nup=L/2,dtype=dtype,check_symm=False,check_herm=False,check_pcon=False)
	v = np.random.ranf(size=(H.Ns,))

	atol = np.finfo(dtype).eps*10
	rtol = 0.0

	u = H.dot(v)
	u_op = H_op.dot(v)

	assert(np.allclose(u,u_op,atol=atol,rtol=rtol))

	u = H.rdot(v)
	u_op = H_op.rdot(v)

	assert(np.allclose(u,u_op,atol=atol,rtol=rtol))

	u = H.T.rdot(v)
	u_op = H_op.T.rdot(v)

	assert(np.allclose(u,u_op,atol=atol,rtol=rtol))

	u = H.conj().rdot(v)
	u_op = H_op.conj().rdot(v)

	assert(np.allclose(u,u_op,atol=atol,rtol=rtol))

	u = H.H.rdot(v)
	u_op = H_op.H.rdot(v)

	assert(np.allclose(u,u_op,atol=atol,rtol=rtol))


def tests():
	for L in xrange(2,11):
		print L
		for dtype in [np.float32,np.float64,np.complex64,np.complex128]:
			dot_test(L,dtype)
		

tests()





