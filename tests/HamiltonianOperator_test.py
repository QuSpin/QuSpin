import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from qspin.operators import HamiltonianOperator,hamiltonian
from qspin.basis import spin_basis_1d
import numpy as np
from numpy.testing import assert_almost_equal
import scipy.sparse as sp

def HamiltonianOperator_test():
	def dot_test_dense(L,dtype):
		J = [[np.random.ranf(),i,(i+1)] for i in xrange(L-1)]
		operator_list = [["zz",J],["xx",J],["yy",J]]
		H_op = HamiltonianOperator(operator_list,L,Nup=L/2,dtype=dtype,check_symm=False,check_herm=False,check_pcon=False)
		LO = H_op.LinearOperator
		H = hamiltonian(operator_list,[],N=L,Nup=L/2,dtype=dtype,check_symm=False,check_herm=False,check_pcon=False)

		v = np.random.ranf(size=(H.Ns,)).astype(np.complex128)

		atol = np.finfo(dtype).eps*10
		rtol = 0.0

		u = H.dot(v)
		u_op = H_op.dot(v)

		assert_almost_equal(u,u_op)

		u = H.dot(v)
		u_op = H_op.dot(v)

		assert_almost_equal(u,u_op)

		u = H.T.dot(v.T)
		u_op = H_op.T.dot(v.T)

		assert_almost_equal(u,u_op)

		u = H.conj().dot(v.T)
		u_op = H_op.conj().dot(v.T)

		assert_almost_equal(u,u_op)

		u = H.H.dot(v.T)
		u_op = H_op.H.dot(v.T)

		assert_almost_equal(u,u_op)

		u = H.rdot(v.T)
		u_op = H_op.rdot(v.T)

		assert_almost_equal(u,u_op)

		u = H.rdot(v.T)
		u_op = H_op.rdot(v.T)

		assert_almost_equal(u,u_op)

		u = H.T.rdot(v.T)
		u_op = H_op.T.rdot(v.T)

		assert_almost_equal(u,u_op)

		u = H.conj().rdot(v.T)
		u_op = H_op.conj().rdot(v.T)

		assert_almost_equal(u,u_op)

		u = H.H.rdot(v.T)
		u_op = H_op.H.rdot(v.T)

		assert_almost_equal(u,u_op)

		v = np.random.ranf(size=(H.Ns,L))	

		u = H.dot(v)
		u_op = H_op.dot(v)

		assert_almost_equal(u,u_op)

		u = H.T.dot(v)
		u_op = H_op.T.dot(v)

		assert_almost_equal(u,u_op)

		u = H.conj().dot(v)
		u_op = H_op.conj().dot(v)

		assert_almost_equal(u,u_op)

		u = H.H.dot(v)
		u_op = H_op.H.dot(v)

		assert_almost_equal(u,u_op)

		u = H.rdot(v.T)
		u_op = H_op.rdot(v.T)

		assert_almost_equal(u,u_op)

		u = H.T.rdot(v.T)
		u_op = H_op.T.rdot(v.T)

		assert_almost_equal(u,u_op)

		u = H.conj().rdot(v.T)
		u_op = H_op.conj().rdot(v.T)

		assert_almost_equal(u,u_op)

		u = H.H.rdot(v.T)
		u_op = H_op.H.rdot(v.T)

		assert_almost_equal(u,u_op)

		print "L = {0}, dtype = {1} dense test passed".format(L,dtype.__name__)


	def dot_test_sparse(L,dtype):
		J = [[np.random.ranf(),i,(i+1)] for i in xrange(L-1)]
		operator_list = [["zz",J],["xx",J],["yy",J]]
		H_op = HamiltonianOperator(operator_list,L,Nup=L/2,dtype=dtype,check_symm=False,check_herm=False,check_pcon=False)
		LO = H_op.LinearOperator
		H = hamiltonian(operator_list,[],N=L,Nup=L/2,dtype=dtype,check_symm=False,check_herm=False,check_pcon=False)

		v = sp.random(H.Ns,5).astype(np.complex128)

		atol = np.finfo(dtype).eps*10
		rtol = 0.0

		u = H.dot(v).todense()
		u_op = H_op.dot(v).todense()

		assert_almost_equal(u,u_op)

		u = H.dot(v).todense()
		u_op = H_op.dot(v).todense()

		assert_almost_equal(u,u_op)

		u = H.T.dot(v).todense()
		u_op = H_op.T.dot(v).todense()

		assert_almost_equal(u,u_op)

		u = H.conj().dot(v).todense()
		u_op = H_op.conj().dot(v).todense()

		assert_almost_equal(u,u_op)

		u = H.H.dot(v).todense()
		u_op = H_op.H.dot(v).todense()

		assert_almost_equal(u,u_op)

		u = H.rdot(v.T).todense()
		u_op = H_op.rdot(v.T).todense()

		assert_almost_equal(u,u_op)

		u = H.rdot(v.T).todense()
		u_op = H_op.rdot(v.T).todense()

		assert_almost_equal(u,u_op)

		u = H.T.rdot(v.T).todense()
		u_op = H_op.T.rdot(v.T).todense()

		assert_almost_equal(u,u_op)

		u = H.conj().rdot(v.T).todense()
		u_op = H_op.conj().rdot(v.T).todense()

		assert_almost_equal(u,u_op)

		u = H.H.rdot(v.T).todense()
		u_op = H_op.H.rdot(v.T).todense()

		assert_almost_equal(u,u_op)

		print "L = {0}, dtype = {1} sparse test passed".format(L,dtype.__name__)



	
	for L in xrange(1,7):
		for dtype in [np.float32,np.float64,np.complex64,np.complex128]:
			yield dot_test_dense, (L,dtype)

	for L in xrange(1,7):
		for dtype in [np.float32,np.float64,np.complex64,np.complex128]:
			yield dot_test_sparse, (L,dtype)

	


for test_fun, param in HamiltonianOperator_test():
	test_fun(*param)
#	try:
#		test_fun(*param)
#	except AssertionError:
#		raise 
		


