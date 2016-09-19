import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from qspin.operators import HamiltonianOperator,hamiltonian
from qspin.basis import spin_basis_1d
import numpy as np
from numpy.testing import assert_array_equal
import scipy.sparse as sp

if hasattr(np,"float128"):
	dtypes = [np.float32,np.float64,np.float128,np.complex64,np.complex128,np.complex256]
else:
	dtypes = [np.float32,np.float64,np.complex64,np.complex128]

def todense(x):
	if sp.issparse(x):
		return x.todense()
	elif isinstance(x,hamiltonian):
		return x.todense()
	else:
		return np.asarray(x)

def check_dot(H,H_op,v):
	u = H.dot(v)
	u_op = H_op.dot(v)
	assert_array_equal(todense(u),todense(u_op))

def check_rdot(H,H_op,v):
	u = H.rdot(v)
	u_op = H_op.rdot(v)
	assert_array_equal(todense(u),todense(u_op))

def check_add(H,H_op,mat):
	result1 = H + mat
	result2 = H_op + mat
	assert_array_equal(todense(result1),todense(result2))

	result1 = mat + H
	result2 = mat + H_op
	assert_array_equal(todense(result1),todense(result2))

	result1 = H - mat
	result2 = H_op - mat
	assert_array_equal(todense(result1),todense(result2))

	result1 = mat - H
	result2 = mat - H_op
	assert_array_equal(todense(result1),todense(result2))

def check_mul(H,H_op,mat):
	result1 = H.dot(mat)
	result2 = H_op * mat
	assert_array_equal(todense(result1),todense(result2))

	result1 = H.rdot(mat)
	result2 = mat * H_op
	assert_array_equal(todense(result1),todense(result2))


def test_ops():
	for L in range(1,5):
		Jz = [[1.0,i,(i+1)] for i in xrange(L-1)]
		Jx = [[2.0,i,(i+1)] for i in xrange(L-1)]
		Jy = [[3.0,i,(i+1)] for i in xrange(L-1)]
		operator_list = [["zz",Jz],["xx",Jx],["yy",Jy]]
		for dtype in dtypes:
			H_op = HamiltonianOperator(operator_list,L,dtype=dtype,check_symm=False,check_herm=False,check_pcon=False)
			H = hamiltonian(operator_list,[],L,dtype=dtype,check_symm=False,check_herm=False,check_pcon=False)

			v = np.random.randint(3,size=(H.Ns,)).astype(dtype)

			yield check_dot,H,H_op,v
			yield check_dot,H.T,H_op.T,v
			yield check_dot,H.H,H_op.H,v
			yield check_dot,H.conj(),H_op.conj(),v

			yield check_rdot,H,H_op,v
			yield check_rdot,H.T,H_op.T,v
			yield check_rdot,H.H,H_op.H,v
			yield check_rdot,H.conj(),H_op.conj(),v

			v = np.random.randint(3,size=(H.Ns,10)).astype(dtype)

			yield check_dot,H,H_op,v
			yield check_dot,H.T,H_op.T,v
			yield check_dot,H.H,H_op.H,v
			yield check_dot,H.conj(),H_op.conj(),v

			v = np.random.randint(3,size=(10,H.Ns)).astype(dtype)

			yield check_rdot,H,H_op,v
			yield check_rdot,H.T,H_op.T,v
			yield check_rdot,H.H,H_op.H,v
			yield check_rdot,H.conj(),H_op.conj(),v

			v = np.random.randint(3,size=(H.Ns,1)).astype(dtype)
			v = sp.csr_matrix(v)

			yield check_dot,H,H_op,v
			yield check_dot,H.T,H_op.T,v
			yield check_dot,H.H,H_op.H,v
			yield check_dot,H.conj(),H_op.conj(),v

			yield check_rdot,H,H_op,v.T
			yield check_rdot,H.T,H_op.T,v.T
			yield check_rdot,H.H,H_op.H,v.T
			yield check_rdot,H.conj(),H_op.conj(),v.T

			v = np.random.randint(3,size=(H.Ns,10)).astype(dtype)
			v = sp.csr_matrix(v)

			yield check_dot,H,H_op,v
			yield check_dot,H.T,H_op.T,v
			yield check_dot,H.H,H_op.H,v
			yield check_dot,H.conj(),H_op.conj(),v

			v = np.random.randint(3,size=(10,H.Ns)).astype(dtype)
			v = sp.csr_matrix(v)

			yield check_rdot,H,H_op,v
			yield check_rdot,H.T,H_op.T,v
			yield check_rdot,H.H,H_op.H,v
			yield check_rdot,H.conj(),H_op.conj(),v

			v = np.random.randint(3,size=(H.Ns,H.Ns)).astype(dtype)

			yield check_mul,H,H_op,v
			yield check_mul,H.T,H_op.T,v
			yield check_mul,H.H,H_op.H,v
			yield check_mul,H.conj(),H_op.conj(),v





for test_tup in test_ops():
	test_tup[0](*test_tup[1:])







