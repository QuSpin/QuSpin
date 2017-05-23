from __future__ import print_function
import sys,os

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)
import numpy as np
import scipy.sparse as sp
from quspin.operators import hamiltonian



def test_trace():
    M = np.arange(9).reshape((3,3)).astype(np.complex128)
    H = hamiltonian([M],[])
    H_trace = H.trace()
    trace = np.trace(M)
    np.testing.assert_equal(H_trace,trace)


def test_hermitian_conj():
    M = 1j*np.arange(9).reshape((3,3))
    H = hamiltonian([M],[])
    np.testing.assert_allclose((H-H.H).todense()-(M-M.T.conj()),0,atol=1e-12)

def test_transpose():
    M = np.arange(9).reshape((3,3))
    H = hamiltonian([M],[])
    np.testing.assert_allclose((H-H.T).todense()-(M-M.T),0,atol=1e-12)


def test_conj():
    M = 1j*np.arange(9).reshape((3,3))
    H = hamiltonian([M],[])
    np.testing.assert_allclose((H-H.conj()).todense()-(M-M.conj()),0,atol=1e-12)


def test_mul_dense():
    M1 = np.random.ranf(size=9).reshape((3,3))
    M2 = np.random.ranf(size=9).reshape((3,3))
    H = hamiltonian([M1],[])
    H_1 = H * M2
    H_2 = M2 * H

    np.testing.assert_allclose(H_1.toarray() - (M1.dot(M2)),0,atol=1e-12)
    np.testing.assert_allclose(H_2.toarray() - (M2.dot(M1)),0,atol=1e-12)


def test_mul_sparse():
    M1 = sp.random(10,10,format="csr")
    M2 = sp.random(10,10,format="csr")
    H = hamiltonian([M1],[])
    H_1 = H * M2
    H_2 = M2 * H

    np.testing.assert_allclose((H_1.tocsr() - (M1.dot(M2))).todense(),0,atol=1e-12)
    np.testing.assert_allclose((H_2.tocsr() - (M2.dot(M1))).todense(),0,atol=1e-12)


def test_shape():
    M1 = sp.random(10,10,format="csr")
    H = hamiltonian([M1],[])
    assert(H.get_shape==M1.shape)

 
test_shape()
test_trace()
test_hermitian_conj()
test_transpose()
test_conj()
test_mul_sparse()
test_mul_dense()