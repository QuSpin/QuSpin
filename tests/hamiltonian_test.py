from __future__ import print_function
import sys,os

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)
import numpy as np
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



test_trace()
test_hermitian_conj()
test_transpose()
test_conj()