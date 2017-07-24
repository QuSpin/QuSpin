from __future__ import print_function
import sys,os

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)
import numpy as np
import scipy.sparse as sp
from scipy.integrate import complex_ode
from quspin.operators import hamiltonian
try:
    from itertools import izip as zip
except ImportError:
    pass



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

    def f(t):
        return np.exp(-1j*t)

    def f_cc(t):
        return np.exp(1j*t)

    M = (M+M.T.conj())/2.0
    H = hamiltonian([M],[[M,f,()],[M,f_cc,()]])
    np.testing.assert_allclose((H-H.H).todense(1342.1293),0,atol=1e-12)


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


def test_evolve():
    def ifunc(t,y):
        return -1j*np.cos(t)*M.dot(y)

    def func(t,y):
        return -np.cos(t)*M.dot(y)

    M = np.random.uniform(-1,1,size=(4,4))+1j*np.random.uniform(-1,1,size=(4,4))
    M = (M.T.conj() + M)/2.0
    M = np.asarray(M)

    H = hamiltonian([],[[M,np.cos,()]])

    psi0 = np.random.uniform(-1,1,size=(4,))+1j*np.random.uniform(-1,1,size=(4,))
    psi0 /= np.linalg.norm(psi0)

    isolver = complex_ode(ifunc)
    isolver.set_integrator("dop853",atol=1e-9,rtol=1e-9,nsteps=np.iinfo(np.int32).max)
    isolver.set_initial_value(psi0, 0)

    solver = complex_ode(func)
    solver.set_integrator("dop853",atol=1e-9,rtol=1e-9,nsteps=np.iinfo(np.int32).max)
    solver.set_initial_value(psi0, 0)

    times = np.arange(0,100.1,10)

    ipsi_t = H.evolve(psi0,0,times,iterate=True)
    psi_t = H.evolve(psi0,0,times,iterate=True,imag_time=True)

    for i,(ipsi,psi) in enumerate(zip(ipsi_t,psi_t)):
        solver.integrate(times[i])
        solver._y/=np.linalg.norm(solver.y)
        np.testing.assert_allclose(psi-solver.y,0,atol=1e-10)

        isolver.integrate(times[i])
        np.testing.assert_allclose(ipsi-isolver.y,0,atol=1e-10)

 


test_shape()
test_trace()
test_hermitian_conj()
test_transpose()
test_conj()
test_mul_sparse()
test_mul_dense()
test_evolve()