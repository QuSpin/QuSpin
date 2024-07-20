import sys, os


# print(os.environ["OMP_NUM_THREADS"])
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from quspin.tools.evolution import expm_multiply_parallel
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import random, eye
import numpy as np


def test_imag_time(L=20, seed=0):
    np.random.seed(seed)

    basis = spin_basis_1d(L, m=0, kblock=0, pblock=1, zblock=1)

    J = [[1.0, i, (i + 1) % L] for i in range(L)]
    static = [["xx", J], ["yy", J], ["zz", J]]
    H = hamiltonian(static, [], basis=basis, dtype=np.float64)

    (E,), psi_gs = H.eigsh(k=1, which="SA")

    psi_gs = psi_gs.ravel()

    A = -(H.tocsr() - E * eye(H.Ns, format="csr", dtype=np.float64))

    U = expm_multiply_parallel(A)

    v1 = np.random.normal(0, 1, size=(H.Ns, 10))
    v1 /= np.linalg.norm(v1, axis=0)

    v2 = v1.copy()

    for i in range(100):
        v2 = U.dot(v2)
        v2 /= np.linalg.norm(v2)

        v1 = expm_multiply(A, v1)
        v1 /= np.linalg.norm(v1)

        if np.all(np.abs(H.expt_value(v2) - E) < 1e-15):
            break  #

        i += 1

    np.testing.assert_allclose(
        v1,
        v2,
        rtol=0,
        atol=5e-15,
        err_msg="imaginary time test failed, seed {:d}".format(seed),
    )


def test_ramdom_matrix(N=3500, ntest=10, seed=0):
    np.random.seed(seed)
    i = 0
    while i < ntest:
        print("testing random matrix {}".format(i + 1))
        A = random(N, N, density=np.log(N) / N) + 1j * random(
            N, N, density=np.log(N) / N
        )
        A = A.tocsr()

        v = np.random.normal(0, 1, size=(N, 10)) + 1j * np.random.normal(
            0, 1, size=(N, 10)
        )
        v /= np.linalg.norm(v)

        v1 = expm_multiply(A, v)
        v2 = expm_multiply_parallel(A).dot(v)

        np.testing.assert_allclose(
            v1,
            v2,
            rtol=0,
            atol=5e-15,
            err_msg="random matrix test failed, seed {:d}".format(seed),
        )
        i += 1


def test_ramdom_int_matrix(N=3500, ntest=10, seed=0):
    np.random.seed(seed)
    i = 0
    while i < ntest:
        print("testing random integer matrix {}".format(i + 1))
        data_rvs = lambda n: np.random.randint(-100, 100, size=n, dtype=np.int8)
        A = random(N, N, density=np.log(N) / N, data_rvs=data_rvs, dtype=np.int8)
        A = A.tocsr()

        v = np.random.normal(0, 1, size=(N, 10)) + 1j * np.random.normal(
            0, 1, size=(N, 10)
        )
        v /= np.linalg.norm(v)

        v1 = expm_multiply(-0.01j * A, v)
        v2 = expm_multiply_parallel(A, a=-0.01j, dtype=np.complex128).dot(v)

        np.testing.assert_allclose(
            v1,
            v2,
            rtol=0,
            atol=5e-15,
            err_msg="random matrix test failed, seed {:d}".format(seed),
        )
        i += 1


test_imag_time()
test_ramdom_matrix()
test_ramdom_int_matrix()
print("expm_multiply_parallel tests passed!")
