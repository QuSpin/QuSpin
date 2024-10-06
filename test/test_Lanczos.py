from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from scipy.sparse.linalg import expm_multiply
from quspin.tools.lanczos import lanczos_full, lanczos_iter, lin_comb_Q_T, expm_lanczos
import numpy as np


def test():

    np.random.seed(0)

    L = 20

    basis = spin_basis_1d(L, m=0, kblock=0, pblock=1, zblock=1)


    J_list = [[1.0, i, (i + 1) % L] for i in range(L)]

    static = [[op, J_list] for op in ["xx", "yy", "zz"]]

    H = -hamiltonian(static, [], basis=basis, dtype=np.float64)


    v0 = np.random.normal(0, 1, size=basis.Ns)
    v0 /= np.linalg.norm(v0)
    v1 = v0.copy()
    v2 = v0.copy()

    print("checking lanczos matrix exponential calculation")
    for i in range(100):
        print("evolving step: {}".format(i))
        E1, V1, lv1 = lanczos_full(H, v1, 20)
        E2, V2, lv2 = lanczos_iter(H, v2, 20)
        v0 = expm_multiply(-0.1j * H.static, v0)
        v1 = expm_lanczos(E1, V1, lv1, a=-0.1j)
        v2 = expm_lanczos(E2, V2, lv2, a=-0.1j)

        np.testing.assert_allclose(v1, v0, atol=1e-10, rtol=0)
        np.testing.assert_allclose(v2, v0, atol=1e-10, rtol=0)


    print("checking ground state lanczos calculation")


    nvec = 50

    E_GS, psi_GS = H.eigsh(k=1, which="LA")

    psi_GS = psi_GS.ravel()

    v0 = np.random.normal(0, 1, size=basis.Ns)
    v0 /= np.linalg.norm(v0)

    E, V, Q = lanczos_full(H, v0, nvec, full_ortho=False)

    v1 = lin_comb_Q_T(V[:, -1], Q)

    try:
        dE = np.abs(E[-1] - E_GS[-1])
        assert dE < 1e-10
    except AssertionError:
        raise AssertionError(
            "Energy failed to converge |E_lanczos-E_exact| = {} > 1e-10".format(dE)
        )

    try:
        F = np.abs(np.log(np.abs(np.vdot(v1, psi_GS))))
        assert F < 1e-10
    except AssertionError:
        raise AssertionError(
            "wavefunction failed to converge Fedility = {} > 1e-10".format(F)
        )

    E, V, Q_iter = lanczos_iter(H, v0, nvec, return_vec_iter=True)

    v2 = lin_comb_Q_T(V[:, -1], Q_iter)

    np.testing.assert_allclose(v2, v1, atol=1e-10, rtol=0)

    print("lanczos_utils tests passed!")
