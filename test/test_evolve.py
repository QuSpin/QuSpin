from matplotlib import pyplot as plt
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import obs_vs_time
import numpy as np
import scipy.sparse as sp

# import matplotlib.pyplot as plt
try:
    from itertools import izip as zip
except ImportError:
    pass

def test_values(lhs, rhs, **kwargs):
    import matplotlib.pyplot as plt
    try:
        np.testing.assert_allclose(lhs, rhs, **kwargs)
    except Exception as e:
        plt.plot(lhs, label="lhs")
        plt.plot(rhs, label="rhs")
        plt.legend()
        plt.figure()
        plt.plot(lhs - rhs, label="diff")
        plt.show()
        
        raise e

L = 10
dtype = np.float64
T = 0.5
n = 20

tol = 1e-6


def drive(t):
    return t * np.cos(2 * np.pi * t / T)


basis = spin_basis_1d(L, Nup=L // 2, kblock=0, pzblock=1, a=2)
times = np.arange(0, (n + 0.1) * T, T / 4)


J1 = [[1.0, i, (i + 1) % L] for i in range(L)]
J2 = [[(-1) ** i, i] for i in range(L)]
Sz = [[1.0 / L, i] for i in range(L)]

static = [
    ["xx", J1],
    ["yy", J1],
    ["zz", J1],
]

dynamic = [
    ["z", J2, drive, ()],
]

H_0 = hamiltonian(static, [], basis=basis, dtype=dtype)
V_t = hamiltonian([], dynamic, basis=basis, dtype=dtype)
O_0 = H_0.todense()


E, psi_0 = H_0.eigsh(k=1, which="SA", maxiter=10000)

H = H_0 + V_t


psi_0 = psi_0.ravel()
rho_0 = np.outer(psi_0.conj(), psi_0).astype(np.complex128)


n_copies = 7
psi_0_vec = np.tile(psi_0, (n_copies, 1)).T
psi_t_vec = H.evolve(
    psi_0_vec, 0, times, eom="SE", iterate=True, atol=1e-10, rtol=1e-10
)
O_expt_vec = obs_vs_time(psi_t_vec, times, dict(O=O_0))["O"]


psi_t = H.evolve(psi_0, 0, times, eom="SE", iterate=True, atol=1e-10, rtol=1e-10)
O_expt = obs_vs_time(psi_t, times, dict(O=O_0))["O"]

rho_t = H.evolve(rho_0, 0, times, eom="LvNE", iterate=True, atol=1e-10, rtol=1e-10)
O_expt_rho = obs_vs_time(rho_t, times, dict(O=O_0))["O"]

## test
for j in range(n_copies):
    test_values(O_expt, O_expt_vec[..., j], atol=1e-5, rtol=1e-5)

try:
    test_values(O_expt, O_expt_rho, atol=1e-5, rtol=1e-5)
except AssertionError:
    rho_t = H.evolve(rho_0, 0, times, eom="LvNE", iterate=True, atol=1e-10, rtol=1e-10)
    for t, O_SE, rho in zip(times, O_expt, rho_t):
        O_LvNE = np.einsum("ij,ji->", rho, O_0).real
        if np.abs(O_SE - O_LvNE) > tol:
            raise Exception(
                "'LvNE' failed at t={}, diff of {}".format(t, np.abs(O_SE - O_LvNE))
            )


print("evolve checks passed")
