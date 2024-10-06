from quspin.operators import hamiltonian, exp_op
from quspin.basis import boson_basis_1d
from quspin.tools.block_tools import block_ops
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 1000

basis = boson_basis_1d(L, Nb=1)

a = 1.0
w = 0.0
t = -1.0 * np.exp(-1j * a * np.pi)

J = [[t, i, (i + 1) % L] for i in range(L - 1)]
J_cc = [[t.conjugate(), i, (i + 1) % L] for i in range(L - 1)]
V = [[np.random.uniform(-w, w) + 0.01 * i, i] for i in range(L - 1)]
static = [["+-", J], ["-+", J_cc], ["n", V]]


H = hamiltonian(static, [], basis=basis)


sites = np.arange(L)

psi = np.exp(-((sites - L / 4) ** 2) / float(L / 10))
psi /= np.linalg.norm(psi)

psi_t = exp_op(H, a=-1j, start=0, stop=1000000, num=100000, iterate=True).dot(psi)

psi = next(psi_t)
ns = np.abs(psi) ** 2


fig = plt.figure()
ax = plt.gca()
(line,) = ax.plot(sites, ns, marker="")


def updatefig(i):
    psi = next(psi_t)
    ns = np.abs(psi) ** 2
    line.set_data(sites, ns)

    return (line,)


ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()
