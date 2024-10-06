from quspin.basis import boson_basis_general
from quspin.basis.transformations import square_lattice_trans
from quspin.operators import hamiltonian
import numpy as np


def run(sps, Lx, Ly):
    N = Lx * Ly
    nmax = sps - 1
    tr = square_lattice_trans(Lx, Ly)

    basis_dict = {}
    Nbs = range(nmax * N + 1)

    for Nb in Nbs:
        basis_blocks = []
        pcon_basis = boson_basis_general(N, Nb=Nb, sps=sps)
        Ns_block = 0
        for blocks in tr.allowed_blocks_iter():
            basis = boson_basis_general(N, Nb=Nb, sps=sps, **blocks)
            Ns_block += basis.Ns
            basis_blocks.append(basis)

        try:
            assert Ns_block == pcon_basis.Ns
        except AssertionError:
            print(Ns_block, pcon_basis.Ns)
            raise AssertionError

        basis_dict[Nb] = (pcon_basis, basis_blocks)

    J = [[1.0, i, tr.T_x[i]] for i in range(N)]
    J.extend([[1.0, i, tr.T_y[i]] for i in range(N)])

    static = [["nn", J], ["+-", J], ["-+", J]]

    for Nb, (pcon_basis, basis_blocks) in basis_dict.items():
        H_pcon = hamiltonian(static, [], basis=pcon_basis, dtype=np.float64)
        if H_pcon.Ns > 0:
            E_pcon = np.linalg.eigvalsh(H_pcon.todense())
        else:
            E_pcon = np.array([])

        E_block = []
        for basis in basis_blocks:
            H = hamiltonian(static, [], basis=basis, dtype=np.complex128)
            if H.Ns > 0:
                E_block.append(np.linalg.eigvalsh(H.todense()))

        E_block = np.hstack(E_block)
        E_block.sort()
        np.testing.assert_allclose(E_pcon, E_block, atol=1e-13)
        print("passed Nb={} sector".format(Nb))


def test():
    run(2, 3, 3)
    run(3, 3, 3)
    run(2, 3, 2)
    run(3, 3, 2)
