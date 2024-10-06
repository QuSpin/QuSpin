from quspin.basis import spinful_fermion_basis_general
from quspin.basis.transformations import square_lattice_trans
from quspin.operators import hamiltonian
import numpy as np


def run(Lx, Ly):
    N = Lx * Ly
    tr = square_lattice_trans(Lx, Ly)

    basis_dict = {}
    Nfs = range(N + 1)

    for Nf in Nfs:
        basis_blocks = []
        pcon_basis = spinful_fermion_basis_general(N, Nf=(Nf, N - Nf))
        Ns_block = 0
        for blocks in tr.allowed_blocks_iter_parity():
            basis = spinful_fermion_basis_general(N, Nf=(Nf, N - Nf), **blocks)
            Ns_block += basis.Ns
            basis_blocks.append(basis)

        assert Ns_block == pcon_basis.Ns

        basis_dict[Nf] = (pcon_basis, basis_blocks)

    J = [[np.sqrt(2.0), i, i] for i in range(N)]

    Jp = [[1.0, i, tr.T_x[i]] for i in range(N)]
    Jp.extend([[1.0, i, tr.T_y[i]] for i in range(N)])

    Jm = [[-1.0, i, tr.T_x[i]] for i in range(N)]
    Jm.extend([[-1.0, i, tr.T_y[i]] for i in range(N)])

    static = [["n|n", J], ["+-|", Jp], ["-+|", Jm], ["|+-", Jp], ["|-+", Jm]]

    for Nf, (pcon_basis, basis_blocks) in basis_dict.items():
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
        print("passed Nf={} sector".format(Nf))


def test():
    run(2, 2)

    run(3, 2)
    run(2, 3)
    #run(3, 3) # takes insanely long

    run(4, 2)
    run(2, 4)
