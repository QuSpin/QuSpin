import sys, os


from quspin.basis import spin_basis_general, spinful_fermion_basis_general
from quspin.basis.transformations import square_lattice_trans
from quspin.operators import hamiltonian
import numpy as np
from itertools import product
import os

"""
Testing double occupancies with the Heisenberg model $H = J \\sum_{i,j} S_i\\dot S_j$ using spinful fermion and spin bases. 
"""


def test(Lx, Ly):

    N = Lx * Ly

    nmax = int(eval("2*1/2"))
    sps = nmax + 1
    tr = square_lattice_trans(Lx, Ly)

    basis_dict = {}
    basis_dict_f = {}
    basis_dict_combined = {}
    Nups = range(nmax * N + 1)

    for Nup in Nups:
        basis_blocks = []
        basis_blocks_f = []

        pcon_basis = spin_basis_general(N, Nup=Nup, pauli=False)
        pcon_basis_f = spinful_fermion_basis_general(
            N, Nf=(Nup, N - Nup), double_occupancy=False
        )

        Ns_block = 0
        for blocks in tr.allowed_blocks_spin_inversion_iter(Nup, sps):
            basis = spin_basis_general(N, Nup=Nup, pauli=False, **blocks)
            Ns_block += basis.Ns
            basis_blocks.append(basis)

        Ns_block_f = 0
        for blocks_f in tr.allowed_blocks_spin_inversion_iter(
            Nup, sps
        ):  # requires simple symmetry definition
            basis_f = spinful_fermion_basis_general(
                N, Nf=(Nup, N - Nup), double_occupancy=False, **blocks_f
            )
            Ns_block_f += basis_f.Ns
            basis_blocks_f.append(basis_f)

        try:
            assert Ns_block == pcon_basis.Ns
        except AssertionError:
            print(Nup, Ns_block, pcon_basis.Ns)
            raise AssertionError("reduced blocks don't sum to particle sector.")

        try:
            assert Ns_block_f == pcon_basis_f.Ns
        except AssertionError:
            print(Nup, Ns_block_f, pcon_basis_f.Ns)
            raise AssertionError("fermion reduced blocks don't sum to particle sector.")

        try:
            assert Ns_block == pcon_basis_f.Ns
        except AssertionError:
            print(Nup, Ns_block_f, pcon_basis_f.Ns)
            raise AssertionError("fermion reduced blocks don't match spin blocks.")

        basis_dict[Nup] = (pcon_basis, basis_blocks)
        basis_dict_f[Nup] = (pcon_basis_f, basis_blocks_f)
        basis_dict_combined[Nup] = (
            pcon_basis,
            basis_blocks,
            pcon_basis_f,
            basis_blocks_f,
        )

    J = [[1.0, i, tr.T_x[i]] for i in range(N)] + [
        [1.0, i, tr.T_y[i]] for i in range(N)
    ]

    J_nn_ij = [[-0.25, i, tr.T_x[i]] for i in range(N)] + [
        [-0.25, i, tr.T_y[i]] for i in range(N)
    ]
    J_nn_ji = [[-0.25, tr.T_x[i], i] for i in range(N)] + [
        [-0.25, tr.T_y[i], i] for i in range(N)
    ]
    J_nn_ij_p = [[0.25, i, tr.T_x[i]] for i in range(N)] + [
        [0.25, i, tr.T_y[i]] for i in range(N)
    ]
    J_nn_ji_p = [[0.25, tr.T_x[i], i] for i in range(N)] + [
        [0.25, tr.T_y[i], i] for i in range(N)
    ]

    J_cccc_ij = [[-1.0, i, tr.T_x[i], tr.T_x[i], i] for i in range(N)] + [
        [-1.0, i, tr.T_y[i], tr.T_y[i], i] for i in range(N)
    ]
    J_cccc_ji = [[-1.0, tr.T_x[i], i, i, tr.T_x[i]] for i in range(N)] + [
        [-1.0, tr.T_y[i], i, i, tr.T_y[i]] for i in range(N)
    ]

    static = [["zz", J], ["+-", J], ["-+", J]]
    static_f = [
        ["nn|", J_nn_ij_p],
        ["|nn", J_nn_ji_p],
        ["n|n", J_nn_ij],
        ["n|n", J_nn_ji],
        ["+-|+-", J_cccc_ij],
        ["+-|+-", J_cccc_ji],
    ]

    E_symm = {}
    E_symm_f = {}

    #'''
    for N, (
        pcon_basis,
        basis_blocks,
        pcon_basis_f,
        basis_blocks_f,
    ) in basis_dict_combined.items():

        H_pcon = hamiltonian(static, [], basis=pcon_basis, dtype=np.float64)
        H_pcon_f = hamiltonian(static_f, [], basis=pcon_basis_f, dtype=np.float64)

        if H_pcon.Ns > 0:
            E_pcon = np.linalg.eigvalsh(H_pcon.todense())
        else:
            E_pcon = np.array([])

        if H_pcon_f.Ns > 0:
            E_pcon_f = np.linalg.eigvalsh(H_pcon_f.todense())
        else:
            E_pcon_f = np.array([])

        E_block = []
        E_block_f = []
        for basis, basis_f in zip(basis_blocks, basis_blocks_f):

            H = hamiltonian(static, [], basis=basis, dtype=np.complex128)
            H_f = hamiltonian(static_f, [], basis=basis_f, dtype=np.complex128)

            if H.Ns > 0:
                E_block.append(np.linalg.eigvalsh(H.todense()))

            if H_f.Ns > 0:
                E_block_f.append(np.linalg.eigvalsh(H_f.todense()))

        E_block = np.hstack(E_block)
        E_block.sort()

        E_block_f = np.hstack(E_block_f)
        E_block_f.sort()

        np.testing.assert_allclose(E_pcon, E_block, atol=1e-13)
        np.testing.assert_allclose(E_pcon_f, E_block_f, atol=1e-13)
        np.testing.assert_allclose(E_pcon, E_pcon_f, atol=1e-13)

        print("passed N={} sector".format(N))


test(3, 3)
