from itertools import product
from quspin.operators import hamiltonian  # Hamiltonians and operators
from quspin.operators import quantum_LinearOperator
from quspin.basis import spin_basis_general  # Hilbert space spin basis
import numpy as np  # generic math functions


def get_H(L, pblock=None, zblock=None):
    p = np.arange(L)[::-1]
    z = -(np.arange(L) + 1)

    blocks = {}

    if pblock is not None:
        blocks["pblock"] = (p, pblock)

    if zblock is not None:
        blocks["zblock"] = (z, zblock)

    basis = spin_basis_general(L, m=0, pauli=False, **blocks)

    Jzz_list = [[1.0, i, (i + 1) % L] for i in range(L)]
    Jxy_list = [[0.5, i, (i + 1) % L] for i in range(L)]
    static = [[op, Jxy_list] for op in ["+-", "-+"]] + [["zz", Jzz_list]]

    kwargs = dict(
        basis=basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    H_LO = quantum_LinearOperator(static, **kwargs)

    H = hamiltonian(static, [], **kwargs)

    return H_LO, H



def test():
    np.random.seed(0)
    for pblock, zblock in product([None, 0, 1], [None, 0, 1]):
        if pblock is not None:
            pb = (-1) ** pblock
        else:
            pb = None

        if zblock is not None:
            zb = (-1) ** zblock
        else:
            zb = None

        H_LO, H = get_H(10, pblock=pblock, zblock=zblock)

        # testing float64
        psi = np.random.normal(0, 1, size=H.Ns)
        psi /= np.linalg.norm(psi)

        psi1 = H_LO.dot(psi)
        psi2 = H.dot(psi)

        np.testing.assert_allclose(psi1, psi2, atol=1e-10, rtol=0)

        # testing complex128
        psi = np.random.normal(0, 1, size=H.Ns) + 1j * np.random.normal(0, 1, size=H.Ns)
        psi /= np.linalg.norm(psi)

        psi1 = H_LO.dot(psi)
        psi2 = H.dot(psi)

        np.testing.assert_allclose(psi1, psi2, atol=1e-10, rtol=0)

        print("testing pblock={},zblock={}".format(pb, zb))
