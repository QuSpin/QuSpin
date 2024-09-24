from quspin.basis import spin_basis_general
import numpy as np
from numpy.random import seed


def check(basis, pcon=False):
    seed(0)


    P = basis.get_proj(np.complex128, pcon=pcon)

    Ns_full, Ns = P.shape

    v = np.random.normal(size=Ns).astype(np.complex128)
    v /= np.linalg.norm(v)

    v_full = np.random.normal(size=Ns_full).astype(np.complex128)
    v_full /= np.linalg.norm(v_full)

    err_msg = "get_vec/get_vec_inv test failed for L={0}".format(basis.__class__)

    np.testing.assert_allclose(
        P.dot(v), basis.project_from(v, sparse=False), atol=1e-10, err_msg=err_msg
    )
    np.testing.assert_allclose(
        P.H.dot(v_full),
        basis.project_to(v_full, sparse=False),
        atol=1e-10,
        err_msg=err_msg,
    )


def test():
    seed(0)

    L = 12
    assert L >= 3

    z = -(np.arange(L) + 1)
    p = np.arange(L)[::-1]
    t = (np.arange(L) + 1) % L

    bases = [
        spin_basis_general(L),
        spin_basis_general(L, Nup=L // 2),
        spin_basis_general(L, zb=(z, 0)),
        spin_basis_general(L, zb=(z, 1)),
        spin_basis_general(L, pb=(p, 0)),
        spin_basis_general(L, pb=(p, 1)),
        spin_basis_general(L, zb=(z, 0), pb=(p, 0)),
        spin_basis_general(L, zb=(z, 0), pb=(p, 1)),
        spin_basis_general(L, zb=(z, 1), pb=(p, 0)),
        spin_basis_general(L, zb=(z, 1), pb=(p, 1)),
        spin_basis_general(L, zb=(z, 0), pb=(t, 0)),
        spin_basis_general(L, zb=(z, 0), pb=(t, 1)),
        spin_basis_general(L, zb=(z, 1), pb=(t, 0)),
        spin_basis_general(L, zb=(z, 1), pb=(t, 1)),
        spin_basis_general(L, zb=(z, 0), pb=(p, 0), tb=(t, 0)),
        spin_basis_general(L, zb=(z, 0), pb=(p, 0), tb=(t, L - 1)),
        spin_basis_general(L, zb=(z, 0), pb=(p, 1), tb=(t, 0)),
        spin_basis_general(L, zb=(z, 1), pb=(p, 0), tb=(t, L - 1)),
        spin_basis_general(L, zb=(z, 1), pb=(p, 1), tb=(t, 0)),
        spin_basis_general(L, Nup=L // 2, pb=(p, 0), tb=(t, 0)),
        spin_basis_general(L, Nup=2, pb=(p, 0), tb=(t, L - 1)),
        spin_basis_general(L, Nup=3, pb=(p, 1), tb=(t, 0)),
        spin_basis_general(L, Nup=1, pb=(p, 0), tb=(t, L - 2)),
        spin_basis_general(L, Nup=L // 2, pb=(p, 1), tb=(t, L - 1)),
    ]

    for basis in bases:
        check(basis)
