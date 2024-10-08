from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np
import scipy.sparse as sp

dtypes = [np.float32, np.float64, np.complex64, np.complex128]



S_dict = {
    (str(i) + "/2" if i % 2 == 1 else str(i // 2)): (i + 1, i / 2.0)
    for i in range(1, 10001)
}


def J(L, jb, length):
    blist = []
    for i, j in jb:
        b = [j]
        b.extend([(i + j) % L for j in range(length)])
        blist.append(b)

    return blist


def Jnn(L, jb, length):
    blist = []
    for i, j in jb:
        b = [j]
        b.extend([(i + j) % L for j in range(0, length + 1, 2)])
        blist.append(b)

    return blist


def getvec_spin(
    L,
    H1,
    static,
    S="1/2",
    m=None,
    kblock=None,
    pblock=None,
    zblock=None,
    pzblock=None,
    a=1,
    sparse=True,
):
    dtype = np.complex128

    b = spin_basis_1d(
        L,
        S=S,
        m=m,
        kblock=kblock,
        pblock=pblock,
        zblock=zblock,
        pzblock=pzblock,
        a=a,
    )

    Ns = b.Ns

    if Ns == 0:
        return

    H2 = hamiltonian(static, [], basis=b, dtype=dtype)

    E, v0 = H2.eigh()
    v = b.get_vec(v0, sparse=sparse)

    if sp.issparse(v):
        v = v.todense()

    H2 = H2.todense()
    H2 = v0.T.conj() * (H2 * v0)
    H1 = v.T.conj().dot(H1.dot(v))
    err_msg = "get_vec() symmetries failed for L={0} {1}".format(b.N, b.blocks)
    np.testing.assert_allclose(H1, H2, atol=1e-10, err_msg=err_msg)


def getvec_spin_sublattice(
    L, H1, static, S="1/2", kblock=None, zAblock=None, zBblock=None, a=2, sparse=True
):
    dtype = np.complex128

    b = spin_basis_1d(L, S=S, kblock=kblock, zAblock=zAblock, zBblock=zBblock, a=a)

    Ns = b.Ns

    if Ns == 0:
        return

    H2 = hamiltonian(static, [], N=L, basis=b, dtype=dtype)

    E, v0 = H2.eigh()
    v = b.get_vec(v0, sparse=sparse)

    if sp.issparse(v):
        v = v.todense()

    H2 = H2.todense()
    H2 = v0.T.conj() * (H2 * v0)
    H1 = v.T.conj() * (H1 * v)
    err_msg = "get_vec() symmetries failed for L={0} {1}".format(b.N, b.blocks)
    np.testing.assert_allclose(H1, H2, atol=1e-10, err_msg=err_msg)


def check_getvec_spin(L, S="1/2", a=1, sparse=True):
    jb = [[i, 1.0] for i in range(L)]
    dtype = np.complex128
    static = [
        ["xx", J(L, jb, 2)],
        ["yy", J(L, jb, 2)],
        ["zz", J(L, jb, 2)],
        ["+-", J(L, jb, 2)],
        ["-+", J(L, jb, 2)],
        ["zzzz", J(L, jb, 4)],
        ["xxxx", J(L, jb, 4)],
        ["yyyy", J(L, jb, 4)],
        ["xxzz", J(L, jb, 4)],
        ["zzxx", J(L, jb, 4)],
        ["yyzz", J(L, jb, 4)],
        ["zzyy", J(L, jb, 4)],
        ["yyxx", J(L, jb, 4)],
        ["xxyy", J(L, jb, 4)],
        ["+zz-", J(L, jb, 4)],
        ["-zz+", J(L, jb, 4)],
        ["+xx-", J(L, jb, 4)],
        ["-xx+", J(L, jb, 4)],
        ["+yy-", J(L, jb, 4)],
        ["-yy+", J(L, jb, 4)],
        ["++--", J(L, jb, 4)],
        ["--++", J(L, jb, 4)],
        ["+-+-", J(L, jb, 4)],
        ["-+-+", J(L, jb, 4)],
    ]

    b_full = spin_basis_1d(L,S=S)

    if S != "1/2":
        static, _ = b_full.expanded_form(static, [])

    H1 = hamiltonian(static, [], basis=b_full, dtype=dtype).todense()

    for k in range(-L // a, L // a):
        getvec_spin(L, H1, static, S=S, kblock=k, a=a, sparse=sparse)

    for j in range(-1, 2, 2):
        getvec_spin(L, H1, static, S=S, pblock=j, a=a, sparse=sparse)
        for k in range(-L // a, L // a):
            getvec_spin(L, H1, static, S=S, kblock=k, pblock=j, a=a, sparse=sparse)

    m = None
    sps, S_val = S_dict[S]

    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            getvec_spin(
                L, H1, static, S=S, m=m, pblock=i, zblock=j, a=a, sparse=sparse
            )
            for k in range(-L // a, L // a):
                getvec_spin(
                    L,
                    H1,
                    static,
                    S=S,
                    kblock=k,
                    m=m,
                    pblock=i,
                    zblock=j,
                    a=a,
                    sparse=sparse,
                )

    for j in range(-1, 2, 2):
        getvec_spin(L, H1, static, S=S, m=m, pzblock=j, a=a, sparse=sparse)
        for k in range(-L // a, L // a):
            getvec_spin(
                L, H1, static, S=S, kblock=k, m=m, pzblock=j, a=a, sparse=sparse
            )

    for j in range(-1, 2, 2):
        getvec_spin(L, H1, static, S=S, m=m, zblock=j, a=a)
        for k in range(-L // a, L // a):
            getvec_spin(
                L, H1, static, S=S, kblock=k, m=m, zblock=j, a=a, sparse=sparse
            )

    for Nup in range(L + 1):
        for k in range(-L // a, L // a):
            getvec_spin(L, H1, static, S=S, m=Nup/L-S_val, kblock=k, a=a, sparse=sparse)

    for Nup in range(0, L + 1):
        for j in range(-1, 2, 2):
            getvec_spin(L, H1, static, S=S, m=Nup/L-S_val, pblock=j, a=a, sparse=sparse)
            for k in range(-L // a, L // a):
                getvec_spin(
                    L, H1, static, S=S, kblock=k, m=Nup/L-S_val, pblock=j, a=a, sparse=sparse
                )

    if (L % 2) == 0:
        m=0
        for i in range(-1, 2, 2):
            for j in range(-1, 2, 2):
                getvec_spin(
                    L, H1, static, S=S, m=m, pblock=i, zblock=j, a=a, sparse=sparse
                )
                for k in range(-L // a, L // a):
                    getvec_spin(
                        L,
                        H1,
                        static,
                        S=S,
                        kblock=k,
                        m=m,
                        pblock=i,
                        zblock=j,
                        a=a,
                        sparse=sparse,
                    )

        for j in range(-1, 2, 2):
            getvec_spin(L, H1, static, S=S, m=m, pzblock=j, a=a, sparse=sparse)
            for k in range(-L // a, L // a):
                getvec_spin(
                    L, H1, static, S=S, kblock=k, m=m, pzblock=j, a=a, sparse=sparse
                )

        for j in range(-1, 2, 2):
            getvec_spin(L, H1, static, S=S, m=m, zblock=j, a=a, sparse=sparse)
            for k in range(-L // a, L // a):
                getvec_spin(
                    L, H1, static, S=S, kblock=k, m=m, zblock=j, a=a, sparse=sparse
                )


def check_getvec_spin_sublattice(L, S="1/2", a=2, sparse=True):
    jb = [[i, 1.0] for i in range(L)]
    dtype = np.complex128
    static = [
        ["xx", J(L, jb, 2)],
        ["zz", Jnn(L, jb, 2)],
        ["zxz", J(L, jb, 3)],
        ["zxzx", J(L, jb, 4)],
        ["xxxx", J(L, jb, 4)],
        ["yyyy", J(L, jb, 4)],
        ["xzxz", J(L, jb, 4)],
        ["yzyz", J(L, jb, 4)],
        ["zyzy", J(L, jb, 4)],
        ["z+z-", J(L, jb, 4)],
        ["z-z+", J(L, jb, 4)],
    ]

    b_full = spin_basis_1d(L, S=S)

    if S != "1/2":
        static, _ = b_full.expanded_form(static, [])

    H1 = hamiltonian(static, [], basis=b_full, dtype=dtype).todense()

    for k in range(-L // a, L // a):
        getvec_spin_sublattice(L, H1, static, S=S, kblock=k, a=a, sparse=sparse)

    for j in range(-1, 2, 2):
        getvec_spin_sublattice(L, H1, static, S=S, zAblock=j, a=a, sparse=sparse)
        for k in range(-L // a, L // a):
            getvec_spin_sublattice(L, H1, static, S=S, kblock=k, zAblock=j, a=a, sparse=sparse)

    for j in range(-1, 2, 2):
        getvec_spin_sublattice(L, H1, static, S=S, zBblock=j, a=a, sparse=sparse)
        for k in range(-L // a, L // a):
            getvec_spin_sublattice(L, H1, static, S=S, kblock=k, zBblock=j, a=a, sparse=sparse)

    for i in range(-1, 2, 2):
        for j in range(-1, 2, 2):
            getvec_spin_sublattice(L, H1, static, S=S, zAblock=i, zBblock=j, a=a, sparse=sparse)
            for k in range(-L // a, L // a):
                getvec_spin_sublattice(
                    L,
                    H1,
                    static,
                    S=S,
                    kblock=k,
                    zAblock=i,
                    zBblock=j,
                    a=a,
                    sparse=sparse,
                )

def test():

    check_getvec_spin(6, S="1/2", sparse=True)
    check_getvec_spin(6, S="1/2", sparse=False)
    check_getvec_spin(6, S="1/2", sparse=True)
    check_getvec_spin(6, S="1/2", sparse=False)

    check_getvec_spin(6, S="1", sparse=True)
    check_getvec_spin(6, S="1", sparse=False)
    check_getvec_spin(6, S="1", sparse=True)
    check_getvec_spin(6, S="1", sparse=False)


def test_sublattice():
    check_getvec_spin_sublattice(6, S="1/2", sparse=True)
    check_getvec_spin_sublattice(6, S="1/2", sparse=False)
    check_getvec_spin_sublattice(6, S="1/2", sparse=True)
    check_getvec_spin_sublattice(6, S="1/2", sparse=False)

    check_getvec_spin_sublattice(6, S="1", sparse=True)
    check_getvec_spin_sublattice(6, S="1", sparse=False)
    check_getvec_spin_sublattice(6, S="1", sparse=True)
    check_getvec_spin_sublattice(6, S="1", sparse=False)


