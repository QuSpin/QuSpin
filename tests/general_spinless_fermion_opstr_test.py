import sys, os

qspin_path = os.path.join(os.getcwd(), "../")
sys.path.insert(0, qspin_path)

from quspin.basis import spinless_fermion_basis_1d
from quspin.basis import spinless_fermion_basis_general
import numpy as np
from itertools import product


def check_ME(b1, b2, opstr, indx, dtype, err_msg):

    if b1.Ns != b2.Ns:
        print(b1._basis)
        print(b2._basis)
        raise Exception("number of states do not match.")

    ME1, row1, col1 = b1.Op(opstr, indx, 1.0, dtype)
    ME2, row2, col2 = b2.Op(opstr, indx, 1.0, dtype)

    if len(ME1) != len(ME2):
        print(ME1)
        print(row1)
        print(col1)
        print()
        print(ME2)
        print(row2)
        print(col2)
        raise Exception("number of matrix elements do not match.")

    if len(ME1) > 0 and len(ME2) > 0:
        try:
            np.testing.assert_allclose(row1 - row2, 0, atol=1e-6, err_msg=err_msg)
            np.testing.assert_allclose(col1 - col2, 0, atol=1e-6, err_msg=err_msg)
            np.testing.assert_allclose(ME1 - ME2, 0, atol=1e-6, err_msg=err_msg)
        except:
            print(ME1)
            print(row1)
            print(col1)
            print()
            print(ME2)
            print(row2)
            print(col2)
            raise Exception


def test_gen_basis_spinless_fermion(l_max, N=4):
    L = 6
    kblocks = [None]
    kblocks.extend(range(L))
    pblocks = [None, 0, 1]

    ops = ["n", "z", "+", "-", "I"]

    Nfs = [None, N]

    t = np.array([(i + 1) % L for i in range(L)])
    p = np.array([L - i - 1 for i in range(L)])

    for Nf, kblock, pblock in product(Nfs, kblocks, pblocks):
        gen_blocks = {}
        basis_blocks = {}

        if kblock == 0 or kblock == L // 2:
            if pblock is not None:
                basis_blocks["pblock"] = (-1) ** pblock
                gen_blocks["pblock"] = (p, pblock)
            else:
                basis_blocks["pblock"] = None
                gen_blocks["pblock"] = None
        else:
            basis_blocks["pblock"] = None
            gen_blocks["pblock"] = None

        if kblock is not None:
            basis_blocks["kblock"] = kblock
            gen_blocks["kblock"] = (t, kblock)
        else:
            basis_blocks["kblock"] = None
            gen_blocks["kblock"] = None

        basis_1d = spinless_fermion_basis_1d(L, Nf=Nf, **basis_blocks)
        gen_basis = spinless_fermion_basis_general(L, Nf=Nf, **gen_blocks)
        n = basis_1d._get_norms(np.float64) ** 2
        n_gen = (gen_basis._n.astype(np.float64)) * gen_basis._pers.prod()

        if basis_1d.Ns != gen_basis.Ns:
            print(L, basis_blocks)
            print(basis_1d)
            print(gen_basis)
            raise ValueError("basis size mismatch")
        np.testing.assert_allclose(basis_1d._basis - gen_basis._basis, 0, atol=1e-6)
        np.testing.assert_allclose(n - n_gen, 0, atol=1e-6)

        for l in range(1, l_max + 1):
            for i0 in range(0, L - l + 1, 1):
                indx = range(i0, i0 + l, 1)
                for opstr in product(*[ops for i in range(l)]):
                    opstr = "".join(list(opstr))
                    printing = dict(basis_blocks)
                    printing["opstr"] = opstr
                    printing["indx"] = indx
                    printing["Nf"] = Nf

                    err_msg = "testing: {opstr:} {indx:} Nf={Nf:} kblock={kblock:} pblock={pblock:}".format(
                        **printing
                    )

                    check_ME(basis_1d, gen_basis, opstr, indx, np.complex128, err_msg)


print("testing Nf=4")
test_gen_basis_spinless_fermion(3, N=4)
print("testing Nf=5")
test_gen_basis_spinless_fermion(3, N=5)
print("testing Nf=6")
test_gen_basis_spinless_fermion(3, N=6)
