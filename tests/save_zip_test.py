import sys, os


from quspin.operators import quantum_operator, save_zip, load_zip
from quspin.basis import spin_basis_general
import numpy as np
import tempfile, os


def Jr(r, alpha):
    return (-1) ** (r + 1) / r ** (alpha)


L = 10
alpha = 2.0

t = (np.arange(L) + 1) % L
p = np.arange(L)[::-1]
z = -(np.arange(L) + 1)

basis = spin_basis_general(L, m=0.0, t=(t, 0), p=(p, 0), z=(z, 0), pauli=False)


Jzz_list = [
    [Jr(r, alpha), i, (i + r) % L] for i in range(L) for r in range(1, L // 2, 1)
]
Jxy_list = [
    [Jr(r, alpha) / 2.0, i, (i + r) % L] for i in range(L) for r in range(1, L // 2, 1)
]
ops = dict(
    Jxy=[[op, Jxy_list] for op in ["+-", "-+"]],
    Jzz=[["zz", Jzz_list]],
    Jd=[np.random.normal(0, 1, size=(basis.Ns, basis.Ns))],
)

op = quantum_operator(
    ops,
    basis=basis,
    dtype=np.float32,
    matrix_formats=dict(Jzz="dia", Jxy="csr", Jd="dense"),
)


with tempfile.TemporaryDirectory() as tmpdirname:
    file = os.path.join(tmpdirname, "test_save.zip")
    save_zip(file, op, save_basis=True)
    new_op_1 = load_zip(file)
    assert len((op - new_op_1)._quantum_operator) == 0
    save_zip(file, op, save_basis=False)
    new_op_2 = load_zip(file)
    assert len((op - new_op_2)._quantum_operator) == 0
