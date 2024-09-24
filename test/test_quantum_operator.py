from quspin.operators import quantum_operator
import numpy as np
import scipy.sparse as sp
import pytest


eps = 1e-13


def test_dot():
    M = np.arange(9).reshape((3, 3)).astype(np.complex128)
    v = np.ones((3, 2))
    v_fail_1 = np.ones((4,))
    v_fail_2 = np.ones((3, 1, 1))

    input_dict = {"J": [M]}

    op_dict = quantum_operator(input_dict)

    v_op = op_dict.dot(v)
    assert np.linalg.norm(v_op - M.dot(v)) < eps

    v_op = op_dict.dot(v, pars={"J": 0.5})
    assert np.linalg.norm(v_op - 0.5 * M.dot(v)) < eps

    v_op = op_dict.dot(v, pars={"J": 0.5}, check=False)
    assert np.linalg.norm(v_op - 0.5 * M.dot(v)) < eps

    with pytest.raises(ValueError):
        v_op = op_dict.dot(v_fail_1)

    with pytest.raises(ValueError):
        v_op = op_dict.dot(v_fail_2)



def test_eigsh():
    M = np.arange(16).reshape((4, 4)).astype(np.complex128)
    M = M.T + M
    input_dict = {"J": [M]}
    op_dict = quantum_operator(input_dict)

    E1, V1 = sp.linalg.eigsh(M, k=2)
    E2, V2 = op_dict.eigsh(k=2)

    assert np.linalg.norm(E1 - E2) / 2.0 < eps

    E1, V1 = sp.linalg.eigsh(0.5 * M, k=2)
    E2, V2 = op_dict.eigsh(k=2, pars={"J": 0.5})

    assert np.linalg.norm(E1 - E2) / 2.0 < eps


def test_eigh():
    M = np.arange(16).reshape((4, 4)).astype(np.complex128)
    M = M.T + M
    input_dict = {"J": [M]}
    op_dict = quantum_operator(input_dict)

    E1, V1 = np.linalg.eigh(M)
    E2, V2 = op_dict.eigh()

    assert np.linalg.norm(E1 - E2) / 4.0 < eps

    E1, V1 = np.linalg.eigh(0.5 * M)
    E2, V2 = op_dict.eigh(pars={"J": 0.5})

    assert np.linalg.norm(E1 - E2) / 4.0 < eps


def test_eigvalsh():
    M = np.arange(16).reshape((4, 4)).astype(np.complex128)
    M = M.T + M
    input_dict = {"J": [M]}
    op_dict = quantum_operator(input_dict)

    E1 = np.linalg.eigvalsh(M)
    E2 = op_dict.eigvalsh()

    assert np.linalg.norm(E1 - E2) / 4.0 < eps

    E1 = np.linalg.eigvalsh(0.5 * M)
    E2 = op_dict.eigvalsh(pars={"J": 0.5})

    assert np.linalg.norm(E1 - E2) / 4.0 < eps


if __name__ == "__main__":
    test_dot()
    test_eigsh()
    test_eigh()
    test_eigvalsh()
