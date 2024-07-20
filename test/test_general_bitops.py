import sys, os


import numpy as np
from quspin.basis import spin_basis_general, spin_basis_1d, basis_int_to_python_int
from quspin.basis import (
    bitwise_not,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    bitwise_leftshift,
    bitwise_rightshift,
)


# test doc example

N = 100  # sites
basis = spin_basis_general(N, Nup=1)  # 1 particle
x = [basis.states[0]]  # large integer stored as a byte
not_x = bitwise_not(x)
print("original integer stored as a byte:")
print(x)
x_int = basis_int_to_python_int(x)  # cast byte as python integer
print("original integer in integer form:")
print(x_int)
print("result in integer form:")
print(basis_int_to_python_int(not_x))


# test functions


def test(y1, y2):
    np.testing.assert_allclose(y1, y2)


def test_large_ints(y1, y2):
    for s_general, s_1d in zip(y1, y2):
        # print(s_1d)
        # print(basis_int_to_python_int(s_general))
        # exit()
        assert basis_int_to_python_int(s_general) == s_1d


def initiate(x1):
    where = np.ones(x1.shape, dtype=bool)
    out = np.zeros_like(x1)
    return out, where


def run_funcs(x1, x2, b):
    # test NOT
    out, where = initiate(x1)
    y1_np = np.invert(x1)
    y1 = bitwise_not(x1)
    y1_where = bitwise_not(x1, where=where)
    bitwise_not(x1, where=where, out=out)
    test(y1, y1_np)
    test(y1, y1_where)
    test(y1, out)

    # test AND, OR, XOR
    funcs = [
        (np.bitwise_and, bitwise_and),
        (np.bitwise_or, bitwise_or),
        (np.bitwise_xor, bitwise_xor),
    ]
    for numpy_func, quspin_func in funcs:
        out, where = initiate(x1)
        y1_np = numpy_func(x1, x2)
        y1 = quspin_func(x1, x2)
        y1_where = quspin_func(x1, x2, where=where)
        quspin_func(x1, x2, where=where, out=out)
        test(y1, y1_np)
        test(y1, y1_where)
        test(y1, out)

    # test shifts
    funcs = [(np.left_shift, bitwise_leftshift), (np.right_shift, bitwise_rightshift)]
    for numpy_func, quspin_func in funcs:
        out, where = initiate(x1)
        y1_np = numpy_func(x1, b)
        y1 = quspin_func(x1, b)
        y1_where = quspin_func(x1, b, where=where)
        quspin_func(x1, b, where=where, out=out)
        test(y1, y1_np)
        test(y1, y1_where)
        test(y1, out)


def run_funcs_large_ints(x1, x2, b, z1, z2, d):
    # test NOT
    out, where = initiate(x1)
    out_py, where_py = initiate(z1)

    """
	y1_py=np.invert(z1)
	y1=bitwise_not(x1)
	bitwise_not(x1,where=where,out=out)
	np.invert(z1,where=where_py,out=out_py)
	test_large_ints(y1,y1_py)
	test_large_ints(out,out_py)
	"""

    # test AND, OR, XOR
    funcs = [
        (np.bitwise_and, bitwise_and),
        (np.bitwise_or, bitwise_or),
        (np.bitwise_xor, bitwise_xor),
    ]
    for numpy_func, quspin_func in funcs:
        out, where = initiate(x1)
        out_py, where_py = initiate(z1)
        y1_py = numpy_func(z1, z2)
        y1 = quspin_func(x1, x2)
        numpy_func(z1, z2, where=where_py, out=out_py)
        quspin_func(x1, x2, where=where, out=out)
        test_large_ints(y1, y1_py)
        test_large_ints(out, out_py)

    # test shifts
    funcs = [(np.left_shift, bitwise_leftshift), (np.right_shift, bitwise_rightshift)]
    for numpy_func, quspin_func in funcs:
        out, where = initiate(x1)
        out_py, where_py = initiate(z1)
        y1_py = numpy_func(z1, d)
        y1 = quspin_func(x1, b)
        numpy_func(z1, d, where=where_py, out=out_py)
        quspin_func(x1, b, where=where, out=out)
        test_large_ints(y1, y1_py)
        test_large_ints(out, out_py)


for N, L in [(4, 400), (6, 600), (8, 800)]:

    # test general basis dtypes
    basis = spin_basis_general(N)
    x1 = basis.states[: 2 ** (N - 1)]
    x2 = basis.states[2 ** (N - 1) :]
    b = 1 * np.ones(x1.shape, dtype=np.uint32)  # shift by b bits
    run_funcs(x1, x2, b)

    # test basis 1d dtypes # basis_int_to_python_int, loop
    basis_py = spin_basis_1d(L, Nup=1)
    z1 = basis_py.states[: L // 2]
    z2 = basis_py.states[L // 2 :]
    d = 2 * np.ones(z1.shape, dtype=np.uint32)  # shift by b bits
    basis = spin_basis_general(L, Nup=1)
    x1 = basis.states[: L // 2]
    x2 = basis.states[L // 2 :]
    b = 2 * np.ones(x1.shape, dtype=np.uint32)  # shift by b bits
    run_funcs_large_ints(x1, x2, b, z1, z2, d)

    print("passed general bitwise_ops test for N={0:d}".format(N))
