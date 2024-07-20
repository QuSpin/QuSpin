import sys, os


from quspin.operators import hamiltonian
from scipy.sparse import random, dia_matrix
import numpy as np
from itertools import product

dtypes = [np.float32, np.float64, np.complex64, np.complex128]
formats = ["dia", "csr", "csc"]


def eps(N, dtype1, dtype2):
    return N * max(np.finfo(dtype1).eps, np.finfo(dtype2).eps)


def func(t):
    return t


def func_cmplx(t):
    return 1j * t


np.random.seed(0)


def tests(N, M):
    for fmt in formats:
        for dtype1, dtype2 in product(dtypes, dtypes):
            # if dtype1!=dtype2:
            # 	continue

            for i in range(5):
                print("testing {} {} {} {}".format(fmt, dtype1, dtype2, i + 1))
                if fmt in ["csr", "csc"]:

                    A = random(N, N, density=np.log(N) / N) + 1j * random(
                        N, N, density=np.log(N) / N
                    )
                    A = (A + A.H) / 2.0
                    A = A.astype(dtype1).asformat(fmt)
                else:
                    ndiags = N // 10
                    diags = np.random.uniform(
                        0, 1, size=(ndiags, N)
                    ) + 1j * np.random.uniform(0, 1, size=(ndiags, N))
                    diags = diags.astype(dtype1)
                    offsets = np.random.choice(
                        np.arange(-N // 2 + 1, N // 2, 1), size=ndiags, replace=False
                    )
                    A = dia_matrix((diags, offsets), shape=(N, N), dtype=dtype1)

                if dtype1 in [np.complex128, np.complex64]:
                    H = hamiltonian(
                        [], [[A, func_cmplx, ()]], static_fmt=fmt, dtype=dtype1
                    )
                else:
                    H = hamiltonian([], [[A, func, ()]], static_fmt=fmt, dtype=dtype1)

                # testing single vector contiguous

                v = np.random.uniform(-1, 1, size=N) + 1j * np.random.uniform(
                    -1, 1, size=N
                )
                v = v.astype(dtype2)

                out = np.zeros_like(v, dtype=np.result_type(dtype1, dtype2))

                t = np.random.normal(0, 1)

                if dtype1 in [np.complex128, np.complex64]:
                    res2 = 1j * t * (A.dot(v))
                else:
                    res2 = t * (A.dot(v))

                res1 = H.dot(v, time=t)
                H.dot(v, time=t, out=out, overwrite_out=True)

                result_dtype = np.result_type(dtype1, dtype2)
                atol = eps(N, dtype1, dtype2)
                try:
                    np.testing.assert_allclose(res1, res2, atol=atol)
                except AssertionError as e:
                    print(res1 - res2, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(res1, out, atol=atol)
                except AssertionError as e:
                    print(res1 - out, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(out, res2, atol=atol)
                except AssertionError as e:
                    print(out - res2, atol, res1.shape)
                    raise AssertionError(e)

                # testing multi vector C to C

                v = np.random.uniform(-1, 1, size=(N, M)) + 1j * np.random.uniform(
                    -1, 1, size=(N, M)
                )
                v = v.astype(dtype2)

                out = np.zeros_like(v, dtype=np.result_type(dtype1, dtype2))

                t = np.random.normal(0, 1)

                if dtype1 in [np.complex128, np.complex64]:
                    res2 = 1j * t * (A.dot(v))
                else:
                    res2 = t * (A.dot(v))

                res1 = H.dot(v, time=t)
                H.dot(v, time=t, out=out, overwrite_out=True)

                result_dtype = np.result_type(dtype1, dtype2)
                atol = eps(N, dtype1, dtype2)
                try:
                    np.testing.assert_allclose(res1, res2, atol=atol)
                except AssertionError as e:
                    print(res1 - res2, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(res1, out, atol=atol)
                except AssertionError as e:
                    print(res1 - out, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(out, res2, atol=atol)
                except AssertionError as e:
                    print(out - res2, atol, res1.shape)
                    raise AssertionError(e)

                # testing multi vector F to F

                v = np.random.uniform(-1, 1, size=(N, M)) + 1j * np.random.uniform(
                    -1, 1, size=(N, M)
                )
                v = v.astype(dtype2, order="F")

                out = np.zeros_like(v, dtype=np.result_type(dtype1, dtype2), order="F")

                t = np.random.normal(0, 1)

                if dtype1 in [np.complex128, np.complex64]:
                    res2 = 1j * t * (A.dot(v))
                else:
                    res2 = t * (A.dot(v))

                res1 = H.dot(v, time=t)
                H.dot(v, time=t, out=out, overwrite_out=True)

                result_dtype = np.result_type(dtype1, dtype2)
                atol = eps(N, dtype1, dtype2)
                try:
                    np.testing.assert_allclose(res1, res2, atol=atol)
                except AssertionError as e:
                    print(res1 - res2, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(res1, out, atol=atol)
                except AssertionError as e:
                    print(res1, "\n", out, atol, res1.shape, out.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(out, res2, atol=atol)
                except AssertionError as e:
                    print(out - res2, atol, res1.shape)
                    raise AssertionError(e)

                continue
                # testing multi vector C to F

                v = np.random.uniform(-1, 1, size=(N, M)) + 1j * np.random.uniform(
                    -1, 1, size=(N, M)
                )
                v = v.astype(dtype2, order="C")

                out = np.zeros_like(v, dtype=np.result_type(dtype1, dtype2), order="F")

                t = np.random.normal(0, 1)

                if dtype1 in [np.complex128, np.complex64]:
                    res2 = 1j * t * (A.dot(v))
                else:
                    res2 = t * (A.dot(v))

                res1 = H.dot(v, time=t)
                H.dot(v, time=t, out=out, overwrite_out=True)

                result_dtype = np.result_type(dtype1, dtype2)
                atol = eps(N, dtype1, dtype2)
                try:
                    np.testing.assert_allclose(res1, res2, atol=atol)
                except AssertionError as e:
                    print(res1 - res2, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(res1, out, atol=atol)
                except AssertionError as e:
                    print(res1 - out, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(out, res2, atol=atol)
                except AssertionError as e:
                    print(out - res2, atol, res1.shape)
                    raise AssertionError(e)

                # testing multi vector F to C

                v = np.random.uniform(-1, 1, size=(N, M)) + 1j * np.random.uniform(
                    -1, 1, size=(N, M)
                )
                v = v.astype(dtype2, order="F")

                out = np.zeros_like(v, dtype=np.result_type(dtype1, dtype2), order="C")

                t = np.random.normal(0, 1)

                if dtype1 in [np.complex128, np.complex64]:
                    res2 = 1j * t * (A.dot(v))
                else:
                    res2 = t * (A.dot(v))

                res1 = H.dot(v, time=t)
                H.dot(v, time=t, out=out, overwrite_out=True)

                result_dtype = np.result_type(dtype1, dtype2)
                atol = eps(N, dtype1, dtype2)
                try:
                    np.testing.assert_allclose(res1, res2, atol=atol)
                except AssertionError as e:
                    print(res1 - res2, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(res1, out, atol=atol)
                except AssertionError as e:
                    print(res1 - out, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(out, res2, atol=atol)
                except AssertionError as e:
                    print(out - res2, atol, res1.shape)
                    raise AssertionError(e)

                # testing single vector C to C strided

                v = np.random.uniform(-1, 1, size=(N, M)) + 1j * np.random.uniform(
                    -1, 1, size=(N, M)
                )
                v = v.astype(dtype2, order="C")

                out = np.zeros_like(v, dtype=np.result_type(dtype1, dtype2), order="C")

                t = np.random.normal(0, 1)

                if dtype1 in [np.complex128, np.complex64]:
                    res2 = 1j * t * (A.dot(v[:, 0]))
                else:
                    res2 = t * (A.dot(v[:, 0]))

                res1 = H.dot(v[:, 0], time=t)
                H.dot(v[:, 0], time=t, out=out[:, 0], overwrite_out=True)

                result_dtype = np.result_type(dtype1, dtype2)
                atol = eps(N, dtype1, dtype2)
                try:
                    np.testing.assert_allclose(res1, res2, atol=atol)
                except AssertionError as e:
                    print(res1 - res2, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(res1, out[:, 0], atol=atol)
                except AssertionError as e:
                    print(res1 - out, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(out[:, 0], res2, atol=atol)
                except AssertionError as e:
                    print(out - res2, atol, res1.shape)
                    raise AssertionError(e)

                # testing single vector F to C strided

                v = np.random.uniform(-1, 1, size=(N, M)) + 1j * np.random.uniform(
                    -1, 1, size=(N, M)
                )
                v = v.astype(dtype2, order="F")

                out = np.zeros_like(v, dtype=np.result_type(dtype1, dtype2), order="C")

                t = np.random.normal(0, 1)

                if dtype1 in [np.complex128, np.complex64]:
                    res2 = 1j * t * (A.dot(v[:, 0]))
                else:
                    res2 = t * (A.dot(v[:, 0]))

                res1 = H.dot(v[:, 0], time=t)
                H.dot(v[:, 0], time=t, out=out[:, 0], overwrite_out=True)

                result_dtype = np.result_type(dtype1, dtype2)
                atol = eps(N, dtype1, dtype2)
                try:
                    np.testing.assert_allclose(res1, res2, atol=atol)
                except AssertionError as e:
                    print(res1 - res2, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(res1, out[:, 0], atol=atol)
                except AssertionError as e:
                    print(res1 - out, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(out[:, 0], res2, atol=atol)
                except AssertionError as e:
                    print(out - res2, atol, res1.shape)
                    raise AssertionError(e)

                # testing single vector C to F strided

                v = np.random.uniform(-1, 1, size=(N, M)) + 1j * np.random.uniform(
                    -1, 1, size=(N, M)
                )
                v = v.astype(dtype2, order="C")

                out = np.zeros_like(v, dtype=np.result_type(dtype1, dtype2), order="F")

                t = np.random.normal(0, 1)

                if dtype1 in [np.complex128, np.complex64]:
                    res2 = 1j * t * (A.dot(v[:, 0]))
                else:
                    res2 = t * (A.dot(v[:, 0]))

                res1 = H.dot(v[:, 0], time=t)
                H.dot(v[:, 0], time=t, out=out[:, 0], overwrite_out=True)

                result_dtype = np.result_type(dtype1, dtype2)
                atol = eps(N, dtype1, dtype2)
                try:
                    np.testing.assert_allclose(res1, res2, atol=atol)
                except AssertionError as e:
                    print(res1 - res2, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(res1, out[:, 0], atol=atol)
                except AssertionError as e:
                    print(res1 - out, atol, res1.shape)
                    raise AssertionError(e)

                try:
                    np.testing.assert_allclose(out[:, 0], res2, atol=atol)
                except AssertionError as e:
                    print(out - res2, atol, res1.shape)
                    raise AssertionError(e)


tests(50, 5)
tests(500, 50)


print("oputils tests passed!")
