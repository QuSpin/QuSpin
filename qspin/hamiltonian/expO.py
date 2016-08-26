from ..hamiltonian import hamiltonian, ishamiltonian
from scipy.sparse.linalg import expm_multiply as _expm_multiply
import numpy as _np
import scipy.sparse as _sp


class expO(object):
    def __init__(self, O):
        if ishamiltonian(O):
            self._O = O
        else:
            self._O = hamiltonian([O], [])

    @property
    def O(self):
        return self._O

    @property
    def get_shape(self):
        return self.O.get_shape

    def get_mat(self, a=-1j, time=0):
        if not _np.isscalar(a):
            raise TypeError('expecting scalar argument for a')

        if self.O.is_dense:
            return _np.linalg.expm(a * self.O.todense(time))
        else:
            return _sp.linalg.expm(a * self.O.tocsr(time).tocsc())

    def dot(self, other, a=-1j, time=0, start=None, stop=None, num=None, endpoint=None, iterate=False):

        if ishamiltonian(other):
            raise ValueError("expO not compatible with hamiltonian objects.")

        sparse_list = False

        if _sp.issparse(other):
            sparse_list = True

        elif other.__class__ not in [_np.matrix, _np.ndarray]:
            other = _np.asanyarray(other)

        if other.ndim not in [1, 2]:
            raise ValueError("Expecting a 1 or 2 dimensional array for 'other'")

        if other.shape[0] != self.get_shape[1]:
            raise ValueError("Dimension mismatch between expO: {0} and other: {1}".format(self._O.get_shape, other.shape))

        M = a * self.O(time)
        if iterate:
            if start is None and stop is None:
                raise ValueError("iterate option only availible for time discretization.")

            grid, step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)
            return _iter_dot(other, M, step, grid)
        else:
            if sparse_list:
                grid, step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)
                mats = _iter_dot(other, M, step, grid)
                return _np.array([mat for mat in mats])
            else:
                return _expm_multiply(M, other.T.conj(), start=start, stop=stop, num=num, endpoint=endpoint)

    def rdot(self, other, a=-1j, time=0, start=None, stop=None, num=None, endpoint=None, iterate=False):
        if ishamiltonian(other):
            raise ValueError("expO not compatible with hamiltonian objects.")

        sparse_list = False

        if _sp.issparse(other):
            sparse_list = True

        elif other.__class__ not in [_np.matrix, _np.ndarray]:
            other = _np.asanyarray(other)

        if other.ndim not in [1, 2]:
            raise ValueError("Expecting a 1 or 2 dimensional array for 'other'")

        if other.shape[1] != self.get_shape[0]:
            raise ValueError("Dimension mismatch between expO: {0} and other: {1}".format(self._O.get_shape, other.shape))

        M = (a * self.O(time)).T.conj()
        if iterate:
            if start is None and stop is None:
                raise ValueError("iterate option only availible for time discretization.")

            grid, step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)
            return _iter_rdot(other.T.conj(), M, step, grid)
        else:
            print sparse_list
            if sparse_list:
                grid, step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)
                mats = _iter_rdot(other.T.conj(), M, step, grid)
                return _np.array([mat for mat in mats])
            else:
                return _expm_multiply(M, other.T.conj(), start=start, stop=stop, num=num, endpoint=endpoint)

    def trans(self, other, a=-1j, time=0, start=None, stop=None, num=None, endpoint=None, iterate=False):
        if ishamiltonian(other):
            raise ValueError("expO not compatible with hamiltonian objects.")

        if not (_sp.issparse(other) or other.__class__ in [_np.matrix, _np.ndarray]):
            other = _np.asanyarray(other)

        if other.ndim != 2:
            raise ValueError("Expecting a 2 dimensional array for 'other'")

        if other.shape[0] != other.shape[1]:
            raise ValueError("Expecting square array for 'other'")

        if other.shape[0] != self.get_shape[0]:
            raise ValueError("Dimension mismatch between expO: {0} and other: {1}".format(self.get_shape, other.shape))

        M = a * self.O(time)
        if iterate:
            if [start, stop] == [None, None]:
                raise ValueError("iterate option only availible for time discretization.")

            grid, step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)
            return _iter_trans(M, other, step, grid)
        else:
            if [start, stop] == [None, None]:

                if [num, endpoint] != [None, None]:
                    raise ValueError('impropor linspace arguements!')

                other = self.dot(other, a=a, time=time)
                return self.rdot(other, a=a, time=time)

            else:
                grid, step = _np.linspace(start, stop, num=num, endpoint=endpoint, retstep=True)

                mat_iter = _iter_trans(M, other, step, grid)

                others = _np.asarray([mat for mat in mat_iter])

                return others


def _iter_dot(M, other, step, grid):
    if grid[0] != 0:
        M *= grid[0]
        other = _expm_multiply(M, other)
        M /= grid[0]

    yield other.copy()

    M *= step
    for t in grid[1:]:
        other = _expm_multiply(M, other)
        yield other.copy()


def _iter_rdot(M, other, step, grid):
    if grid[0] != 0:
        M *= grid[0]
        other = _expm_multiply(M, other)
        M /= grid[0]

    yield _np.array(other.T.conj())

    M *= step
    for t in grid[1:]:
        other = _expm_multiply(M, other)
        yield _np.array(other.T.conj())


def _iter_trans(M, other, step, grid):
    if grid[0] != 0:
        M *= grid[0]
        other = _expm_multiply(M, other)
        r_other = _expm_multiply(M.T.conj(), other.T.conj()).T.conj()
        M /= grid[0]

        yield r_other.copy()
    else:
        yield other.copy()

    for t in grid[1:]:
        M *= step
        other = _expm_multiply(M, other)
        M /= step
        if t != 0:
            M *= t
            r_other = _expm_multiply(M.T.conj(), other.T.conj()).T.conj()
            M /= t

        yield r_other.copy()
