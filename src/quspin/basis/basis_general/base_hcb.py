from quspin_extensions.basis.basis_general._basis_general_core import (
    hcb_basis_core_wrap,
)
from quspin_extensions.basis.basis_general._basis_general_core import (
    get_basis_type,
    basis_zeros,
)
from quspin.basis.basis_general.base_general import basis_general
import numpy as _np
from scipy.special import comb
import cProfile


# general basis for hardcore bosons/spin-1/2
class hcb_basis_general(basis_general):
    def __init__(
        self,
        N,
        Nb=None,
        Ns_block_est=None,
        _Np=None,
        _make_basis=True,
        block_order=None,
        **kwargs,
    ):
        basis_general.__init__(self, N, block_order=block_order, **kwargs)
        self._check_pcon = False
        self._count_particles = False
        if _Np is not None and Nb is None:
            self._count_particles = True
            if type(_Np) is not int:
                raise ValueError("_Np must be integer")
            if _Np >= -1:
                if _Np + 1 > N:
                    Nb = list(range(N + 1))
                elif _Np == -1:
                    Nb = None
                else:
                    Nb = list(range(_Np + 1))
            else:
                raise ValueError(
                    "_Np == -1 for no particle conservation, _Np >= 0 for particle conservation"
                )

        if Nb is None:
            self._Ns = 1 << N
        elif type(Nb) is int:
            self._check_pcon = True
            self._get_proj_pcon = True
            self._Ns = comb(N, Nb, exact=True)
        else:
            try:
                Np_iter = iter(Nb)
            except TypeError:
                raise TypeError("Nb must be integer or iteratable object.")
            Nb = list(Nb)
            self._Ns = 0
            for b in Nb:
                if b > N or b < 0:
                    raise ValueError("particle number Nb must satisfy: 0 <= Nb <= N")
                self._Ns += comb(N, b, exact=True)

        if len(self._pers) > 0:
            if Ns_block_est is None:
                self._Ns = int(float(self._Ns) / _np.multiply.reduce(self._pers)) * 2
            else:
                if type(Ns_block_est) is not int:
                    raise TypeError("Ns_block_est must be integer value.")

                self._Ns = Ns_block_est

        # create basis constructor
        self._basis_dtype = get_basis_type(N, None, 2)
        self._core = hcb_basis_core_wrap(
            self._basis_dtype, N, self._maps, self._pers, self._qs
        )
        self._N = N
        self._Ns_block_est = self._Ns
        self._Np = Nb
        self._sps = 2
        self._allowed_ops = set(["I", "x", "y", "z", "+", "-", "n"])

        # make the basis; make() is function method of base_general
        if _make_basis:
            self.make()
        else:
            self._Ns = 1
            self._basis = basis_zeros(self._Ns, dtype=self._basis_dtype)
            self._n = _np.zeros(self._Ns, dtype=self._n_dtype)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._core = hcb_basis_core_wrap(
            self._basis_dtype, self._N, self._maps, self._pers, self._qs
        )
