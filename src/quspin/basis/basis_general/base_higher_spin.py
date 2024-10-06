from quspin_extensions.basis.basis_general._basis_general_core import (
    higher_spin_basis_core_wrap,
    get_basis_type,
    basis_zeros,
)
from quspin.basis.basis_general.base_general import basis_general
from quspin.basis.basis_general.boson import H_dim
import numpy as _np
from scipy.special import comb


# general basis for higher spin representations
class higher_spin_basis_general(basis_general):
    def __init__(
        self,
        N,
        sps,
        Nup=None,
        Ns_block_est=None,
        _Np=None,
        _make_basis=True,
        block_order=None,
        **kwargs,
    ):
        basis_general.__init__(self, N, block_order=block_order, **kwargs)
        self._check_pcon = False
        self._count_particles = False
        if _Np is not None and Nup is None:
            self._count_particles = True
            if type(_Np) is not int:
                raise ValueError("_Np must be integer")
            if _Np >= -1:
                if _Np + 1 > N:
                    Nup = list(range(N + 1))
                elif _Np == -1:
                    Nup = None
                else:
                    Nup = list(range(_Np + 1))
            else:
                raise ValueError(
                    "_Np == -1 for no particle conservation, _Np >= 0 for particle conservation"
                )

        if sps is None:
            raise ValueError("sps required for higher_spin_core")

        if Nup is None:
            self._Ns = sps**N
            self._basis_dtype = get_basis_type(N, Nup, sps)
        elif type(Nup) is int:
            self._check_pcon = True
            self._get_proj_pcon = True
            self._Ns = H_dim(Nup, N, sps - 1)
            self._basis_dtype = get_basis_type(N, Nup, sps)
        else:
            try:
                Np_iter = iter(Nup)
            except TypeError:
                raise TypeError("Nup must be integer or iteratable object.")
            self._Ns = 0
            for Nup in Np_iter:
                self._Ns += H_dim(Nup, N, sps - 1)

            self._basis_dtype = get_basis_type(N, max(iter(Nup)), sps)

        if len(self._pers) > 0:
            if Ns_block_est is None:
                self._Ns = int(float(self._Ns) / _np.multiply.reduce(self._pers)) * sps
            else:
                if type(Ns_block_est) is not int:
                    raise TypeError("Ns_block_est must be integer value.")
                if Ns_block_est <= 0:
                    raise ValueError("Ns_block_est must be an integer > 0")

                self._Ns = Ns_block_est

        self._basis_dtype = get_basis_type(N, Nup, sps)
        self._core = higher_spin_basis_core_wrap(
            self._basis_dtype, N, sps, self._maps, self._pers, self._qs
        )

        self._N = N
        self._Ns_block_est = self._Ns
        self._Np = Nup
        self._sps = sps
        self._allowed_ops = set(["I", "z", "+", "-"])

        # make the basisl; make() is function method of base_general
        if _make_basis:
            self.make()
        else:
            self._Ns = 1
            self._basis = basis_zeros(self._Ns, dtype=self._basis_dtype)
            self._n = _np.zeros(self._Ns, dtype=self._n_dtype)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._core = higher_spin_basis_core_wrap(
            self._basis_dtype, self._N, self._sps, self._maps, self._pers, self._qs
        )
