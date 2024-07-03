import numpy as _np


class site_info(object):
    def __init__(self, N):
        self._N = N
        self._sites = _np.arange(N)

    @property
    def N(self):
        return self._N

    @property
    def sites(self):
        return self._sites


class site_info_2d(site_info):
    def __init__(self, Lx, Ly):
        site_info.__init__(self, Lx * Ly)

    @property
    def coor_iter(self):
        return enumerate(zip(self._X, self._Y))
