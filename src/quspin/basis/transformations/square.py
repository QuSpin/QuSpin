import numpy as _np
from quspin.basis.transformations.site_info import site_info_2d
from itertools import product


class site_info_square(site_info_2d):
    def __init__(self, Lx, Ly):
        self._Lx = Lx
        self._Ly = Ly
        site_info_2d.__init__(self, Lx, Ly)

        self._X = self.sites % Lx
        self._Y = self.sites // Lx

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y


class square_lattice_trans(object):
    def __init__(self, Lx, Ly):
        self._Lx, self._Ly = Lx, Ly
        self._site_info = site_info_square(Lx, Ly)
        sites = self.site_info.sites
        X = self.site_info.X
        Y = self.site_info.Y

        self._Z = -(sites + 1)
        self._Z_A = _np.array(
            [
                (-(i + 1) if (x + y) % 2 == 0 else i)
                for i, (x, y) in enumerate(zip(X, Y))
            ]
        )
        self._Z_B = _np.array(
            [
                (-(i + 1) if (x + y) % 2 == 1 else i)
                for i, (x, y) in enumerate(zip(X, Y))
            ]
        )
        self._T_x = (X + 1) % Lx + Y * Lx
        self._T_y = X + ((Y + 1) % Ly) * Lx
        self._P_x = (Lx - X - 1) + Y * Lx
        self._P_y = X + (Ly - Y - 1) * Lx
        if Lx == Ly:
            self._P_d = Y + Lx * X
            self._P_e = (Ly - Y - 1) + Lx * (Lx - X - 1)
        else:
            self._P_d = None
            self._P_e = None

    @property
    def site_info(self):
        return self._site_info

    @property
    def Z(self):
        return self._Z

    @property
    def Z_A(self):
        return self._Z_A

    @property
    def Z_B(self):
        return self._Z_B

    @property
    def T_x(self):
        return self._T_x

    @property
    def T_y(self):
        return self._T_y

    @property
    def P_x(self):
        return self._P_x

    @property
    def P_y(self):
        return self._P_y

    @property
    def P_e(self):
        if self._P_e is None:
            raise Exception("P_e symmetry only exsits for square lattice")

        return self._P_e

    @property
    def P_d(self):
        if self._P_d is None:
            raise Exception("P_d symmetry only exsits for square lattice")

        return self._P_d

    def allowed_blocks_spin_inversion_iter(self, Np, sps):
        Lx = self._Lx
        Ly = self._Ly
        Z = self._Z
        nmax = sps - 1
        if (Np == nmax * (Lx * Ly) // 2 and (Lx * Ly) % 2 == 0) or Np is None:
            for blocks in self.allowed_blocks_iter():
                for zblock in range(2):
                    blocks["zblock"] = (Z, zblock)
                    yield blocks
                    blocks.pop("zblock")
        else:
            for blocks in self.allowed_blocks_iter():
                yield blocks

    def allowed_blocks_iter_parity(self):
        P_x = self._P_x
        P_y = self._P_y
        P_e = self._P_e
        P_d = self._P_d
        Lx = self._Lx
        Ly = self._Ly

        for px, py in product(range(2), range(2)):

            yield dict(pxblock=(P_x, px), pyblock=(P_y, py))
            """
			if px == py and Lx==Ly:
				for pd in range(2):
					yield dict(pxblock=(P_x,px),pyblock=(P_y,py) ,pdblock=(P_d,pd))
			else:
				for pe in range(2):
					yield dict(pxblock=(P_x,px),pyblock=(P_y,py),pdblock=(P_e,pe))
			"""

    def allowed_blocks_iter(self):
        T_x = self._T_x
        T_y = self._T_y
        P_x = self._P_x
        P_y = self._P_y
        P_e = self._P_e
        P_d = self._P_d
        Lx = self._Lx
        Ly = self._Ly

        for kx, ky in product(
            range(-Lx // 2 + 1, Lx // 2 + 1, 1), range(-Ly // 2 + 1, Ly // 2 + 1, 1)
        ):
            if kx == 0:
                if ky == 0:
                    for px, py in product(range(2), range(2)):
                        if px == py and Lx == Ly:
                            for pd in range(2):
                                yield dict(
                                    kxblock=(T_x, kx),
                                    kyblock=(T_y, ky),
                                    pxblock=(P_x, px),
                                    pyblock=(P_y, py),
                                    pdblock=(P_d, pd),
                                )
                        else:
                            yield dict(
                                kxblock=(T_x, kx),
                                kyblock=(T_y, ky),
                                pxblock=(P_x, px),
                                pyblock=(P_y, py),
                            )
                else:
                    for px in range(2):
                        yield dict(
                            kxblock=(T_x, kx), kyblock=(T_y, ky), pxblock=(P_x, px)
                        )

            elif kx == Lx // 2 and (Lx % 2 == 0):
                if ky == Ly // 2 and (Ly % 2 == 0):
                    for px, py in product(range(2), range(2)):
                        if px == py:
                            for pd in range(2):
                                yield dict(
                                    kxblock=(T_x, kx),
                                    kyblock=(T_y, ky),
                                    pxblock=(P_x, px),
                                    pyblock=(P_y, py),
                                    pdblock=(P_d, pd),
                                )
                        else:
                            yield dict(
                                kxblock=(T_x, kx),
                                kyblock=(T_y, ky),
                                pxblock=(P_x, px),
                                pyblock=(P_y, py),
                            )
                else:
                    for px in range(2):
                        yield dict(
                            kxblock=(T_x, kx), kyblock=(T_y, ky), pxblock=(P_x, px)
                        )
            else:
                if ky == 0 or (ky == Ly // 2 and Ly % 2 == 0):
                    for py in range(2):
                        yield dict(
                            kxblock=(T_x, kx), kyblock=(T_y, ky), pyblock=(P_y, py)
                        )
                elif kx == ky and (Lx == Ly):
                    for pd in range(2):
                        yield dict(
                            kxblock=(T_x, kx), kyblock=(T_y, ky), pdblock=(P_d, pd)
                        )
                elif kx == -ky and (Lx == Ly):
                    for pe in range(2):
                        yield dict(
                            kxblock=(T_x, kx), kyblock=(T_y, ky), pdblock=(P_e, pe)
                        )
