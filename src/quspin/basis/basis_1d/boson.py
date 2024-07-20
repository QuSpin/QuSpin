from quspin_extensions.basis.basis_1d._basis_1d_core import hcp_basis, hcp_ops
from quspin_extensions.basis.basis_1d._basis_1d_core import boson_basis, boson_ops
from quspin.basis.basis_1d.base_1d import basis_1d
import numpy as _np


class boson_basis_1d(basis_1d):
    """Constructs basis for boson operators in a specified 1-d symmetry sector.

    The supported operator strings for `boson_basis_1d` are:

    .. math::
            \\begin{array}{cccc}
                    \\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}     \\newline
                    \\texttt{boson_basis_1d}&   \\hat{1}        &   \\hat b^\\dagger      &       \\hat b          & \\hat b^\\dagger b     &  \\hat b^\\dagger\\hat b - \\frac{\\mathrm{sps}-1}{2}  \\newline
            \\end{array}

    Notes
    -----
    * if `Nb` or `nb` are specified, by default `sps` is set to the number of bosons on the lattice.
    * if `sps` is specified, while `Nb` or `nb` are not, all particle sectors are filled up to the maximumal
            occupation.
    * if `Nb` or `nb` and `sps` are specified, the finite boson basis is constructed with the local Hilbert space
            restrited by `sps`.

    Examples
    --------

    The code snippet below shows how to use the `boson_basis_1d` class to construct the basis in the zero momentum sector of positive parity for the bosonic Hamiltonian

    .. math::
            H(t)=-J\\sum_j b^\\dagger_{j+1}b_j + \\mathrm{h.c.} -\\mu\\sum_j n_j + U\\sum_j n_j n_j + g\\cos\\Omega t\\sum_j (b^\\dagger_j + b_j)

    .. literalinclude:: ../../doc_examples/boson_basis_1d-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(self, L, Nb=None, nb=None, sps=None, **blocks):
        """Intializes the `boson_basis_1d` object (basis for bosonic operators).

        Parameters
        ----------
        L: int
                Length of chain/number of sites.
        Nb: {int,list}, optional
                Number of bosons in chain. Can be integer or list to specify one or more particle sectors.
        nb: float, optional
                Density of bosons in chain (bosons per site).
        sps: int, optional
                Number of states per site (including zero bosons), or on-site Hilbert space dimension.
        **blocks: optional
                Extra keyword arguments which include:

                        **a** (*int*) - specifies unit cell size for translation.

                        **kblock** (*int*) - specifies momentum block. The physical manifestation of this symmetry transformation is translation by `a` lattice sites.

                        **pblock** (*int*) - specifies parity block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain.

                and the following which only work for hardcore bosons (`sps=2`):

                        **cblock** (*int*) - specifies particle-hole symmetry block. The physical manifestation of this symmetry transformation is the exchange of a hard-core boson for a hole (i.e. no particle).

                        **pcblock** (*int*) - specifies parity followed by particle-hole symmetry block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain, and a simultaneous exchange of a hard-core boson for a hole (i.e. no particle).

                        **cAblock** (*int*) - specifies particle-hole symmetry block for sublattice A (defined as all even lattice sites). The physical manifestation of this symmetry transformation is the exchange of a hard-core boson for a hole (i.e. no particle) on all even sites.

                        **cBblock** (*int*) - specifies particle-hole symmetry block for sublattice B (defined as all odd lattice sites). The physical manifestation of this symmetry transformation is the exchange of a hard-core boson for a hole (i.e. no particle) on all odd sites.

        """

        input_keys = set(blocks.keys())

        expected_keys = set(
            [
                "_Np",
                "kblock",
                "cblock",
                "cAblock",
                "cBblock",
                "pblock",
                "pcblock",
                "a",
                "check_z_symm",
                "L",
            ]
        )
        wrong_keys = input_keys - expected_keys
        if wrong_keys:
            temp = ", ".join(["{}" for key in wrong_keys])
            raise ValueError(
                ("unexpected optional argument(s): " + temp).format(*wrong_keys)
            )

        if blocks.get("a") is None:  # by default a = 1
            blocks["a"] = 1

        if blocks.get("check_z_symm") is None or blocks.get("check_z_symm") is True:
            check_z_symm = True
        else:
            check_z_symm = False

        if sps is None:
            if Nb is not None:
                if nb is not None:
                    raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")
            elif nb is not None:
                if Nb is not None:
                    raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")
                Nb = int(nb * L)
            else:
                raise ValueError("expecting value for 'Nb','nb' or 'sps'")
        else:
            if Nb is not None:
                if nb is not None:
                    raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")
            elif nb is not None:
                Nb = int(nb * L)

        self._sps = sps

        if Nb is None:
            Nb_list = None
        elif type(Nb) is int:
            Nb_list = [Nb]
        else:
            try:
                Nb_list = list(Nb)
            except TypeError:
                raise TypeError("Nb must be iterable returning integers")

            if any((type(Nb) is not int) for Nb in Nb_list):
                TypeError("Nb must be iterable returning integers")

        count_particles = False
        if blocks.get("_Np") is not None:
            _Np = blocks.get("_Np")
            if Nb_list is not None:
                raise ValueError("do not use _Np and Nup/nb simultaineously.")
            blocks.pop("_Np")

            if _Np == -1:
                Nb_list = None
            else:
                count_particles = True
                _Np = min((self._sps - 1) * L, _Np)
                Nb_list = list(range(_Np))

        if Nb_list is None:
            self._Np = None
        else:
            self._Np = sum(Nb_list)

        if self._sps is None:
            self._sps = max(Nb_list) + 1

        self._blocks = blocks

        pblock = blocks.get("pblock")
        zblock = blocks.get("cblock")
        zAblock = blocks.get("cAblock")
        zBblock = blocks.get("cBblock")
        kblock = blocks.get("kblock")
        pzblock = blocks.get("pcblock")
        a = blocks.get("a")

        if self._sps > 2 and any(
            type(block) is int for block in [zblock, zAblock, zBblock]
        ):
            raise ValueError("particle hole symmetry doesn't exist with sps > 2.")

        if type(zblock) is int:
            del blocks["cblock"]
            blocks["zblock"] = zblock

        if type(zAblock) is int:
            del blocks["cAblock"]
            blocks["zAblock"] = zAblock

        if type(zBblock) is int:
            del blocks["cBblock"]
            blocks["zBblock"] = zBblock

        if (type(pblock) is int) and (type(zblock) is int):
            blocks["pzblock"] = pblock * zblock
            self._blocks["pcblock"] = pblock * zblock

        if (type(zAblock) is int) and (type(zBblock) is int):
            blocks["zblock"] = zAblock * zBblock
            self._blocks["cblock"] = zAblock * zBblock

        if check_z_symm:

            # checking if spin inversion is compatible with Np and L
            if (Nb_list is not None) and (
                (type(zblock) is int) or (type(pzblock) is int)
            ):
                if len(Nb_list) > 1:
                    ValueError(
                        "spin inversion/particle-hole symmetry only reduces the 0 magnetization or half filled particle sector"
                    )

                Nb = Nb_list[0]

                if (L * (self.sps - 1) % 2) != 0:
                    raise ValueError(
                        "spin inversion/particle-hole symmetry with particle/magnetization conservation must be used with chains with 0 magnetization sector or at half filling"
                    )
                if Nb != L * (self.sps - 1) // 2:
                    raise ValueError(
                        "spin inversion/particle-hole symmetry only reduces the 0 magnetization or half filled particle sector"
                    )

            if (Nb_list is not None) and (
                (type(zAblock) is int) or (type(zBblock) is int)
            ):
                raise ValueError(
                    "zA/cA and zB/cB symmetries incompatible with magnetisation/particle symmetry"
                )

            # checking if ZA/ZB spin inversion is compatible with unit cell of translation symemtry
            if (type(kblock) is int) and (
                (type(zAblock) is int) or (type(zBblock) is int)
            ):
                if a % 2 != 0:  # T and ZA (ZB) symemtries do NOT commute
                    raise ValueError("unit cell size 'a' must be even")

        self._allowed_ops = set(["I", "+", "-", "n", "z"])

        if self._sps <= 2:
            Imax = (1 << L) - 1
            stag_A = sum(1 << i for i in range(0, L, 2))
            stag_B = sum(1 << i for i in range(1, L, 2))
            pars = [0, L, Imax, stag_A, stag_B]  # set sign to not be calculated
            self._operators = (
                "availible operators for boson_basis_1d:"
                + "\n\tI: identity "
                + "\n\t+: raising operator"
                + "\n\t-: lowering operator"
                + "\n\tn: number operator"
                + "\n\tz: c-symm number operator"
            )

            basis_1d.__init__(
                self,
                hcp_basis,
                hcp_ops,
                L,
                Np=Nb_list,
                pars=pars,
                count_particles=count_particles,
                **blocks,
            )
        else:
            pars = (
                (L,) + tuple(self._sps**i for i in range(L + 1)) + (0,)
            )  # flag to turn off higher spin matrix elements for +/- operators

            self._operators = (
                "availible operators for ferion_basis_1d:"
                + "\n\tI: identity "
                + "\n\t+: raising operator"
                + "\n\t-: lowering operator"
                + "\n\tn: number operator"
                + "\n\tz: ph-symm number operator"
            )

            basis_1d.__init__(
                self,
                boson_basis,
                boson_ops,
                L,
                Np=Nb_list,
                pars=pars,
                count_particles=count_particles,
                **blocks,
            )

    def __type__(self):
        return "<type 'qspin.basis.boson_basis_1d'>"

    def __repr__(self):
        return "< instance of 'qspin.basis.boson_basis_1d' with {0} states >".format(
            self._Ns
        )

    def __name__(self):
        return "<type 'qspin.basis.boson_basis_1d'>"

    # functions called in base class:

    def _sort_opstr(self, op):
        if op[0].count("|") > 0:
            raise ValueError("'|' character found in op: {0},{1}".format(op[0], op[1]))
        if len(op[0]) != len(op[1]):
            raise ValueError(
                "number of operators in opstr: {0} not equal to length of indx {1}".format(
                    op[0], op[1]
                )
            )

        op = list(op)
        zipstr = list(zip(op[0], op[1]))
        if zipstr:
            zipstr.sort(key=lambda x: x[1])
            op1, op2 = zip(*zipstr)
            op[0] = "".join(op1)
            op[1] = tuple(op2)
        return tuple(op)

    def _non_zero(self, op):
        opstr = _np.array(list(op[0]))
        indx = _np.array(op[1])
        if _np.any(indx):
            indx_p = indx[opstr == "+"].tolist()
            p = not any(indx_p.count(x) > self.sps - 1 for x in indx_p)
            indx_p = indx[opstr == "-"].tolist()
            m = not any(indx_p.count(x) > self.sps - 1 for x in indx_p)
            return p and m
        else:
            return True

    def _hc_opstr(self, op):
        op = list(op)
        # take h.c. + <--> - , reverse operator order , and conjugate coupling
        op[0] = list(op[0].replace("+", "%").replace("-", "+").replace("%", "-"))
        op[0].reverse()
        op[0] = "".join(op[0])
        op[1] = list(op[1])
        op[1].reverse()
        op[1] = tuple(op[1])
        op[2] = op[2].conjugate()
        return self._sort_opstr(op)  # return the sorted op.

    def _expand_opstr(self, op, num):
        op = list(op)
        op.append(num)
        return [tuple(op)]
