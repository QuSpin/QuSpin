from quspin.basis.basis_general.base_hcb import hcb_basis_general
from quspin.basis.basis_general.base_higher_spin import higher_spin_basis_general
import numpy as _np

try:
    S_dict = {
        (str(i) + "/2" if i % 2 == 1 else str(i // 2)): (i + 1, i / 2.0)
        for i in xrange(1, 10001)
    }
except NameError:
    S_dict = {
        (str(i) + "/2" if i % 2 == 1 else str(i // 2)): (i + 1, i / 2.0)
        for i in range(1, 10001)
    }


class spin_basis_general(hcb_basis_general, higher_spin_basis_general):
    """Constructs basis for spin operators for USER-DEFINED symmetries.

    Any unitary symmetry transformation :math:`Q` of periodicity :math:`m_Q` (:math:`Q^{m_Q}=1`) has eigenvalues :math:`\\exp(-2\\pi i q/m_Q)`, labelled by an ingeter :math:`q\\in\\{0,1,\\dots,m_Q-1\\}`. These integers :math:`q` are used to define the symmetry blocks.

    For instance, if :math:`Q=P` is parity (reflection), then :math:`q=0,1`. If :math:`Q=T` is translation by one lattice site, then :math:`q` labels the mometum blocks in the same fashion as for the `..._basis_1d` classes.

    User-defined symmetries with the `spin_basis_general` class can be programmed as follows. Suppose we have a system of L sites, enumerated :math:`s=(s_0,s_1,\\dots,s_{L-1})`. There are two types of operations one can perform on the sites:
            * exchange the labels of two sites: :math:`s_i \\leftrightarrow s_j` (e.g., translation, parity)
            * invert the population on a given site: :math:`s_i\\leftrightarrow -(s_j+1)` (e.g., spin inversion)

    These two operations already comprise a variety of symmetries, including translation, parity (reflection) and spin inversion. For a specific example, see below.

    The supported operator strings for `spin_basis_general` are:

    .. math::
            \\begin{array}{cccc}
                    \\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &     \\texttt{"z"}   &   \\texttt{"x"}   &   \\texttt{"y"}  \\newline
                    \\texttt{spin_basis_general} &   \\hat{1}        &   \\hat\\sigma^+       &   \\hat\\sigma^-      &     \\hat\\sigma^z       &   (\\hat\\sigma^x)     &   (\\hat\\sigma^y)  \\  \\newline
            \\end{array}

    **Notes:**
            * The relation between spin and Pauli matrices is :math:`\\vec S = \\vec \\sigma/2`.
            * The default operators for spin-1/2 are the Pauli matrices, NOT the spin operators. To change this, see the argument `pauli` of the `spin_basis` class. Higher spins can only be defined using the spin operators, and do NOT support the operator strings "x" and "y".
            * QuSpin raises a warning to alert the reader when non-commuting symmetries are passed. In such cases, we recommend the user to manually check the combined usage of symmetries by, e.g., comparing the eigenvalues.


    Examples
    --------

    The code snippet below shows how to construct the two-dimensional transverse-field Ising model.

    .. math::
            H = J \\sum_{\\langle ij\\rangle} \\sigma^z_{i}\\sigma^z_j+ g\\sum_j\\sigma^x_j

    Moreover, it demonstrates how to pass user-defined symmetries to the `spin_basis_general` constructor. In particular,
    we do translation invariance and parity (reflection) (along each lattice direction), and spin inversion. Note that parity
    (reflection) and translation invariance are non-commuting symmetries, and QuSpin raises a warning when constructing the basis.
    However, they do commute in the zero-momentum (also in the pi-momentum) symmetry sector; hence, one can ignore the warning and
    use the two symemtries together to reduce the Hilbert space dimension.


    .. literalinclude:: ../../doc_examples/spin_basis_general-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(
        self,
        N,
        Nup=None,
        m=None,
        S="1/2",
        pauli=1,
        Ns_block_est=None,
        make_basis=True,
        block_order=None,
        **blocks,
    ):
        """Intializes the `spin_basis_general` object (basis for spin operators).

        Parameters
        ----------
        N: int
                number of sites.
        Nup: {int,list}, optional
                Total magnetisation, :math:`Nup = NS + \\sum_j S^z_j, Nup\\ge 0`, and :math:`Nup = -NS + \\sum_j S^z_j, Nup<0` projection. Can be integer or list to specify one or
                more particle sectors. Negative values are taken to be subtracted from the fully polarized up state as: Nup_max + Nup + 1.
                e.g. to get the Nup_max state use Nup = -1, for Nup_max-1 state use Nup = -2, etc.
        m: float, optional
                Density of spin up in chain (spin up per site).
        S: str, optional
                Size of local spin degrees of freedom. Can be any (half-)integer from:
                "1/2","1","3/2",...,"9999/2","5000".
        pauli: int, optional (requires `S="1/2"`)
                * for `pauli=0` the code uses spin-1/2 operators:

                .. math::

                        S^x = \\frac{1}{2}\\begin{pmatrix}0 & 1\\\\ 1 & 0\\end{pmatrix},\\quad S^y = \\frac{1}{2}\\begin{pmatrix}0 & -i\\\\ i & 0\\end{pmatrix},\\quad S^z = \\frac{1}{2}\\begin{pmatrix}1 & 0\\\\ 0 & -1\\end{pmatrix},\\quad S^+ = \\begin{pmatrix}0 & 1\\\\ 0 & 0\\end{pmatrix},\\quad S^- = \\begin{pmatrix}0 & 0\\\\ 1 & 0\\end{pmatrix}

                * for `pauli=1` the code uses Pauli matrices with:

                .. math::

                        \\sigma^x = \\begin{pmatrix}0 & 1\\\\ 1 & 0\\end{pmatrix},\\quad \\sigma^y = \\begin{pmatrix}0 & -i\\\\ i & 0\\end{pmatrix},\\quad \\sigma^z = \\begin{pmatrix}1 & 0\\\\ 0 & -1\\end{pmatrix},\\quad \\sigma^+ = \\begin{pmatrix}0 & 2\\\\ 0 & 0\\end{pmatrix},\\quad \\sigma^- = \\begin{pmatrix}0 & 0\\\\ 2 & 0\\end{pmatrix}


                * for `pauli=-1` the code uses Pauli matrices with:

                .. math::

                        \\sigma^x = \\begin{pmatrix}0 & 1\\\\ 1 & 0\\end{pmatrix},\\quad \\sigma^y = \\begin{pmatrix}0 & -i\\\\ i & 0\\end{pmatrix},\\quad \\sigma^z = \\begin{pmatrix}1 & 0\\\\ 0 & -1\\end{pmatrix},\\quad \\sigma^+ = \\begin{pmatrix}0 & 1\\\\ 0 & 0\\end{pmatrix},\\quad \\sigma^- = \\begin{pmatrix}0 & 0\\\\ 1 & 0\\end{pmatrix}

        Ns_block_est: int, optional
                Overwrites the internal estimate of the size of the reduced Hilbert space for the given symmetries. This can be used to help conserve memory if the exact size of the H-space is known ahead of time.
        make_basis: bool, optional
                Boolean to control whether to make the basis. Allows the use to use some functionality of the basis constructor without constructing the entire basis.
        block_order: list of strings, optional
                A list of strings containing the names of the symmetry blocks which specifies the order in which the symmetries will be applied to the state when calculating the basis. The first element in the list is applied to the state first followed by the second element, etc. If the list is not specificed the ordering is such that the symmetry with the largest cycle is the first, followed by the second largest, etc.
        **blocks: optional
                keyword arguments which pass the symmetry generator arrays. For instance:

                >>> basis(...,kxblock=(Q,q),...)

                The keys of the symmetry sector, e.g. `kxblock`, can be chosen arbitrarily by the user. The
                values are tuples where the first entry contains the symmetry transformation :math:`Q` acting on the
                lattice sites (see class example), and the second entry is an integer :math:`q` to label the symmetry
                sector.

        """
        self._S = S
        self._pauli = int(pauli)
        sps, S = S_dict[S]

        if self._pauli not in [-1, 0, 1]:
            raise ValueError(
                "Invalid value for optional argument pauli. Allowed values are the integers [-1,0,1]."
            )

        _Np = blocks.get("_Np")
        if _Np is not None:
            blocks.pop("_Np")

        if Nup is not None and m is not None:
            raise ValueError("Cannot use Nup and m at the same time")
        if m is not None and Nup is None:
            if m < -S or m > S:
                raise ValueError("m must be between -S and S")

            Nup = int((m + S) * N)

        try:
            Nup_iter = iter(Nup)
            M = int(2 * S * N)
            Nup = [(M + (Nup + 1) if Nup < 0 else Nup) for Nup in Nup_iter]
        except TypeError:
            if Nup is not None and Nup < 0:
                Nup = int(2 * S * N) + Nup + 1

        self._pcon_args = dict(N=N, Nup=Nup, S=self._S)
        if _Np is not None:
            self._pcon_args["_Np"] = _Np

        if sps == 2:
            hcb_basis_general.__init__(
                self,
                N,
                Nb=Nup,
                Ns_block_est=Ns_block_est,
                _Np=_Np,
                _make_basis=make_basis,
                block_order=block_order,
                **blocks,
            )
        else:
            higher_spin_basis_general.__init__(
                self,
                N,
                Nup=Nup,
                sps=sps,
                Ns_block_est=Ns_block_est,
                _Np=_Np,
                _make_basis=make_basis,
                block_order=block_order,
                **blocks,
            )

        if self._sps <= 2:
            self._operators = (
                "availible operators for spin_basis_1d:"
                + "\n\tI: identity "
                + "\n\t+: raising operator"
                + "\n\t-: lowering operator"
                + "\n\tx: x pauli/spin operator"
                + "\n\ty: y pauli/spin operator"
                + "\n\tz: z pauli/spin operator"
            )

            self._allowed_ops = set(["I", "+", "-", "x", "y", "z"])
        else:
            self._operators = (
                "availible operators for spin_basis_1d:"
                + "\n\tI: identity "
                + "\n\t+: raising operator"
                + "\n\t-: lowering operator"
                + "\n\tz: z pauli/spin operator"
            )

            self._allowed_ops = set(["I", "+", "-", "z"])

    def __setstate__(self, state):
        if state["_sps"] == 2:
            hcb_basis_general.__setstate__(self, state)
        else:
            higher_spin_basis_general.__setstate__(self, state)

    def _Op(self, opstr, indx, J, dtype):

        if self._S == "1/2":
            ME, row, col = hcb_basis_general._Op(self, opstr, indx, J, dtype)
            if self._pauli == 1:
                n = len(opstr.replace("I", ""))
                ME *= 1 << n
            elif self._pauli == -1:
                n = len(opstr.replace("I", "").replace("+", "").replace("-", ""))
                ME *= 1 << n

            return ME, row, col

        else:
            return higher_spin_basis_general._Op(self, opstr, indx, J, dtype)

    def _inplace_Op(
        self,
        v_in,
        op_list,
        dtype,
        transposed=False,
        conjugated=False,
        v_out=None,
        a=1.0,
    ):
        if self._S == "1/2":

            if self._pauli == 1:
                scale = lambda s: (1 << len(s.replace("I", "")))
            elif self._pauli == -1:
                scale = lambda s: (
                    1 << len(s.replace("I", "").replace("+", "").replace("-", ""))
                )
            else:
                scale = lambda s: 1

            op_list = [[op, indx, J * scale(op)] for op, indx, J in op_list]

            return hcb_basis_general._inplace_Op(
                self,
                v_in,
                op_list,
                dtype,
                transposed=transposed,
                conjugated=conjugated,
                v_out=v_out,
                a=a,
            )

        else:
            return higher_spin_basis_general._inplace_Op(
                self,
                v_in,
                op_list,
                dtype,
                transposed=transposed,
                conjugated=conjugated,
                v_out=v_out,
                a=a,
            )

    def Op_shift_sector(self, other_basis, op_list, v_in, v_out=None, dtype=None):
        if self._S == "1/2":

            if self._pauli == 1:
                scale = lambda s: (1 << len(s.replace("I", "")))
            elif self._pauli == -1:
                scale = lambda s: (
                    1 << len(s.replace("I", "").replace("+", "").replace("-", ""))
                )
            else:
                scale = lambda s: 1

            op_list = [[op, indx, J * scale(op)] for op, indx, J in op_list]

            return hcb_basis_general.Op_shift_sector(
                self, other_basis, op_list, v_in, v_out=v_out, dtype=dtype
            )

        else:
            return higher_spin_basis_general.Op_shift_sector(
                self, other_basis, op_list, v_in, v_out=v_out, dtype=dtype
            )

    Op_shift_sector.__doc__ = hcb_basis_general.Op_shift_sector.__doc__

    def __type__(self):
        return "<type 'qspin.basis.general_hcb'>"

    def __repr__(self):
        return "< instance of 'qspin.basis.general_hcb' with {0} states >".format(
            self._Ns
        )

    def __name__(self):
        return "<type 'qspin.basis.general_hcb'>"

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
        opstr = str(op[0])
        indx = list(op[1])
        J = op[2]

        if len(opstr) <= 1:
            if opstr == "x":
                op1 = list(op)
                op1[0] = op1[0].replace("x", "+")
                if self._pauli in [0, 1]:
                    op1[2] *= 0.5
                op1.append(num)

                op2 = list(op)
                op2[0] = op2[0].replace("x", "-")
                if self._pauli in [0, 1]:
                    op2[2] *= 0.5
                op2.append(num)

                return (tuple(op1), tuple(op2))
            elif opstr == "y":
                op1 = list(op)
                op1[0] = op1[0].replace("y", "+")
                if self._pauli in [0, 1]:
                    op1[2] *= -0.5j
                else:
                    op1[2] *= -1j
                op1.append(num)

                op2 = list(op)
                op2[0] = op2[0].replace("y", "-")
                if self._pauli in [0, 1]:
                    op2[2] *= 0.5j
                else:
                    op2[2] *= 1j
                op2.append(num)

                return (tuple(op1), tuple(op2))
            else:
                op = list(op)
                op.append(num)
                return [tuple(op)]
        else:

            i = len(opstr) // 2
            op1 = list(op)
            op1[0] = opstr[:i]
            op1[1] = tuple(indx[:i])
            op1[2] = complex(J)
            op1 = tuple(op1)

            op2 = list(op)
            op2[0] = opstr[i:]
            op2[1] = tuple(indx[i:])
            op2[2] = complex(1)
            op2 = tuple(op2)

            l1 = self._expand_opstr(op1, num)
            l2 = self._expand_opstr(op2, num)

            l = []
            for op1 in l1:
                for op2 in l2:
                    op = list(op1)
                    op[0] += op2[0]
                    op[1] += op2[1]
                    op[2] *= op2[2]
                    l.append(tuple(op))

            return tuple(l)

    def Op_bra_ket(self, opstr, indx, J, dtype, ket_states, reduce_output=True):

        if self._S == "1/2":
            ME, bra, ket = hcb_basis_general.Op_bra_ket(
                self, opstr, indx, J, dtype, ket_states, reduce_output=reduce_output
            )
            if self._pauli == 1:
                n = len(opstr.replace("I", ""))
                ME *= 1 << n
            elif self._pauli == -1:
                n = len(opstr.replace("I", "").replace("+", "").replace("-", ""))
                ME *= 1 << n
        else:
            return higher_spin_basis_general.Op_bra_ket(
                self, opstr, indx, J, dtype, ket_states, reduce_output=reduce_output
            )

        return ME, bra, ket
