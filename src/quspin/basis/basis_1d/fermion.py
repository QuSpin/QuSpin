from quspin_extensions.basis.basis_1d._basis_1d_core import (
    hcp_basis,
    hcp_ops,
    spf_basis,
    spf_ops,
)
from quspin.basis.basis_1d import _check_1d_symm_spf as _check
from quspin.basis.basis_1d.base_1d import basis_1d
from quspin.basis.base import MAXPRINT
import numpy as _np


class spinless_fermion_basis_1d(basis_1d):
    """Constructs basis for spinless fermionic operators in a specified 1-d symmetry sector.

    The supported operator strings for `spinless_fermion_basis_1d` are:

    .. math::
                    \\begin{array}{cccc}
                            \\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}    \\newline
                            \\texttt{spinless_fermion_basis_1d}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}      \\newline
                    \\end{array}

    Notes
    -----

    Particle-hole like symmetries for fermions can be programmed using the `spinful_fermion_basis_general` class.

    Examples
    --------

    The code snippet below shows how to use the `spinless_fermion_basis_1d` class to construct the basis in the zero momentum sector of positive parity for the fermion Hamiltonian

    .. math::
            H(t)=-J\\sum_j c_jc^\\dagger_{j+1} + \\mathrm{h.c.} - \\mu\\sum_j n_j + U\\cos\\Omega t\\sum_j n_j n_{j+1}

    .. literalinclude:: ../../doc_examples/spinless_fermion_basis_1d-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(self, L, Nf=None, nf=None, **blocks):
        """Intializes the `fermion_basis_1d` object (basis for fermionic operators).

        Parameters
        ----------
        L: int
                Length of chain/number of sites.
        Nf: {int,list}, optional
                Number of fermions in chain. Can be integer or list to specify one or more particle sectors.
        nf: float, optional
                Density of fermions in chain (fermions per site).
        **blocks: optional
                extra keyword arguments which include:

                        **a** (*int*) - specifies unit cell size for translation.

                        **kblock** (*int*) - specifies momentum block. The physical manifestation of this symmetry transformation is translation by `a` lattice sites.

                        **pblock** (*int*) - specifies parity block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain.

        """

        input_keys = set(blocks.keys())

        # Why can we NOT have a check_z_symm toggler just for one of those weirdass cases that worked?

        expected_keys = set(["_Np", "kblock", "pblock", "a", "L"])
        wrong_keys = input_keys - expected_keys
        if wrong_keys:
            temp = ", ".join(["{}" for key in wrong_keys])
            raise ValueError(
                ("unexpected optional argument(s): " + temp).format(*wrong_keys)
            )

        if blocks.get("a") is None:  # by default a = 1
            blocks["a"] = 1

        if Nf is not None and nf is not None:
            raise ValueError("Cannot Nf and nf simultaineously.")
        elif Nf is None and nf is not None:
            if nf < 0 or nf > 1:
                raise ValueError("nf must be between 0 and 1")
            Nf = int(nf * L)

        if Nf is None:
            Nf_list = None
        elif type(Nf) is int:
            Nf_list = [Nf]
        else:
            try:
                Nf_list = list(Nf)
            except TypeError:
                raise TypeError("Nf must be iterable returning integers")

            if any((type(Nf) is not int) for Nf in Nf_list):
                TypeError("Nf must be iterable returning integers")

        count_particles = False
        if blocks.get("_Np") is not None:
            _Np = blocks.get("_Np")
            if Nf_list is not None:
                raise ValueError("do not use _Np and Nup/nb simultaineously.")
            blocks.pop("_Np")

            if _Np == -1:
                Nf_list = None
            else:
                count_particles = True
                _Np = min(L, _Np)
                Nf_list = list(range(_Np))

        if Nf_list is None:
            self._Np = None
        else:
            self._Np = sum(Nf_list)

        self._blocks = blocks
        self._sps = 2
        Imax = (1 << L) - 1
        stag_A = sum(1 << i for i in range(0, L, 2))
        stag_B = sum(1 << i for i in range(1, L, 2))
        pars = [1, L, Imax, stag_A, stag_B]  # sign to be calculated
        self._operators = (
            "availible operators for fermion_basis_1d:"
            + "\n\tI: identity "
            + "\n\t+: raising operator"
            + "\n\t-: lowering operator"
            + "\n\tn: number operator"
            + "\n\tz: c-symm number operator"
        )

        self._allowed_ops = set(["I", "+", "-", "n", "z"])
        basis_1d.__init__(
            self,
            hcp_basis,
            hcp_ops,
            L,
            Np=Nf_list,
            pars=pars,
            count_particles=count_particles,
            **blocks,
        )
        self._noncommuting_bits = [(_np.arange(self.N), _np.array(-1, dtype=_np.int8))]

    def __type__(self):
        return "<type 'qspin.basis.fermion_basis_1d'>"

    def __repr__(self):
        return "< instance of 'qspin.basis.fermion_basis_1d' with {0} states >".format(
            self._Ns
        )

    def __name__(self):
        return "<type 'qspin.basis.fermion_basis_1d'>"

    # functions called in base class:

    def _sort_opstr(self, op):
        return _sort_opstr_spinless(op)

    def _hc_opstr(self, op):
        return _hc_opstr_spinless(op)

    def _non_zero(self, op):
        return _non_zero_spinless(op)

    def _expand_opstr(self, op, num):
        return _expand_opstr_spinless(op, num)


class spinful_fermion_basis_1d(spinless_fermion_basis_1d, basis_1d):
    """Constructs basis for spinful fermionic operators in a specified 1-d symmetry sector.

    The supported operator strings for `spinful_fermion_basis_1d` are:

    .. math::
                    \\begin{array}{cccc}
                            \\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}    \\newline
                            \\texttt{spinful_fermion_basis_1d}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}      \\newline
                    \\end{array}


    Notes
    -----

    * The `spinful_fermion_basis` operator strings are separated by a pipe symbol, "|", to distinguish the spin-up from spin-down species. However, the index array has NO pipe symbol.

    * Particle-hole like symmetries for fermions can be programmed using the `spinful_fermion_basis_general` class.


    Examples
    --------

    The code snippet below shows how to use the `spinful_fermion_basis_1d` class to construct the basis in the zero momentum sector of positive fermion spin for the Fermi-Hubbard Hamiltonian

    .. math::
            H=-J\\sum_{j,\\sigma} c^\\dagger_{j+1\\sigma}c_{j\\sigma} + \\mathrm{h.c.} - \\mu\\sum_{j,\\sigma} n_{j\\sigma} + U \\sum_j n_{j\\uparrow} n_{j\\downarrow}

    Notice that the operator strings for constructing Hamiltonians with a `spinful_fermion_basis` object are separated by
    a pipe symbol, '|', while the index array has no splitting pipe character.


    .. literalinclude:: ../../doc_examples/spinful_fermion_basis_1d-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(self, L, Nf=None, nf=None, double_occupancy=True, **blocks):
        """Intializes the `fermion_basis_1d` object (basis for fermionic operators).

        Parameters
        ----------
        L: int
                Length of chain/number of sites.
        Nf: tuple(int,list), optional
                Number of fermions in chain. First (left) entry refers to spin-up and second (right) entry refers
                to spin-down. Each of the two entries can be integer or list to specify one or more particle sectors.
        nf: tuple(float), optional
                Density of fermions in chain (fermions per site). First (left) entry refers to spin-up. Second (right)
                entry refers to spin-down.
        double_occupancy: bool, optional
                Boolean to toggle the presence of doubly-occupied sites (both a spin up and a spin-down fermion present on the same lattice site) in the basis. Default is `double_occupancy=True`, for which doubly-occupied states are present.
        **blocks: optional
                extra keyword arguments which include:

                        **a** (*int*) - specifies unit cell size for translation.

                        **kblock** (*int*) - specifies momentum block. The physical manifestation of this symmetry transformation is translation by `a` lattice sites.

                        **pblock** (*int*) - specifies parity block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain.

                        **sblock** (*int*) - specifies fermion spin inversion block. The physical manifestation of this symmetry transformation is the exchange of a spin-up and a spin-down fermion on a fixed lattice site.

                        **psblock** (*int*) - specifies parity followed by fermion spin inversion symmetry block. The physical manifestation of this symmetry transformation is reflection about the middle of the chain, and a simultaneous exchange of a spin-up and a spin-down fermion on a fixed lattice site.

        """

        input_keys = set(blocks.keys())

        expected_keys = set(
            ["_Np", "kblock", "pblock", "sblock", "psblock", "a", "check_z_symm", "L"]
        )
        wrong_keys = input_keys - expected_keys
        if wrong_keys:
            temp = ", ".join(["{}" for key in wrong_keys])
            raise ValueError(
                ("unexpected optional argument(s): " + temp).format(*wrong_keys)
            )

        if blocks.get("a") is None:  # by default a = 1
            blocks["a"] = 1

        if Nf is not None and nf is not None:
            raise ValueError("cannot use 'nf' and 'Nf' simultaineously.")
        if Nf is None and nf is not None:
            Nf = [(int(nf[0] * L), int(nf[1] * L))]

        self._sps = 2

        count_particles = False
        _Np = blocks.get("_Np")
        if _Np is not None and Nf is None:
            count_particles = True
            if type(_Np) is not int:
                raise ValueError("_Np must be integer")
            if _Np >= -1:
                if _Np + 1 > L:
                    Nf = []
                    for n in range(L + 1):
                        Nf.extend((n - i, i) for i in range(n + 1))

                    Nf = tuple(Nf)
                elif _Np == -1:
                    Nf = None
                else:
                    Nf = []
                    for n in range(_Np + 1):
                        Nf.extend((n - i, i) for i in range(n + 1))

                    Nf = tuple(Nf)
            else:
                raise ValueError(
                    "_Np == -1 for no particle conservation, _Np >= 0 for particle conservation"
                )

        if Nf is None:
            Nf_list = None
            self._Np = None
        else:
            if type(Nf) is tuple:
                if len(Nf) == 2:
                    Nup, Ndown = Nf
                    if (type(Nup) is not int) and (type(Ndown) is not int):
                        raise ValueError(
                            "Nf must be tuple of integers or iteratable object of tuples."
                        )

                    if Nup > L:
                        raise ValueError("Nup cannot exceed system size L.")
                    if Ndown > L:
                        raise ValueError("Ndown cannot exceed system size L.")
                    self._Np = Nup + Ndown
                    Nf_list = [Nf]
                else:
                    Nf_list = list(Nf)
                    N_up_list, N_down_list = zip(*Nf_list)
                    self._Np = sum(N_up_list)
                    self._Np += sum(N_down_list)
                    if any(
                        (type(tup) is not tuple) and len(tup) != 2 for tup in Nf_list
                    ):
                        raise ValueError(
                            "Nf must be tuple of integers or iteratable object of tuples."
                        )

                    if any(
                        (type(Nup) is not int) and (type(Ndown) is not int)
                        for Nup, Ndown in Nf_list
                    ):
                        raise ValueError(
                            "Nf must be tuple of integers or iteratable object of tuples."
                        )

                    if any(
                        Nup > L or Nup < 0 or Ndown > L or Ndown < 0
                        for Nup, Ndown in Nf_list
                    ):
                        raise ValueError(
                            "particle numbers in Nf must satisfy: 0 <= n <= L"
                        )

            else:
                try:
                    Nf_iter = iter(Nf)
                except TypeError:
                    raise ValueError(
                        "Nf must be tuple of integers or iterable object of tuples."
                    )

                Nf_list = list(Nf)
                N_up_list, N_down_list = zip(*Nf_list)
                self._Np = sum(N_up_list)
                self._Np += sum(N_down_list)

                if any((type(tup) is not tuple) and len(tup) != 2 for tup in Nf_list):
                    raise ValueError(
                        "Nf must be tuple of integers or iteratable object of tuples."
                    )

                if any(
                    (type(Nup) is not int) and (type(Ndown) is not int)
                    for Nup, Ndown in Nf_list
                ):
                    raise ValueError(
                        "Nf must be tuple of integers or iteratable object of tuples."
                    )

                if any(
                    Nup > L or Nup < 0 or Ndown > L or Ndown < 0
                    for Nup, Ndown in Nf_list
                ):
                    raise ValueError("particle numbers in Nf must satisfy: 0 <= n <= L")

        if blocks.get("check_z_symm") is None or blocks.get("check_z_symm") is True:
            check_z_symm = True
        else:
            check_z_symm = False

        self._blocks = blocks
        pblock = blocks.get("pblock")
        zblock = blocks.get("sblock")
        kblock = blocks.get("kblock")
        pzblock = blocks.get("psblock")

        if zblock is not None:
            blocks.pop("sblock")
            blocks["zblock"] = zblock

        if pzblock is not None:
            blocks.pop("psblock")
            blocks["pzblock"] = pzblock

        if (type(pblock) is int) and (type(zblock) is int):
            blocks["pzblock"] = pblock * zblock
            self._blocks["psblock"] = pblock * zblock
            pzblock = pblock * zblock

        if check_z_symm:
            # checking if fermion spin inversion is compatible with Np and L
            if (Nf_list is not None) and (
                (type(zblock) is int) or (type(pzblock) is int)
            ):
                if len(Nf_list) > 1:
                    ValueError(
                        "fermion spin inversion symmetry only reduces the half-filled particle sector"
                    )

                Nup, Ndown = Nf_list[0]

                if (L * (self.sps - 1) % 2) != 0:
                    raise ValueError(
                        "fermion spin inversion symmetry with particle conservation must be used with chains at half filling"
                    )
                if Nup != L * (self.sps - 1) // 2 or Ndown != L * (self.sps - 1) // 2:
                    raise ValueError(
                        "fermion spin inversion symmetry only reduces the half-filled particle sector"
                    )

        double_occupancy = bool(double_occupancy)
        Imax = (1 << L) - 1
        pars = [L, Imax, 0, 0, int(double_occupancy)]  # sign to be calculated
        self._operators = (
            "availible operators for fermion_basis_1d:"
            + "\n\tI: identity "
            + "\n\t+: raising operator"
            + "\n\t-: lowering operator"
            + "\n\tn: number operator"
            + "\n\tz: c-symm number operator"
        )

        self._allowed_ops = set(["I", "+", "-", "n", "z"])
        basis_1d.__init__(
            self,
            spf_basis,
            spf_ops,
            L,
            Np=Nf_list,
            pars=pars,
            count_particles=count_particles,
            **blocks,
        )
        self._noncommuting_bits = [(_np.arange(self.N), _np.array(-1, dtype=_np.int8))]

    @property
    def N(self):
        """int: Total number of sites (spin-up + spin-down) the basis is constructed with; `N=2L`."""
        return 2 * self._L

    def _Op(self, opstr, indx, J, dtype):

        i = opstr.index("|")
        indx = _np.array(indx, dtype=_np.int32)
        indx[i:] += self.L
        opstr = opstr.replace("|", "")

        return basis_1d._Op(self, opstr, indx, J, dtype)

    def index(self, up_state, down_state):
        """Finds the index of user-defined Fock state in spinful fermion basis.

        Notes
        -----
        Particularly useful for defining initial Fock states through a unit vector in the direction specified
        by `index()`.

        Parameters
        ----------
        up_state : str
                string which define the Fock state for the spin up fermions.

        down_state : str
                string which define the Fock state for the spin down fermions.

        Returns
        -------
        int
                Position of the Fock state in the `spinful_fermion_basis_1d`.

        Examples
        --------

        >>> s_up = "".join("1" for i in range(2)) + "".join("0" for i in range(2))
        >>> s_down = "".join("0" for i in range(2)) + "".join("1" for i in range(2))
        >>> print( basis.index(s_up,s_down) )

        """
        if type(up_state) is int:
            pass
        elif type(up_state) is str:
            up_state = int(up_state, 2)
        else:
            raise ValueError("up_state must be integer or string.")

        if type(down_state) is int:
            pass
        elif type(down_state) is str:
            down_state = int(down_state, 2)
        else:
            raise ValueError("down_state must be integer or string.")

        s = down_state + (up_state << self.L)

        indx = _np.argwhere(self._basis == s)

        if len(indx) != 0:
            return _np.squeeze(indx)
        else:
            raise ValueError("state must be representive state in basis.")

    def int_to_state(self, state, bracket_notation=True):

        if int(state) != state:
            raise ValueError("state must be integer")

        n_space = len(str(self.sps))

        if self.N <= 64:
            bits_up = ((state >> (self.N - i - 1)) & 1 for i in range(self.N // 2))
            s_str_up = " ".join(("{:1d}").format(bit) for bit in bits_up)

            bits_down = (
                (state >> (self.N // 2 - i - 1)) & 1 for i in range(self.N // 2)
            )
            s_str_down = " ".join(("{:1d}").format(bit) for bit in bits_down)

        else:
            left_bits_up = (
                state // int(self.sps ** (self.N - i - 1)) % self.sps for i in range(16)
            )
            right_bits_up = (
                state // int(self.sps ** (self.N - i - 1)) % self.sps
                for i in range(self.N // 2 - 16, self.N // 2, 1)
            )

            str_list_up = [
                ("{:" + str(n_space) + "d}").format(bit) for bit in left_bits_up
            ]
            str_list_up.append("...")
            str_list_up.extend(
                ("{:" + str(n_space) + "d}").format(bit) for bit in right_bits_up
            )
            s_str_up = " ".join(str_list_up)

            left_bits_down = (
                state // int(self.sps ** (self.N // 2 - i - 1)) % self.sps
                for i in range(16)
            )
            right_bits_down = (
                state // int(self.sps ** (self.N // 2 - i - 1)) % self.sps
                for i in range(self.N // 2 - 16, self.N // 2, 1)
            )

            str_list_down = [
                ("{:" + str(n_space) + "d}").format(bit) for bit in left_bits_down
            ]
            str_list_down.append("...")
            str_list_down.extend(
                ("{:" + str(n_space) + "d}").format(bit) for bit in right_bits_down
            )
            s_str_down = " ".join(str_list_down)

        if bracket_notation:
            return "|" + s_str_up + ">" + "|" + s_str_down + ">"
        else:
            return (s_str_up + s_str_down).replace(" ", "")

    int_to_state.__doc__ = spinless_fermion_basis_1d.int_to_state.__doc__

    def state_to_int(self, state):
        state = state.replace("|", "").replace(">", "").replace("<", "")
        up_state, down_state = state[: self.N // 2], state[self.N // 2 :]
        return int(self._basis[self.index(up_state, down_state)])

    state_to_int.__doc__ = spinless_fermion_basis_1d.state_to_int.__doc__

    def partial_trace(
        self,
        state,
        sub_sys_A=None,
        density=True,
        subsys_ordering=True,
        return_rdm=None,
        enforce_pure=False,
        return_rdm_EVs=False,
        sparse=False,
        alpha=1.0,
        sparse_diag=True,
        maxiter=None,
    ):
        """Calculates reduced density matrix, through a partial trace of a quantum state in a lattice `basis`.

        Parameters
        ----------
        state : obj
                State of the quantum system. Can be either one of:

                        * numpy.ndarray [shape (Ns,)]: pure state (default).
                        * numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
                        * dict('V_states',V_states) [shape (Ns,Nvecs)]: collection of `Nvecs` states stored in the columns of `V_states`.
        sub_sys_A : tuple/list, optional
                Defines the sites contained in subsystem A [by python convention the first site of the chain is labelled j=0].
                Default is `tuple(range(N//2),range(N//2))` with `N` the number of physical lattice sites (e.g. sites which both species of fermions can occupy).
                The format is `(spin_up_subsys,spin_down_subsys)` (see example below).
        return_rdm : str, optional
                Toggles returning the reduced DM. Can be tierh one of:

                        * "A": returns reduced DM of subsystem A.
                        * "B": returns reduced DM of subsystem B.
                        * "both": returns reduced DM of both A and B subsystems.
        subsys_ordering : bool, optional
                Whether or not to reorder the sites in `sub_sys_A` in ascending order. Default is `True`.
        enforce_pure : bool, optional
                Whether or not to assume `state` is a colelction of pure states or a mixed density matrix, if
                it is a square array. Default is `False`.
        sparse : bool, optional
                Whether or not to return a sparse DM. Default is `False`.

        Returns
        -------
        numpy.ndarray
                Density matrix associated with `state`. Depends on optional arguments.

        Examples
        --------

        >>> sub_sys_A_up=range(basis.L//2) # subsystem for spin-up fermions
        >>> sub_sys_A_down=range(basis.L//2+1) # subsystem for spin-down fermions
        >>> subsys_A=(sub_sys_A_up,sub_sys_A_down)
        >>> state=1.0/np.sqrt(basis.Ns)*np.ones(basis.Ns) # infinite temperature state
        >>> partial_trace(state,sub_sys_A=subsys_A,return_rdm="A",enforce_pure=False,sparse=False,subsys_ordering=True)

        """
        if sub_sys_A is None:
            sub_sys_A = (list(range(self.L // 2)), list(range(self.L // 2)))

        if type(sub_sys_A) is tuple and len(sub_sys_A) != 2:
            raise ValueError(
                "sub_sys_A must be a tuple which contains the subsystems for the spin-up fermions in the \
							  first (left) part of the tuple and the spin-down fermions in the last (right) part of the tuple."
            )

        sub_sys_A_up, sub_sys_A_down = sub_sys_A

        sub_sys_A = list(sub_sys_A_up)
        sub_sys_A.extend([i + self.L for i in sub_sys_A_down])

        return basis_1d._partial_trace(
            self,
            state,
            sub_sys_A=sub_sys_A,
            subsys_ordering=subsys_ordering,
            return_rdm=return_rdm,
            enforce_pure=enforce_pure,
            sparse=sparse,
        )

    def ent_entropy(
        self,
        state,
        sub_sys_A=None,
        density=True,
        subsys_ordering=True,
        return_rdm=None,
        enforce_pure=False,
        return_rdm_EVs=False,
        sparse=False,
        alpha=1.0,
        sparse_diag=True,
        maxiter=None,
        svd_solver=None,
        svd_kwargs=None,
    ):
        """Calculates entanglement entropy of subsystem A and the corresponding reduced density matrix

        Notes
        -----
        Algorithm is based on both partial tracing and sigular value decomposition (SVD), optimised for speed.

        Parameters
        ----------
        state : obj
                State of the quantum system. Can be either one of:

                        * numpy.ndarray [shape (Ns,)]: pure state (default).
                        * numpy.ndarray [shape (Ns,Ns)]: density matrix (DM).
                        * dict('V_states',V_states) [shape (Ns,Nvecs)]: collection of `Nvecs` states stored in the columns of `V_states`.
        sub_sys_A : tuple, optional
                Defines the sites contained in subsystem A [by python convention the first site of the chain is labelled j=0].
                Default is `tuple(range(N//2),range(N//2))` with `N` the number of physical lattice sites (e.g. sites which both species of fermions can occupy).
                The format is `(spin_up_subsys,spin_down_subsys)` (see example below).
        return_rdm : str, optional
                Toggles returning the reduced DM. Can be tierh one of:

                        * "A": returns reduced DM of subsystem A.
                        * "B": returns reduced DM of subsystem B.
                        * "both": returns reduced DM of both A and B subsystems.
        enforce_pure : bool, optional
                Whether or not to assume `state` is a colelction of pure states or a mixed density matrix, if
                it is a square array. Default is `False`.
        subsys_ordering : bool, optional
                Whether or not to reorder the sites in `sub_sys_A` in ascending order. Default is `True`.
        sparse : bool, optional
                Whether or not to return a sparse DM. Default is `False`.
        return_rdm_EVs : bool, optional
                Whether or not to return the eigenvalues of rthe educed DM. If `return_rdm` is specified,
                the eigenvalues of the corresponding DM are returned. If `return_rdm` is NOT specified,
                the spectrum of `rdm_A` is returned by default. Default is `False`.
        alpha : float, optional
                Renyi :math:`\\alpha` parameter for the entanglement entropy. Default is :math:`\\alpha=1`:

                .. math::
                        S_\\mathrm{ent}(\\alpha) =  \\frac{1}{1-\\alpha}\\log \\mathrm{tr}_{A} \\left( \\mathrm{tr}_{A^c} \\vert\\psi\\rangle\\langle\\psi\\vert \\right)^\\alpha

                **Note:** The logarithm used is the natural logarithm (base e).
        sparse_diag : bool, optional
                When `sparse=True`, this flag enforces the use of
                `scipy.sparse.linalg.eigsh() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_
                to calculate the eigenvaues of the reduced DM.
        maxiter : int, optional
                Specifies the number of iterations for Lanczos diagonalisation. Look up documentation for
                `scipy.sparse.linalg.eigsh() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_.
        svd_solver : object, optional
                Specifies the svd solver to be used, e.g. `numpy.linalg.svd` or `scipy.linalg.svd`, or a custom solver. Effective when `enforce_pure=True` or `sparse=False`.
        svd_kwargs : dict, optional
                Specifies additional arguments for `svd_solver`.


        Returns
        -------
        dict
                Dictionary with following keys, depending on input parameters:
                        * "Sent_A": entanglement entropy of subsystem A (default).
                        * "Sent_B": entanglement entropy of subsystem B.
                        * "p_A": singular values of reduced DM of subsystem A (default).
                        * "p_B": singular values of reduced DM of subsystem B.
                        * "rdm_A": reduced DM of subsystem A.
                        * "rdm_B": reduced DM of subsystem B.

        Examples
        --------

        >>> sub_sys_A_up=range(basis.L//2) # subsystem for spin-up fermions
        >>> sub_sys_A_down=range(basis.L//2+1) # subsystem for spin-down fermions
        >>> subsys_A=(sub_sys_A_up,sub_sys_A_down)
        >>> state=1.0/np.sqrt(basis.Ns)*np.ones(basis.Ns) # infinite temperature state
        >>> ent_entropy(state,sub_sys_A=subsys_A,return_rdm="A",enforce_pure=False,return_rdm_EVs=False,
        >>>				sparse=False,alpha=1.0,sparse_diag=True,subsys_ordering=True)

        """
        if sub_sys_A is None:
            sub_sys_A = (list(range(self.L // 2)), list(range(self.L // 2)))

        if type(sub_sys_A) is tuple and len(sub_sys_A) != 2:
            raise ValueError(
                "sub_sys_A must be a tuple which contains the subsystems for the up spins in the \
							  first (left) part of the tuple and the down spins in the last (right) part of the tuple."
            )

        sub_sys_A_up, sub_sys_A_down = sub_sys_A

        sub_sys_A = list(sub_sys_A_up)
        sub_sys_A.extend([i + self.L for i in sub_sys_A_down])

        return basis_1d._ent_entropy(
            self,
            state,
            sub_sys_A,
            density=density,
            subsys_ordering=subsys_ordering,
            return_rdm=return_rdm,
            enforce_pure=enforce_pure,
            return_rdm_EVs=return_rdm_EVs,
            sparse=sparse,
            alpha=alpha,
            sparse_diag=sparse_diag,
            maxiter=maxiter,
            svd_solver=svd_solver,
            svd_kwargs=svd_kwargs,
        )

    def __type__(self):
        return "<type 'qspin.basis.fermion_basis_1d'>"

    def __repr__(self):
        return "< instance of 'qspin.basis.fermion_basis_1d' with {0} states >".format(
            self._Ns
        )

    def __name__(self):
        return "<type 'qspin.basis.fermion_basis_1d'>"

    # functions called in base class:

    def _sort_opstr(self, op):
        return _sort_opstr_spinful(op)

    def _hc_opstr(self, op):
        return _hc_opstr_spinful(op)

    def _non_zero(self, op):
        return _non_zero_spinful(op)

    def _expand_opstr(self, op, num):
        return _expand_opstr_spinful(op, num)

    """
	def _get_state(self,b):
		b = int(b)
		bits_left = ((b>>(self.N-i-1))&1 for i in range(self.N//2))
		state_left = "|"+(" ".join(("{:1d}").format(bit) for bit in bits_left))+">"
		bits_right = ((b>>(self.N//2-i-1))&1 for i in range(self.N//2))
		state_right = "|"+(" ".join(("{:1d}").format(bit) for bit in bits_right))+">"
		return state_left+state_right

	def _get__str__(self):
		temp1 = "     {0:"+str(len(str(self.Ns)))+"d}.  "
		if self._Ns > MAXPRINT:
			half = MAXPRINT // 2
			str_list = [(temp1.format(i))+self._get_state(b) for i,b in zip(range(half),self._basis[:half])]
			str_list.extend([(temp1.format(i))+self._get_state(b) for i,b in zip(range(self._Ns-half,self._Ns,1),self._basis[-half:])])
		else:
			str_list = [(temp1.format(i))+self._get_state(b) for i,b in enumerate(self._basis)]

		return tuple(str_list)
	"""

    def _check_symm(self, static, dynamic, photon_basis=None):
        kblock = self._blocks_1d.get("kblock")
        pblock = self._blocks_1d.get("pblock")
        zblock = self._blocks_1d.get("zblock")
        pzblock = self._blocks_1d.get("pzblock")
        a = self._blocks_1d.get("a")
        L = self.L

        if photon_basis is None:
            photon = False
            basis_sort_opstr = self._sort_opstr
            static_list, dynamic_list = self._get_local_lists(static, dynamic)
        else:
            photon = True
            basis_sort_opstr = photon_basis._sort_opstr
            static_list, dynamic_list = photon_basis._get_local_lists(static, dynamic)

        static_blocks = {}
        dynamic_blocks = {}
        if kblock is not None:
            missingops = _check.check_T(basis_sort_opstr, static_list, L, a)
            if missingops:
                static_blocks["T symm"] = (tuple(missingops),)

            missingops = _check.check_T(basis_sort_opstr, dynamic_list, L, a)
            if missingops:
                dynamic_blocks["T symm"] = (tuple(missingops),)

        if pblock is not None:
            missingops = _check.check_P(basis_sort_opstr, static_list, L)
            if missingops:
                static_blocks["P symm"] = (tuple(missingops),)

            missingops = _check.check_P(basis_sort_opstr, dynamic_list, L)
            if missingops:
                dynamic_blocks["P symm"] = (tuple(missingops),)

        if zblock is not None:
            missingops = []

            oddops = _check.check_Z(basis_sort_opstr, static_list, photon)
            if missingops or oddops:
                static_blocks["Z/C/S symm"] = (tuple(oddops), tuple(missingops))

            oddops = _check.check_Z(basis_sort_opstr, dynamic_list, photon)
            if missingops or oddops:
                dynamic_blocks["Z/C/S symm"] = (tuple(oddops), tuple(missingops))

        if pzblock is not None:
            missingops = _check.check_PZ(basis_sort_opstr, static_list, L, photon)
            if missingops:
                static_blocks["PZ/PC/PS symm"] = (tuple(missingops),)

            missingops = _check.check_PZ(basis_sort_opstr, dynamic_list, L, photon)
            if missingops:
                dynamic_blocks["PZ/PC/PS symm"] = (tuple(missingops),)

        return static_blocks, dynamic_blocks


def _sort_opstr_spinless(op):
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
        n = len(zipstr)
        swapped = True
        anticommutes = 0
        while swapped:
            swapped = False
            for i in range(n - 1):
                if zipstr[i][1] > zipstr[i + 1][1]:
                    temp = zipstr[i]
                    zipstr[i] = zipstr[i + 1]
                    zipstr[i + 1] = temp
                    swapped = True

                    if zipstr[i][0] in ["+", "-"] and zipstr[i + 1][0] in ["+", "-"]:
                        anticommutes += 1

        op1, op2 = zip(*zipstr)
        op[0] = "".join(op1)
        op[1] = tuple(op2)
        op[2] *= 1 if anticommutes % 2 == 0 else -1
    return tuple(op)


def _sort_opstr_spinful(op):
    op = list(op)
    opstr = op[0]
    indx = op[1]

    if opstr.count("|") == 0:
        raise ValueError("missing '|' charactor in: {0}, {1}".format(opstr, indx))

    # if opstr.count("|") > 1:
    # 	raise ValueError("only one '|' charactor allowed in: {0}, {1}".format(opstr,indx))

    if len(opstr) - opstr.count("|") != len(indx):
        raise ValueError(
            "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
        )

    i = opstr.index("|")
    indx_left = indx[:i]
    indx_right = indx[i:]

    opstr_left, opstr_right = opstr.split("|", 1)

    op1 = list(op)
    op1[0] = opstr_left
    op1[1] = tuple(indx_left)
    op1[2] = 1

    op2 = list(op)
    op2[0] = opstr_right
    op2[1] = tuple(indx_right)
    op2[2] = 1

    op1 = _sort_opstr_spinless(op1)
    op2 = _sort_opstr_spinless(op2)

    op[0] = "|".join((op1[0], op2[0]))
    op[1] = op1[1] + op2[1]
    op[2] *= op1[2] * op2[2]

    return tuple(op)


def _non_zero_spinless(op):
    opstr = _np.array(list(op[0]))
    indx = _np.array(op[1])
    if _np.any(indx):
        indx_p = indx[opstr == "+"].tolist()
        p = not any(indx_p.count(x) > 1 for x in indx_p)
        indx_p = indx[opstr == "-"].tolist()
        m = not any(indx_p.count(x) > 1 for x in indx_p)
        return p and m
    else:
        return True


def _non_zero_spinful(op):
    op = list(op)
    opstr = op[0]
    indx = op[1]

    if opstr.count("|") > 1:
        raise ValueError(
            "only one '|' charactor allowed in: {0}, {1}".format(opstr, indx)
        )

    if len(opstr) - 1 != len(indx):
        raise ValueError(
            "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
        )

    i = opstr.index("|")
    indx_left = indx[:i]
    indx_right = indx[i:]

    opstr_left, opstr_right = opstr.split("|", 1)

    op1 = list(op)
    op1[0] = opstr_left
    op1[1] = indx_left

    op2 = list(op)
    op2[0] = opstr_right
    op2[1] = indx_right

    return _non_zero_spinless(op1) and _non_zero_spinless(op2)


def _hc_opstr_spinless(op):
    op = list(op)
    # take h.c. + <--> - , reverse operator order , and conjugate coupling
    op[0] = list(op[0].replace("+", "%").replace("-", "+").replace("%", "-"))
    op[0].reverse()
    op[0] = "".join(op[0])
    op[1] = list(op[1])
    op[1].reverse()
    op[1] = tuple(op[1])
    op[2] = op[2].conjugate()
    return _sort_opstr_spinless(op)  # return the sorted op.


def _hc_opstr_spinful(op):
    op = list(op)
    opstr = op[0]
    indx = op[1]

    if opstr.count("|") > 1:
        raise ValueError(
            "only one '|' charactor allowed in: {0}, {1}".format(opstr, indx)
        )

    if len(opstr) - 1 != len(indx):
        raise ValueError(
            "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
        )

    i = opstr.index("|")
    indx_left = indx[:i]
    indx_right = indx[i:]

    opstr_left, opstr_right = opstr.split("|", 1)
    n_left = opstr_left.count("+") + opstr_left.count("-")
    n_right = opstr_right.count("+") + opstr_right.count("-")

    op1 = list(op)
    op1[0] = opstr_left
    op1[1] = indx_left
    op1[2] = op[2]

    op2 = list(op)
    op2[0] = opstr_right
    op2[1] = indx_right
    op2[2] = complex(1.0)

    op1 = _hc_opstr_spinless(op1)
    op2 = _hc_opstr_spinless(op2)

    op[0] = "|".join((op1[0], op2[0]))
    op[1] = op1[1] + op2[1]

    op[2] = ((-1) ** (n_left * n_right)) * op1[2] * op2[2]

    return tuple(op)


def _expand_opstr_spinless(op, num):
    op = list(op)
    op.append(num)
    return [tuple(op)]


def _expand_opstr_spinful(op, num):
    op = list(op)
    opstr = op[0]
    indx = op[1]

    if opstr.count("|") > 1:
        raise ValueError(
            "only one '|' charactor allowed in: {0}, {1}".format(opstr, indx)
        )

    if len(opstr) - 1 != len(indx):
        raise ValueError(
            "number of indices doesn't match opstr in: {0}, {1}".format(opstr, indx)
        )

    i = opstr.index("|")
    indx_left = indx[:i]
    indx_right = indx[i:]

    opstr_left, opstr_right = opstr.split("|", 1)

    op1 = list(op)
    op1[0] = opstr_left
    op1[1] = indx_left
    op1[2] = 1.0

    op2 = list(op)
    op2[0] = opstr_right
    op2[1] = indx_right

    op1_list = _expand_opstr_spinless(op1, num)
    op2_list = _expand_opstr_spinless(op2, num)

    op_list = []
    for new_op1 in op1_list:
        for new_op2 in op2_list:
            new_op = list(new_op1)
            new_op[0] = "|".join((new_op1[0], new_op2[0]))
            new_op[1] += tuple(new_op2[1])
            new_op[2] *= new_op2[2]

            op_list.append(tuple(new_op))

    return tuple(op_list)
