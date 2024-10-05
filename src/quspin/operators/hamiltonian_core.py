from quspin.basis import spin_basis_1d as _default_basis
from quspin.basis import isbasis as _isbasis

from quspin.tools.evolution import evolve

from quspin.tools.matvec import _matvec
from quspin.tools.matvec import _get_matvec_function

# from quspin.operators_oputils import matvec as _matvec
# from quspin.operators._oputils import _get_matvec_function

# from quspin.operators.exp_op_core import isexp_op,exp_op

from quspin.operators._make_hamiltonian import make_static
from quspin.operators._make_hamiltonian import make_dynamic
from quspin.operators._make_hamiltonian import test_function
from quspin.operators._make_hamiltonian import _check_almost_zero
from quspin.operators._functions import function

# need linear algebra packages
import scipy
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

from operator import mul
import functools
from six import iteritems, itervalues, viewkeys

try:
    from itertools import izip as zip
except ImportError:
    pass

try:
    from functools import reduce as reduce
except ImportError:
    pass

import warnings


__all__ = ["commutator", "anti_commutator", "hamiltonian", "ishamiltonian"]


def commutator(H1, H2):
    """Calculates the commutator of two Hamiltonians :math:`H_1` and :math:`H_2`.

    .. math::
            [H_1,H_2] = H_1 H_2 - H_2 H_1

    Examples
    --------
    The following script shows how to compute the commutator of two `hamiltonian` objects.

    .. literalinclude:: ../../doc_examples/commutator-example.py
            :linenos:
            :language: python
            :lines: 7-

    Parameters
    ----------
    H1 : obj
            `numpy.ndarray` or `hamiltonian` class object to define the Hamiltonian operator as a matrix.
    H2 : obj
            `numpy.ndarray` or `hamiltonian` class object to define the Hamiltonian operator as a matrix.

    Returns
    -------
    obj
            Commutator: :math:`[H_1,H_2] = H_1 H_2 - H_2 H_1`
    """
    if ishamiltonian(H1) or ishamiltonian(H2):
        return H1 * H2 - H2 * H1
    else:
        return H1.dot(H2) - H2.dot(H1)


def anti_commutator(H1, H2):
    """Calculates the anticommutator of two Hamiltonians :math:`H_1` and :math:`H_2`.

    .. math::
            \\{H_1,H_2\\}_+ = H_1 H_2 + H_2 H_1


    Examples
    --------
    The following script shows how to compute the anticommutator of two `hamiltonian` objects.

    .. literalinclude:: ../../doc_examples/anti_commutator-example.py
            :linenos:
            :language: python
            :lines: 7-

    Parameters
    ----------
    H1 : obj
            `numpy.ndarray` or `hamiltonian` class object to define the Hamiltonian operator as a matrix.
    H2 : obj
            `numpy.ndarray` or `hamiltonian` class object to define the Hamiltonian operator as a matrix.

    Returns
    -------
    obj
            Anticommutator: :math:`\\{H_1,H_2\\}_+ = H_1 H_2 + H_2 H_1`

    """
    if ishamiltonian(H1) or ishamiltonian(H2):
        return H1 * H2 + H2 * H1
    else:
        return H1.dot(H2) + H2.dot(H1)


class HamiltonianEfficiencyWarning(Warning):
    pass


# global names:
supported_dtypes = tuple([_np.float32, _np.float64, _np.complex64, _np.complex128])


def _check_static(sub_list):
    """Checks format of static list."""
    if (type(sub_list) in [list, tuple]) and (len(sub_list) == 2):
        if type(sub_list[0]) is not str:
            raise TypeError("expecting string type for opstr")
        if type(sub_list[1]) in [list, tuple]:
            for sub_sub_list in sub_list[1]:
                if (type(sub_sub_list) in [list, tuple]) and (len(sub_sub_list) > 0):
                    for element in sub_sub_list:
                        if not _np.isscalar(element):
                            raise TypeError("expecting scalar elements of indx")
                else:
                    raise TypeError("expecting list for indx")
        else:
            raise TypeError("expecting a list of one or more indx")
        return True
    else:
        return False


def _check_dynamic(sub_list):
    """Checks format of dynamic list."""
    if type(sub_list) in [list, tuple]:
        if len(sub_list) == 4:
            if type(sub_list[0]) is not str:
                raise TypeError("expecting string type for opstr")
            if type(sub_list[1]) in [list, tuple]:
                for sub_sub_list in sub_list[1]:
                    if (type(sub_sub_list) in [list, tuple]) and (
                        len(sub_sub_list) > 0
                    ):
                        for element in sub_sub_list:
                            if not _np.isscalar(element):
                                raise TypeError("expecting scalar elements of indx")
                    else:
                        raise TypeError("expecting list for indx")
            else:
                raise TypeError("expecting a list of one or more indx")
            if not hasattr(sub_list[2], "__call__"):
                raise TypeError("expecting callable object for driving function")
            if type(sub_list[3]) not in [list, tuple]:
                raise TypeError("expecting list for function arguments")
            return True
        elif len(sub_list) == 3:
            if not hasattr(sub_list[1], "__call__"):
                raise TypeError("expecting callable object for driving function")
            if type(sub_list[2]) not in [list, tuple]:
                raise TypeError("expecting list for function arguments")
            return False
        elif len(sub_list) == 2:
            if not hasattr(sub_list[1], "__call__"):
                raise TypeError("expecting callable object for driving function")
            return False
    else:
        raise TypeError(
            "expecting list with object, driving function, and function arguments"
        )


def _hamiltonian_dot(hamiltonian, time, v):
    """Used to create linear operator of a hamiltonian."""
    return hamiltonian.dot(v, time=time, check=False)


class hamiltonian(object):
    """Constructs time-dependent (hermitian and nonhermitian) operators.

    The `hamiltonian` class wraps most of the functionalty of the QuSpin package. This object allows the user to construct
    lattice Hamiltonians and operators, solve the time-dependent Schroedinger equation, do full/Lanczos
    diagonalization, etc.

    The user can create both static and time-dependent, hermitian and non-hermitian operators for any particle
    type (boson, spin, fermion) specified by the basis constructor.

    Notes
    -----
    One can instantiate the class either by parsing a set of symmetries, or an instance of `basis`. Note that
    instantiation with a `basis` will automatically ignore all symmetry inputs.

    Examples
    --------

    Here is an example how to employ a `basis` object to construct the periodically driven XXZ Hamiltonian

    .. math::
            H(t) = \\sum_{j=0}^{L-1} \\left( JS^z_{j+1}S^z_j + hS^z_j + g\cos(\\Omega t)S^x_j \\right)

    in the zero-momentum sector (`kblock=0`) of positive parity (`pblock=1`). We use periodic boundary conditions.

    The code snippet below initiates the class, and is required to run the example codes for the function methods.

    .. literalinclude:: ../../doc_examples/hamiltonian-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(
        self,
        static_list,
        dynamic_list,
        N=None,
        basis=None,
        shape=None,
        dtype=_np.complex128,
        static_fmt=None,
        dynamic_fmt=None,
        copy=True,
        check_symm=True,
        check_herm=True,
        check_pcon=True,
        **basis_kwargs,
    ):
        """Intializes the `hamtilonian` object (any quantum operator).

        Parameters
        ----------
        static_list : list
                Contains list of objects to calculate the static part of a `hamiltonian` operator. The format goes like:

                >>> static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]

        dynamic_list : list
                Contains list of objects to calculate the dynamic (time-dependent) part of a `hamiltonian` operator.
                The format goes like:

                >>> dynamic_list=[[opstr_1,[indx_11,...,indx_1n],fun_1,fun_1_args],[matrix_2,fun_2,fun_2_args],...]

                * `fun`: function object which multiplies the matrix or operator given in the same list.
                * `fun_args`: tuple of the extra arguments which go into the function to evaluate it like:

                        >>> f_val = fun(t,*fun_args)

                If the operator is time-INdependent, one must pass an empty list: `dynamic_list = []`.
        N : int, optional
                Number of lattice sites for the `hamiltonian` object.
        dtype : numpy.datatype, optional
                Data type (e.g. numpy.float64) to construct the operator with.
        static_fmt : str {"csr","csc","dia","dense"}, optional
                Specifies format of static part of Hamiltonian.
        dynamic_fmt: str {"csr","csc","dia","dense"} or  dict, keys: (func,func_args), values: str {"csr","csc","dia","dense"}
                Specifies the format of the dynamic parts of the hamiltonian. To specify a particular dynamic part of the hamiltonian use a tuple (func,func_args) which matches a function+argument pair
                used in the construction of the hamiltonian as a key in the dictionary.
        shape : tuple, optional
                Shape to create the `hamiltonian` object with. Default is `shape = None`.
        copy: bool, optional
                If set to `True`, this option creates a copy of the input array.
        check_symm : bool, optional
                Enable/Disable symmetry check on `static_list` and `dynamic_list`.
        check_herm : bool, optional
                Enable/Disable hermiticity check on `static_list` and `dynamic_list`.
        check_pcon : bool, optional
                Enable/Disable particle conservation check on `static_list` and `dynamic_list`.
        basis_kwargs : dict
                Optional additional arguments to pass to the `basis` class, if not already using a `basis` object
                to create the operator.

        """

        self._is_dense = False
        self._ndim = 2
        self._basis = basis

        if not (dtype in supported_dtypes):
            raise TypeError("hamiltonian does not support type: " + str(dtype))
        else:
            self._dtype = dtype

        if type(static_list) in [list, tuple]:
            static_opstr_list = []
            static_other_list = []
            for ele in static_list:
                if _check_static(ele):
                    static_opstr_list.append(ele)
                else:
                    static_other_list.append(ele)
        else:
            raise TypeError(
                "expecting list/tuple of lists/tuples containing opstr and list of indx"
            )

        if type(dynamic_list) in [list, tuple]:
            dynamic_opstr_list = []
            dynamic_other_list = []
            for ele in dynamic_list:
                if _check_dynamic(ele):
                    dynamic_opstr_list.append(ele)
                else:
                    dynamic_other_list.append(ele)
        else:
            raise TypeError(
                "expecting list/tuple of lists/tuples containing opstr and list of indx, functions, and function args"
            )

        # need for check_symm
        self._static_opstr_list = static_opstr_list
        self._dynamic_opstr_list = dynamic_opstr_list

        # if any operator strings present must get basis.
        if static_opstr_list or dynamic_opstr_list:
            if self._basis is not None:
                if len(basis_kwargs) > 0:
                    wrong_keys = set(basis_kwargs.keys())
                    temp = ", ".join(["{}" for key in wrong_keys])
                    raise ValueError(
                        ("unexpected optional argument(s): " + temp).format(*wrong_keys)
                    )

            # if not
            if self._basis is None:
                if N is None:  # if L is missing
                    raise Exception(
                        "if opstrs in use, argument N needed for basis class"
                    )

                if type(N) is not int:  # if L is not int
                    raise TypeError("argument N must be integer")

                self._basis = _default_basis(N, **basis_kwargs)

            elif not _isbasis(self._basis):
                raise TypeError("expecting instance of basis class for argument: basis")

            if check_herm:
                self._basis.check_hermitian(static_opstr_list, dynamic_opstr_list)

            if check_symm:
                self._basis.check_symm(static_opstr_list, dynamic_opstr_list)

            if check_pcon:
                self._basis.check_pcon(static_opstr_list, dynamic_opstr_list)

            self._static = make_static(self._basis, static_opstr_list, dtype)
            self._dynamic = make_dynamic(self._basis, dynamic_opstr_list, dtype)
            self._shape = self._static.shape

        if static_other_list or dynamic_other_list:
            if not hasattr(self, "_shape"):
                found = False
                if (
                    shape is None
                ):  # if no shape argument found, search to see if the inputs have shapes.
                    for i, O in enumerate(static_other_list):
                        try:  # take the first shape found
                            shape = O.shape
                            found = True
                            break
                        except AttributeError:
                            continue

                    if not found:
                        for tup in dynamic_other_list:
                            if len(tup) == 2:
                                O, _ = tup
                            else:
                                O, _, _ = tup

                            try:
                                shape = O.shape
                                found = True
                                break
                            except AttributeError:
                                continue
                else:
                    found = True

                if not found:
                    raise ValueError("missing argument shape")
                if shape[0] != shape[1]:
                    raise ValueError("hamiltonian must be square matrix")

                self._shape = shape
                self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)
                self._dynamic = {}

            for O in static_other_list:
                if _sp.issparse(O):
                    self._mat_checks(O)
                    if self._static is None:
                        self._static = O.astype(self._dtype, copy=copy)
                        continue

                    try:
                        self._static += O.astype(self._dtype)
                    except NotImplementedError:
                        self._static = self._static + O.astype(self._dtype)

                else:
                    O = _np.asarray(O, dtype=self._dtype)
                    self._mat_checks(O)

                    self._is_dense = True

                    if self._static is None:
                        self._static = O.astype(self._dtype, copy=copy)
                        continue

                    try:
                        self._static += O
                    except NotImplementedError:
                        self._static = self._static + O.astype(self._dtype)

            if not _sp.issparse(self._static):
                self._static = _np.asarray(self._static)

            try:
                self._static.sum_duplicates()
                self._static.eliminate_zeros()
            except:
                pass

            for tup in dynamic_other_list:
                if len(tup) == 2:
                    O, func = tup
                else:
                    O, f, f_args = tup
                    test_function(f, f_args, self._dtype)
                    func = function(f, tuple(f_args))

                if _sp.issparse(O):
                    self._mat_checks(O)

                    O = O.astype(self._dtype, copy=copy)
                else:
                    O = _np.array(O, copy=copy, dtype=self._dtype)
                    self._mat_checks(O)
                    self._is_dense = True

                if func in self._dynamic:
                    try:
                        self._dynamic[func] += O
                    except:
                        self._dynamic[func] = self._dynamic[func] + O
                else:
                    self._dynamic[func] = O

        else:
            if not hasattr(self, "_shape"):
                if shape is None:
                    # if not
                    if self._basis is None:
                        if N is None:  # if N is missing
                            raise Exception(
                                "argument N or shape needed to create empty hamiltonian"
                            )

                        if type(N) is not int:  # if L is not int
                            raise TypeError("argument N must be integer")

                        self._basis = _default_basis(N, **basis_kwargs)

                    elif not _isbasis(self._basis):
                        raise TypeError(
                            "expecting instance of basis class for argument: basis"
                        )

                    shape = (self._basis.Ns, self._basis.Ns)

                else:
                    self._basis = basis_kwargs.get("basis")
                    if not basis is None:
                        raise ValueError(
                            "empty hamiltonian only accepts basis or shape, not both"
                        )

                if len(shape) != 2:
                    raise ValueError("expecting ndim = 2")
                if shape[0] != shape[1]:
                    raise ValueError("hamiltonian must be square matrix")

                self._shape = shape
                self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)
                self._dynamic = {}

        self.update_matrix_formats(static_fmt, dynamic_fmt)
        self._Ns = self._shape[0]

    @property
    def basis(self):
        """:obj:`basis`: basis used to build the `hamiltonian` object.

        Defaults to `None` if operator has no basis (i.e. was created externally and passed as a precalculated array).

        """
        if self._basis is not None:
            return self._basis
        else:
            raise AttributeError("object has no attribute 'basis'")

    @property
    def ndim(self):
        """int: number of dimensions, always equal to 2."""
        return self._ndim

    @property
    def Ns(self):
        """int: number of states in the (symmetry-reduced) Hilbert space spanned by `basis`."""
        return self._Ns

    @property
    def shape(self):
        """tuple: shape of the `hamiltonian` object, always equal to `(Ns,Ns)`."""
        return self._shape

    # @property
    # def ndim(self):
    # 	"""int: number of dimensions, always equal to 2. """
    # 	return len(self._shape)

    @property
    def get_shape(self):
        """tuple: shape of the `hamiltonian` object, always equal to `(Ns,Ns)`."""
        return self._shape

    @property
    def is_dense(self):
        """bool: checks sparsity of operator matrix.

        `True` if the operator contains a dense matrix as a component of either
        the static or dynamic lists.

        """
        return self._is_dense

    @property
    def dtype(self):
        """type: data type of `hamiltonian` object."""
        return _np.dtype(self._dtype).name

    @property
    def static(self):
        """scipy.sparse.csr: static part of the operator."""
        return self._static

    @property
    def dynamic(self):
        """dict: contains dynamic parts of the operator as `dict(func=Hdyn)`.

        The key `func` is the memory address of the time-dependent function which can be called as `func(time)`. The function arguments are hard-coded, and are not passed.
        The value `Hdyn` is the sparse matrix to which the drive couples.

        """
        return self._dynamic

    @property
    def T(self):
        """:obj:`hamiltonian`: transposes the operator matrix, :math:`H_{ij}\\mapsto H_{ji}`."""
        return self.transpose()

    @property
    def H(self):
        """:obj:`hamiltonian`: transposes and conjugates the operator matrix, :math:`H_{ij}\\mapsto H_{ji}^*`."""
        return self.getH()

    @property
    def nbytes(self):
        """float: Total bytes consumed by the elements of the `hamiltonian` array."""
        nbytes = 0
        if _sp.issparse(self._static):
            nbytes += self._static.data.nbytes
            nbytes += self._static.indices.nbytes
            nbytes += self._static.indptr.nbytes
        else:
            nbytes += self._static.nbytes

        for Hd in itervalues(self._dynamic):
            if _sp.issparse(Hd):
                nbytes += Hd.data.nbytes
                nbytes += Hd.indices.nbytes
                nbytes += Hd.indptr.nbytes
            else:
                nbytes += Hd.nbytes

        return nbytes

    def check_is_dense(self):
        """updates attribute `_.is_dense`."""
        is_sparse = _sp.issparse(self._static)
        for Hd in itervalues(self._dynamic):
            is_sparse *= _sp.issparse(Hd)

        self._is_dense = not is_sparse

    def _get_matvecs(self):
        self._static_matvec = _get_matvec_function(self._static)
        self._dynamic_matvec = {}
        for func, Hd in iteritems(self._dynamic):
            self._dynamic_matvec[func] = _get_matvec_function(Hd)

    ### state manipulation/observable routines

    def dot(self, V, time=0, check=True, out=None, overwrite_out=True, a=1.0):
        """Matrix-vector multiplication of `hamiltonian` operator at time `time`, with state `V`.

        .. math::
                aH(t=\\texttt{time})|V\\rangle

        Notes
        -----
        * this function does the matrix multiplication with the state(s) and Hamiltonian as is, see Example 17 (Lidblad dynamics / Optical Bloch Equations)
        * for right-multiplication of quantum operators, see function `rdot()`.

        Parameters
        ----------
        V : {numpy.ndarray, scipy.spmatrix}
                Array containing the quantums state to multiply the `hamiltonian` operator with.
        time : obj, optional
                Can be either one of the following:

                * float: time to evalute the time-dependent part of the operator at (if operator has time dependence).
                        Default is `time = 0`.
                * (N,) array_like: if `V.shape[-1] == N`, the `hamiltonian` operator is evaluated at the i-th time
                                and dotted into `V[...,i]` to get the i-th slice of the output array. Here V must be either
                                2- or 3-d array, where 2-d would be for pure states and 3-d would be for mixed states.

        check : bool, optional
                Whether or not to do checks for shape compatibility.
        out : array_like, optional
                specify the output array for the the result. This is not supported if `V` is a sparse matrix or if `times` is an array.
        overwrite_out : bool, optional
                flag used to toggle between two different ways to treat `out`. If set to `True` all values in `out` will be overwritten with the result.
                If `False` the result of the dot product will be added to the values of `out`.
        a : scalar, optional
                scalar to multiply the final product with: :math:`B = aHV`.

        Returns
        -------
        numpy.ndarray
                Vector corresponding to the `hamiltonian` operator applied on the state `V`.

        Examples
        --------
        >>> B = H.dot(A,time=0,check=True)

        corresponds to :math:`B = HA`.

        """

        from quspin.operators.exp_op_core import isexp_op

        if ishamiltonian(V):
            return a * (self * V)
        elif isexp_op(V):
            raise ValueError(
                "This is an ambiguous operation. Use the .rdot() method of the `exp_op` class instead."
            )

        times = _np.array(time)

        if check:
            try:
                shape = V.shape
            except AttributeError:
                V = _np.asanyarray(V)
                shape = V.shape

            if shape[0] != self._shape[1]:
                raise ValueError(
                    "matrix dimension mismatch with shapes: {0} and {1}.".format(
                        V.shape, self._shape
                    )
                )

            if V.ndim > 3:
                raise ValueError("Expecting V.ndim < 4.")

        if V.ndim == 3 and times.ndim == 0:
            times = _np.broadcast_to(times, (V.shape[-1],))

        result_dtype = _np.result_type(V.dtype, self._dtype)

        if result_dtype not in supported_dtypes:
            raise TypeError("resulting dtype is not supported.")

        if times.ndim > 0:
            if times.ndim > 1:
                raise ValueError("Expecting time to be one dimensional array-like.")
            if V.shape[-1] != times.shape[0]:
                raise ValueError(
                    "For non-scalar times V.shape[-1] must be equal to len(time)."
                )

            if _sp.issparse(V):
                V = V.tocsc()
                return _sp.hstack(
                    [
                        a * self.dot(V.getcol(i), time=t, check=check)
                        for i, t in enumerate(time)
                    ]
                )
            else:
                if V.ndim == 3 and V.shape[0] != V.shape[1]:
                    raise ValueError("Density matrices must be square!")

                # allocate C-contiguous array to output results in.
                out = _np.zeros(V.shape[-1:] + V.shape[:-1], dtype=result_dtype)

                for i, t in enumerate(times):
                    v = _np.ascontiguousarray(V[..., i], dtype=result_dtype)
                    self._static_matvec(
                        self._static, v, overwrite_out=True, out=out[i, ...], a=a
                    )
                    for func, Hd in iteritems(self._dynamic):
                        self._dynamic_matvec[func](
                            Hd, v, overwrite_out=False, a=a * func(t), out=out[i, ...]
                        )

                # transpose, leave non-contiguous results which can be handled by numpy.
                if out.ndim == 2:
                    out = out.transpose()
                else:
                    out = out.transpose((1, 2, 0))

                return out

        else:
            if isinstance(V, _np.ndarray):
                V = V.astype(result_dtype, copy=False, order="C")

                if out is None:
                    out = self._static_matvec(self._static, V, a=a)
                else:
                    try:
                        if out.dtype != result_dtype:
                            raise TypeError(
                                "'out' must be array with correct dtype and dimensions for output array."
                            )
                        if out.shape != V.shape:
                            raise ValueError(
                                "'out' must be array with correct dtype and dimensions for output array."
                            )

                        if not out.flags["B"]:
                            raise ValueError(
                                "'out' must be array with correct dtype and dimensions for output array."
                            )
                    except AttributeError:
                        raise TypeError(
                            "'out' must be array with correct dtype and dimensions for output array."
                        )

                    self._static_matvec(
                        self._static, V, out=out, overwrite_out=overwrite_out, a=a
                    )

                for func, Hd in iteritems(self._dynamic):
                    self._dynamic_matvec[func](
                        Hd, V, overwrite_out=False, a=a * func(time), out=out
                    )

            elif _sp.issparse(V):

                if out is not None:
                    raise TypeError("'out' option does not apply for sparse inputs.")

                out = self._static * V
                for func, Hd in iteritems(self._dynamic):
                    out = out + func(time) * (Hd.dot(V))

                out = a * out
            else:
                # should we raise an error here?
                pass

            return out

    def rdot(self, V, time=0, check=True, out=None, overwrite_out=True, a=1.0):
        """Vector-Matrix multiplication of `hamiltonian` operator at time `time`, with state `V`.

        .. math::
                a\\langle V|H(t=\\texttt{time})

        Notes
        -----
        * this function does the matrix multiplication with the state(s) and Hamiltonian as is, see Example 17 (Lidblad dynamics / Optical Bloch Equations).

        Parameters
        ----------
        V : numpy.ndarray
                Vector (quantum state) to multiply the `hamiltonian` operator with on the left.
        time : obj, optional
                Can be either one of the following:

                * float: time to evalute the time-dependent part of the operator at (if existent).
                        Default is `time = 0`.
                * (N,) array_like: if `V.shape[-1] == N`, the `hamiltonian` operator is evaluated at the i-th time
                                and the mattrix multiplication on the right is calculated with respect to `V[...,i]`. Here V must be either
                                2- or 3-d array, where 2-d would be for pure states and 3-d would be for mixed states.
        check : bool, optional
                Whether or not to do checks for shape compatibility.
        out : array_like, optional
                specify the output array for the the result. This is not supported if `V` is a sparse matrix or if `times` is an array.
        overwrite_out : bool, optional
                flag used to toggle between two different ways to treat `out`. If set to `True` all values in `out` will be overwritten with the result.
                If `False` the result of the dot product will be added to the values of `out`.
        a : scalar, optional
                scalar to multiply the final product with: :math:`B = aVH`.

        Returns
        -------
        numpy.ndarray
                Vector corresponding to the `hamiltonian` operator applied on the state `V`.

        Examples
        --------
        >>> B = H.rdot(A,time=0,check=True)

        corresponds to :math:`B = AH`.

        """

        times = _np.array(time)

        try:
            ndim = V.ndim
        except AttributeError:
            V = _np.asanyarray(V)
            ndim = V.ndim

        if ndim not in [1, 2, 3]:
            raise ValueError("expecting V.ndim < 4.")

        if ndim == 1:
            return self.transpose().dot(
                V, time=times, check=check, out=out.T, overwrite_out=overwrite_out, a=a
            )
        elif ndim == 2:
            if _np.array(times).ndim > 0:
                return self.transpose().dot(
                    V,
                    time=times,
                    check=check,
                    out=out.T,
                    overwrite_out=overwrite_out,
                    a=a,
                )
            else:
                return (
                    self.transpose()
                    .dot(
                        V.transpose(),
                        time=times,
                        check=check,
                        out=out.T,
                        overwrite_out=overwrite_out,
                        a=a,
                    )
                    .transpose()
                )
        else:
            V_transpose = V.transpose((1, 0, 2))
            return (
                self.transpose()
                .dot(
                    V_transpose,
                    time=times,
                    check=check,
                    out=out.T,
                    overwrite_out=overwrite_out,
                    a=a,
                )
                .transpose((1, 0, 2))
            )

    def quant_fluct(self, V, time=0, check=True, enforce_pure=False):
        """Calculates the quantum fluctuations (variance) of `hamiltonian` operator at time `time`, in state `V`.

        .. math::
                \\langle V|H^2(t=\\texttt{time})|V\\rangle - \\langle V|H(t=\\texttt{time})|V\\rangle^2

        Parameters
        ----------
        V : numpy.ndarray
                Depending on the shape, can be a single state or a collection of pure or mixed states
                [see `enforce_pure`].
        time : obj, optional
                Can be either one of the following:

                * float: time to evalute the time-dependent part of the operator at (if existent).
                        Default is `time = 0`.
                * (N,) array_like: if `V.shape[-1] == N`, the `hamiltonian` operator is evaluated at the i-th time
                                and the fluctuations are calculated with respect to `V[...,i]`. Here V must be either
                                2- or 3-d array, where 2-d would be for pure states and 3-d would be for mixed states.

        enforce_pure : bool, optional
                Flag to enforce pure expectation value of `V` is a square matrix with multiple pure states
                in the columns.
        check : bool, optional

        Returns
        -------
        float
                Quantum fluctuations of `hamiltonian` operator in state `V`.

        Examples
        --------
        >>> H_fluct = H.quant_fluct(V,time=0,diagonal=False,check=True)

        corresponds to :math:`\\left(\\Delta H\\right)^2 = \\langle V|H^2(t=\\texttt{time})|V\\rangle - \\langle V|H(t=\\texttt{time})|V\\rangle^2`.

        """

        from quspin.operators.exp_op_core import isexp_op

        if ishamiltonian(V):
            raise TypeError("Can't take expectation value of hamiltonian")

        if isexp_op(V):
            raise TypeError("Can't take expectation value of exp_op")

        # fluctuations =  expctH2 - expctH^2
        kwargs = dict(enforce_pure=enforce_pure)
        V_dot = self.dot(V, time=time, check=check)
        expt_value_sq = self._expt_value_core(V, V_dot, **kwargs) ** 2

        if V_dot.ndim > 1 and V_dot.shape[0] != V_dot.shape[1] or enforce_pure:
            sq_expt_value = self._expt_value_core(V_dot, V_dot, **kwargs)
        else:
            V_dot = self.dot(V_dot, time=time, check=check)
            sq_expt_value = self._expt_value_core(V, V_dot, **kwargs)

        return sq_expt_value - expt_value_sq

    def expt_value(self, V, time=0, check=True, enforce_pure=False):
        """Calculates expectation value of `hamiltonian` operator at time `time`, in state `V`.

        .. math::
                \\langle V|H(t=\\texttt{time})|V\\rangle,\\qquad \\mathrm{tr}(V H(t=\\texttt{time}))

        Parameters
        ----------
        V : numpy.ndarray
                Depending on the shape, can be a single state or a collection of pure or mixed states
                [see `enforce_pure` argument of `basis.ent_entropy`].
        time : obj, optional
                Can be either one of the following:

                * float: time to evalute the time-dependent part of the operator at (if existent).
                        Default is `time = 0`.
                * (N,) array_like: if `V.shape[-1] == N`, the `hamiltonian` operator is evaluated at the i-th time
                                and the expectation value is calculated with respect to `V[...,i]`. Here V must be either
                                2- or 3-d array, where 2-d would be for pure states and 3-d would be for mixed states.
        enforce_pure : bool, optional
                Flag to enforce pure expectation value of `V` is a square matrix with multiple pure states
                in the columns.
        check : bool, optional

        Returns
        -------
        float
                Expectation value of `hamiltonian` operator in state `V`.

        Examples
        --------
        >>> H_expt = H.expt_value(V,time=0,diagonal=False,check=True)

        corresponds to :math:`H_{expt} = \\langle V|H(t=0)|V\\rangle`.

        """
        from quspin.operators.exp_op_core import isexp_op

        if ishamiltonian(V):
            raise TypeError("Can't take expectation value of hamiltonian")

        if isexp_op(V):
            raise TypeError("Can't take expectation value of exp_op")

        V_dot = self.dot(V, time=time, check=check)
        return self._expt_value_core(V, V_dot, enforce_pure=enforce_pure)

    def _expt_value_core(self, V_left, V_right, enforce_pure=False):
        if _sp.issparse(V_right):
            if V_left.shape[0] != V_left.shape[1] or enforce_pure:  # pure states
                return _np.asarray(
                    (V_right.multiply(V_left.conj())).sum(axis=0)
                ).squeeze()
            else:  # density matrix
                return V_right.diagonal().sum()
        else:
            V_right = _np.asarray(V_right)
            if V_right.ndim == 1:  # single pure state
                return _np.vdot(V_left, V_right)
            elif (
                V_right.shape[0] != V_right.shape[1] or enforce_pure
            ):  # multiple pure states
                return _np.einsum("ij,ij->j", V_left.conj(), V_right)
            else:  # density matrices
                return _np.einsum("ii...->...", V_right)

    def matrix_ele(self, Vl, Vr, time=0, diagonal=False, check=True):
        """Calculates matrix element of `hamiltonian` operator at time `time` in states `Vl` and `Vr`.

        .. math::
                \\langle V_l|H(t=\\texttt{time})|V_r\\rangle

        Notes
        -----
        Taking the conjugate or transpose of the state `Vl` is done automatically.

        Parameters
        ----------
        Vl : {numpy.ndarray, scipy.spmatrix}
                Vector(s)/state(s) to multiple with on left side.
        Vl : {numpy.ndarray, scipy.spmatrix}
                Vector(s)/state(s) to multiple with on right side.
        time : obj, optional
                Can be either one of the following:

                * float: time to evalute the time-dependent part of the operator at (if existent).
                        Default is `time = 0`.
                * (N,) array_like: if `V.shape[1] == N`, the `hamiltonian` operator is evaluated at the i-th time
                                and the fluctuations are calculated with respect to `V[:,i]`. Here V must be a 2-d array
                                containing pure states in the columns of the array.
        diagonal : bool, optional
                When set to `True`, returs only diagonal part of expectation value. Default is `diagonal = False`.
        check : bool,

        Returns
        -------
        float
                Matrix element of `hamiltonian` operator between the states `Vl` and `Vr`.

        Examples
        --------
        >>> H_lr = H.expt_value(Vl,Vr,time=0,diagonal=False,check=True)

        corresponds to :math:`H_{lr} = \\langle V_l|H(t=0)|V_r\\rangle`.

        """

        Vr = self.dot(Vr, time=time, check=check)

        if check:
            try:
                shape = Vl.shape
            except AttributeError:
                Vl = _np.asarray(Vl)
                shape = Vl.shape

            if Vl.shape[0] != self._shape[1]:
                raise ValueError(
                    "matrix dimension mismatch with shapes: {0} and {1}.".format(
                        Vl.shape, self._shape
                    )
                )

            if diagonal:
                if Vl.shape[1] != Vr.shape[1]:
                    raise ValueError(
                        "number of vectors must be equal for diagonal=True."
                    )

            if Vr.ndim > 2:
                raise ValueError("Expecting Vr to have ndim < 3")

        if _sp.issparse(Vl):
            if diagonal:
                return _np.asarray(Vl.conj().multiply(Vr).sum(axis=0)).squeeze()
            else:
                return Vl.T.conj().dot(Vr)
        elif _sp.issparse(Vr):
            if diagonal:
                return _np.asarray(Vr.multiply(Vl.conj()).sum(axis=0)).squeeze()
            else:
                return Vr.T.dot(Vl.conj())
        else:
            if diagonal:
                return _np.einsum("ij,ij->j", Vl.conj(), Vr)
            else:
                return Vl.T.conj().dot(Vr)

    ### transformation routines

    def project_to(self, proj):
        """Projects/Transforms `hamiltonian` operator with projector/operator `proj`.

        Let us call the projector/transformation :math:`V`. Then, the function computes

        .. math::
                V^\\dagger H V

        Notes
        -----
        The `proj` argument can be a square array, in which case the function just transforms the
        `hailtonian` operator :math:`H`. Or it can be a projector which then projects :math:`H` onto
        a smaller Hilbert space.

        Projectors onto bases with symmetries other than `H.basis` can be conveniently obtain using the
        `basis.get_proj()` method of the basis constructor class.

        Parameters
        ----------
        proj : obj
                Can be either one of the following:

                        * `hamiltonian` object
                        * `exp_op` object
                        * `numpy.ndarray`
                        * `scipy.sparse` array

                The shape of `proj` need not be square, but has to comply with the matrix multiplication requirements
                in the definition above.

        Returns
        -------
        obj
                Projected/Transformed `hamiltonian` operator. The output object type depends on the object
                type of `proj`.

        Examples
        --------

        >>> H_new = H.project_to(V)

        correponds to :math:`V^\\dagger H V`.

        """
        from quspin.operators.exp_op_core import isexp_op

        if ishamiltonian(proj):
            new = self._rmul_hamiltonian(proj.getH())
            return new._imul_hamiltonian(proj)

        elif isexp_op(proj):
            return proj.sandwich(self)

        elif _sp.issparse(proj):
            if self._shape[1] != proj.shape[0]:
                raise ValueError(
                    "matrix dimension mismatch with shapes: {0} and {1}.".format(
                        proj.shape, self._shape
                    )
                )

            new = self._rmul_sparse(proj.getH())
            new._shape = (proj.shape[1], proj.shape[1])
            return new._imul_sparse(proj)

        elif _np.isscalar(proj):
            raise NotImplementedError

        elif proj.__class__ == _np.ndarray:
            if self._shape[1] != proj.shape[0]:
                raise ValueError(
                    "matrix dimension mismatch with shapes: {0} and {1}.".format(
                        proj.shape, self._shape
                    )
                )

            new = self._rmul_dense(proj.T.conj())
            new._shape = (proj.shape[1], proj.shape[1])
            return new._imul_dense(proj)

        elif proj.__class__ == _np.matrix:
            if self._shape[1] != proj.shape[0]:
                raise ValueError(
                    "matrix dimension mismatch with shapes: {0} and {1}.".format(
                        proj.shape, self._shape
                    )
                )

            new = self._rmul_dense(proj.T.conj())
            new._shape = (proj.shape[1], proj.shape[1])
            return new._imul_dense(proj)

        else:
            proj = _np.asanyarray(proj)
            if self._shape[1] != proj.shape[0]:
                raise ValueError(
                    "matrix dimension mismatch with shapes: {0} and {1}.".format(
                        proj.shape, self._shape
                    )
                )

            new = self._rmul_dense(proj.T.conj())
            new._shape = (proj.shape[1], proj.shape[1])
            return new._imul_dense(proj)

    def rotate_by(self, other, generator=False, **exp_op_kwargs):
        """Rotates/Transforms `hamiltonian` object by an operator `other`.

        Let us denote the transformation by :math:`V`. With `generator=False`, `other` corresponds to the
        transformation :math:`V`, and this function implements

        .. math::
                V^\\dagger H V

        while for `generator=True`, `other` corresponds to a generator :math:`K`, and the function implements

        .. math::
                \\exp(a^*K^\\dagger) H \\exp(a K)

        Notes
        -----
        If `generator = False`, this function calls `project_to`.

        Parameters
        ----------
        other : obj
                Can be either one of the following:

                        * `hamiltonian` object
                        * `exp_op` object
                        * `numpy.ndarray`
                        * `scipy.sparse` array
        generator : bool, optional
                If set to `True`, this flag renders `other` a generator, and implements the calculation of

                .. math::
                        \\exp(a^*K^\\dagger) H \\exp(a K)

                If set to `False`, the function implements

                .. math::
                        V^\\dagger H V

                Default is `generator = False`.

        All other optional arguments are the same as for the `exp_op` class.

        Returns
        -------
        obj
                Transformed `hamiltonian` operator. The output object type depends on the object type of `other`.

        Examples
        --------
        >>> H_new = H.rotate_by(V,generator=False)

        corresponds to :math:`V^\\dagger H V`.

        >>> H_new = H.rotate_by(K,generator=True,**exp_op_kwargs)

        corresponds to :math:`\\exp(K^\\dagger) H \\exp(K)`.

        """
        from quspin.operators.exp_op_core import exp_op

        if generator:
            return exp_op(other, **exp_op_kwargs).sandwich(self)
        else:
            return self.project_to(other)

    ### Diagonalisation routines

    def eigsh(self, time=0.0, **eigsh_args):
        """Computes SOME eigenvalues and eigenvectors of hermitian `hamiltonian` operator using SPARSE hermitian methods.

        This function method solves for eigenvalues and eigenvectors, but can only solve for a few of them accurately.
        It calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_, which is a wrapper for ARPACK.

        Notes
        -----
        Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
        to reassure themselves of the hermiticity properties before use.

        Parameters
        ----------
        time : float
                Time to evalute the `hamiltonian` operator at (if time dependent). Default is `time = 0.0`.
        eigsh_args :
                For all additional arguments see documentation of `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_.

        Returns
        -------
        tuple
                Tuple containing the `(eigenvalues, eigenvectors)` of the `hamiltonian` operator.

        Examples
        --------
        >>> eigenvalues,eigenvectors = H.eigsh(time=time,**eigsh_args)

        """
        if self.Ns <= 0:
            try:
                return_eigenvectors = eigsh_args["return_eigenvectors"]
            except KeyError:
                return_eigenvectors = True

            if return_eigenvectors:
                return _np.array([], dtype=self._dtype).real, _np.array(
                    [[]], dtype=self._dtype
                )
            else:
                return _np.array([], dtype=self._dtype).real

        return _sla.eigsh(self.tocsr(time=time), **eigsh_args)

    def eigh(self, time=0, **eigh_args):
        """Computes COMPLETE eigensystem of hermitian `hamiltonian` operator using DENSE hermitian methods.

        This function method solves for all eigenvalues and eigenvectors. It calls
        `numpy.linalg.eigh <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html>`_,
        and uses wrapped LAPACK functions which are contained in the module py_lapack.

        Notes
        -----
        Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
        to reassure themselves of the hermiticity properties before use.

        Parameters
        ----------
        time : float
                Time to evalute the `hamiltonian` operator at (if time dependent). Default is `time = 0.0`.
        eigh_args :
                For all additional arguments see documentation of `numpy.linalg.eigh <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html>`_.

        Returns
        -------
        tuple
                Tuple containing the `(eigenvalues, eigenvectors)` of the `hamiltonian` operator.

        Examples
        --------
        >>> eigenvalues,eigenvectors = H.eigh(time=time,**eigh_args)

        """
        if self.Ns <= 0:
            return _np.array([], dtype=self._dtype).real, _np.array(
                [[]], dtype=self._dtype
            )

        eigh_args["overwrite_a"] = True
        # fill dense array with hamiltonian
        H_dense = self.todense(time=time)
        # calculate eigh
        return _la.eigh(H_dense, **eigh_args)

    def eigvalsh(self, time=0, **eigvalsh_args):
        """Computes ALL eigenvalues of hermitian `hamiltonian` operator using DENSE hermitian methods.

        This function method solves for all eigenvalues. It calls
        `numpy.linalg.eigvalsh <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html>`_,
        and uses wrapped LAPACK functions which are contained in the module py_lapack.

        Notes
        -----
        Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
        to reassure themselves of the hermiticity properties before use.

        Parameters
        ----------
        time : float
                Time to evalute the `hamiltonian` operator at (if time dependent). Default is `time = 0.0`.
        eigvalsh_args :
                For all additional arguments see documentation of `numpy.linalg.eigvalsh <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html>`_.

        Returns
        -------
        numpy.ndarray
                Eigenvalues of the `hamiltonian` operator.

        Examples
        --------
        >>> eigenvalues = H.eigvalsh(time=time,**eigvalsh_args)

        """
        if self.Ns <= 0:
            return _np.array([], dtype=self._dtype).real

        H_dense = self.todense(time=time)
        eigvalsh_args["overwrite_a"] = True
        return _la.eigvalsh(H_dense, **eigvalsh_args)

    ### Schroedinger evolution routines

    def __LO(self, time, rho, rho_out):
        """
        args:
                rho, flattened density matrix to multiple with
                time, the time to evalute drive at.

        description:
                This function is what gets passed into the ode solver. This is the real time Liouville operator.

        """
        rho = rho.reshape((self.Ns, self.Ns))
        self._static_matvec(
            self._static, rho, out=rho_out, a=+1.0, overwrite_out=True
        )  # rho_out = self._static.dot(rho)
        self._static_matvec(
            self._static.T, rho.T, out=rho_out.T, a=-1.0, overwrite_out=False
        )  # rho_out -= (self._static.T.dot(rho.T)).T
        for func, Hd in iteritems(self._dynamic):
            ft = func(time)
            self._dynamic_matvec[func](
                Hd, rho, out=rho_out, a=+ft, overwrite_out=False
            )  # rho_out += ft*Hd.dot(rho)
            self._dynamic_matvec[func](
                Hd.T, rho.T, out=rho_out.T, a=-ft, overwrite_out=False
            )  # rho_out -= ft*(Hd.T.dot(rho.T)).T

        rho_out *= -1j
        return rho_out.ravel()

    def __ISO(self, time, V, V_out):
        """
        args:
                V, the vector to multiple with
                V_out, the vector to use with output.
                time, the time to evalute drive at.

        description:
                This function is what gets passed into the ode solver. This is the Imaginary time Schrodinger operator -H(t)*|V >
        """
        V = V.reshape(V_out.shape)
        self._static_matvec(self._static, V, out=V_out, overwrite_out=True)
        for func, Hd in iteritems(self._dynamic):
            self._dynamic_matvec[func](
                Hd, V, a=func(time), out=V_out, overwrite_out=False
            )

        V_out *= -1.0
        return V_out.ravel()

    def __SO_real(self, time, V, V_out):
        """
        args:
                V, the vector to multiple with
                V_out, the vector to use with output.
                time, the time to evalute drive at.

        description:
                This function is what gets passed into the ode solver. This is the real time Schrodinger operator -i*H(t)*|V >
                This function is designed for real hamiltonians and increases the speed of integration compared to __SO

        u_dot + i*v_dot = -i*H(u + i*v)
        u_dot = Hv
        v_dot = -Hu
        """
        V = V.reshape(V_out.shape)
        self._static_matvec(
            self._static, V[self._Ns :], out=V_out[: self._Ns], a=+1, overwrite_out=True
        )  # V_dot[:self._Ns] =  self._static.dot(V[self._Ns:])
        self._static_matvec(
            self._static, V[: self._Ns], out=V_out[self._Ns :], a=-1, overwrite_out=True
        )  # V_dot[self._Ns:] = -self._static.dot(V[:self._Ns])
        for func, Hd in iteritems(self._dynamic):
            ft = func(time)
            self._dynamic_matvec[func](
                Hd, V[self._Ns :], out=V_out[: self._Ns], a=+ft, overwrite_out=False
            )  # V_dot[:self._Ns] += func(time)*Hd.dot(V[self._Ns:])
            self._dynamic_matvec[func](
                Hd, V[: self._Ns], out=V_out[self._Ns :], a=-ft, overwrite_out=False
            )  # V_dot[self._Ns:] += -func(time)*Hd.dot(V[:self._Ns])

        return V_out

    def __SO(self, time, V, V_out):
        """
        args:
                V, the vector to multiple with
                V_out, the vector to use with output.
                time, the time to evalute drive at.

        description:
                This function is what gets passed into the ode solver. This is the Imaginary time Schrodinger operator -H(t)*|V >
        """
        V = V.reshape(V_out.shape)
        self._static_matvec(self._static, V, out=V_out, overwrite_out=True)
        for func, Hd in iteritems(self._dynamic):
            self._dynamic_matvec[func](Hd,V,a=func(time),out=V_out,overwrite_out=False)
            #V_out+=func(time)*Hd@V

        V_out *= -1j
        return V_out.ravel()

    # def SO(self, time, V, V_out):
    #     return self.__SO(time, V, V_out)

    def evolve(
        self,
        v0,
        t0,
        times,
        eom="SE",
        solver_name="dop853",
        stack_state=False,
        verbose=False,
        iterate=False,
        imag_time=False,
        **solver_args,
    ):
        """Implements (imaginary) time evolution generated by the `hamiltonian` object.

        The functions handles evolution generated by both time-dependent and time-independent Hamiltonians.

        Currently the following three built-in routines are supported (see parameter `eom`):
                i) real-time Schroedinger equation: :math:`\\partial_t|v(t)\\rangle=-iH(t)|v(t)\\rangle`.
                ii) imaginary-time Schroedinger equation: :math:`\\partial_t|v(t)\\rangle=-H(t)|v(t)\\rangle`.
                iii) Liouvillian dynamics: :math:`\\partial_t\\rho(t)=-i[H,\\rho(t)]`.

        Notes
        -----
        Supports evolution of multiple states simulataneously (`eom="SE") and evolution of mixed
        and pure density matrices (`eom="LvNE"). For a user-defined custom ODE solver which can handle non-linear equations, check out the
        `measurements.evolve()` routine, which has a similar functionality but allows for a complete freedom
        over the differential equation to be solved.

        Parameters
        ----------
        v0 : numpy.ndarray
                Initial state :math:`|v(t)\\rangle` or density matrix (pure and mixed) :math:`\\rho(t)`.
        t0 : float
                Initial time.
        times : numpy.ndarray
                Vector of times to compute the time-evolved state at.
        eom : str, optional
                Specifies the ODE type. Can be either one of

                        * "SE", real and imaginary-time Schroedinger equation.
                        * "LvNE", real-time Liouville equation.

                Default is "eom = SE" (Schroedinger evolution).
        iterate : bool, optional
                If set to `True`, creates a generator object for the time-evolved the state. Default is `False`.
        solver_name : str, optional
                Scipy solver integrator name. Default is `dop853`.

                See `scipy integrator (solver) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_ for other options.
        solver_args : dict, optional
                Dictionary with additional `scipy integrator (solver) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html>`_.
        stack_state : bool, optional
                Flag to determine if `f` is real or complex-valued. Default is `False` (i.e. complex-valued).
        imag_time : bool, optional
                Must be set to `True` when `f` defines imaginary-time evolution, in order to normalise the state
                at each time in `times`. Default is `False`.
        verbose : bool, optional
                If set to `True`, prints normalisation of state at teach time in `times`.

        Returns
        -------
        obj
                Can be either one of the following:

                * numpy.ndarray containing evolved state against time.
                * generator object for time-evolved state (requires `iterate = True`).

                Note that for Liouvillian dynamics the output is a square complex `numpy.ndarray`.

        Examples
        --------
        >>> v_t = H.evolve(v0,t0,times,eom="SE",solver_name="dop853",verbose=False,iterate=False,imag_time=False,**solver_args)

        """

        try:
            shape0 = v0.shape
        except AttributeError:
            v0 = _np.asanyarray(v0)
            shape0 = v0.shape

        if _np.iscomplexobj(times):
            raise ValueError("times must be real number(s).")

        evolve_args = (v0, t0, times)
        evolve_kwargs = solver_args
        evolve_kwargs["solver_name"] = solver_name
        evolve_kwargs["stack_state"] = stack_state
        evolve_kwargs["verbose"] = verbose
        evolve_kwargs["iterate"] = iterate
        evolve_kwargs["imag_time"] = imag_time

        if eom == "SE":
            if v0.ndim > 2:
                raise ValueError("v0 must have ndim <= 2")

            if v0.shape[0] != self.Ns:
                raise ValueError("v0 must have {0} elements".format(self.Ns))

            if imag_time:
                if stack_state:
                    raise NotImplementedError(
                        "stack state is not compatible with imaginary time evolution."
                    )

                evolve_args = evolve_args + (self.__ISO,)
                result_dtype = _np.result_type(v0.dtype, self.dtype, _np.float64)
                v0 = _np.array(v0, dtype=result_dtype, copy=True, order="C")
                evolve_kwargs["f_params"] = (v0,)
                evolve_kwargs["real"] = not _np.iscomplexobj(v0)

            else:
                evolve_kwargs["real"] = False
                if stack_state:
                    if _np.iscomplexobj(
                        _np.array(1, dtype=self.dtype)
                    ):  # no idea how to do this in python :D
                        raise ValueError(
                            "stack_state option cannot be used with complex-valued Hamiltonians"
                        )
                    shape = (v0.shape[0] * 2,) + v0.shape[1:]
                    v0 = _np.zeros(shape, dtype=_np.float64, order="C")
                    evolve_kwargs["f_params"] = (v0,)

                    evolve_args = evolve_args + (self.__SO_real,)
                else:
                    v0 = _np.array(v0, dtype=_np.complex128, copy=True, order="C")
                    evolve_kwargs["f_params"] = (v0,)
                    evolve_args = evolve_args + (self.__SO,)

        elif eom == "LvNE":
            n = 1.0
            if v0.ndim != 2:
                raise ValueError("v0 must have ndim = 2")

            if v0.shape != self._shape:
                raise ValueError("v0 must be same shape as Hamiltonian")

            if imag_time:
                raise NotImplementedError(
                    "imaginary time not implemented for Liouville-von Neumann dynamics"
                )
            else:
                if stack_state:
                    raise NotImplementedError(
                        "stack_state not implemented for Liouville-von Neumann dynamics"
                    )
                else:
                    v0 = _np.array(v0, dtype=_np.complex128, copy=True, order="C")
                    evolve_kwargs["f_params"] = (v0,)
                    evolve_args = evolve_args + (self.__LO,)
        else:
            raise ValueError(
                "'{} equation' not recognized, must be 'SE' or 'LvNE'".format(eom)
            )

        return evolve(*evolve_args, **evolve_kwargs)

    ### routines to change object type

    def aslinearoperator(self, time=0.0):
        """Returns copy of a `hamiltonian` object at time `time` as a `scipy.sparse.linalg.LinearOperator`.

        Casts the `hamiltonian` object as a
        `scipy.sparse.linalg.LinearOperator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
        object.

        Parameters
        ----------
        time : float, optional
                Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.

        Returns
        -------
        :obj:`scipy.sparse.linalg.LinearOperator`

        Examples
        --------
        >>> H_aslinop=H.aslinearoperator(time=time)

        """
        time = _np.array(time)
        if time.ndim > 0:
            raise TypeError("expecting scalar argument for time")

        matvec = functools.partial(_hamiltonian_dot, self, time)
        rmatvec = functools.partial(_hamiltonian_dot, self.T.conj(), time)
        return _sla.LinearOperator(
            self.get_shape, matvec, rmatvec=rmatvec, matmat=matvec, dtype=self._dtype
        )

    def tocsr(self, time=0):
        """Returns copy of a `hamiltonian` object at time `time` as a `scipy.sparse.csr_matrix`.

        Casts the `hamiltonian` object as a
        `scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_
        object.

        Parameters
        ----------
        time : float, optional
                Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.

        Returns
        -------
        :obj:`scipy.sparse.csr_matrix`

        Examples
        --------
        >>> H_csr=H.tocsr(time=time)

        """

        if _np.array(time).ndim > 0:
            raise TypeError("expecting scalar argument for time")

        H = _sp.csr_matrix(self._static)

        for func, Hd in iteritems(self._dynamic):
            Hd = _sp.csr_matrix(Hd)
            try:
                H += Hd * func(time)
            except:
                H = H + Hd * func(time)

        return H

    def tocsc(self, time=0):
        """Returns copy of a `hamiltonian` object at time `time` as a `scipy.sparse.csc_matrix`.

        Casts the `hamiltonian` object as a
        `scipy.sparse.csc_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html>`_
        object.

        Parameters
        ----------
        time : float, optional
                Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.

        Returns
        -------
        :obj:`scipy.sparse.csc_matrix`

        Examples
        --------
        >>> H_csc=H.tocsc(time=time)

        """
        if _np.array(time).ndim > 0:
            raise TypeError("expecting scalar argument for time")

        H = _sp.csc_matrix(self._static)
        for func, Hd in iteritems(self._dynamic):
            Hd = _sp.csc_matrix(Hd)
            try:
                H += Hd * func(time)
            except:
                H = H + Hd * func(time)

        return H

    def todense(self, time=0, order=None, out=None):
        """Returns copy of a `hamiltonian` object at time `time` as a dense array.

        This function can overflow memory if not used carefully!

        Notes
        -----
        If the array dimension is too large, scipy may choose to cast the `hamiltonian` operator as a
        `numpy.matrix` instead of a `numpy.ndarray`. In such a case, one can use the `hamiltonian.toarray()`
        method.

        Parameters
        ----------
        time : float, optional
                Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
        order : str, optional
                Whether to store multi-dimensional data in C (rom-major) or Fortran (molumn-major) order in memory.

                Default is `order = None`, indicating the NumPy default of C-ordered.

                Cannot be specified in conjunction with the `out` argument.
        out : numpy.ndarray
                Array to fill in with the output.

        Returns
        -------
        obj
                Depending of size of array, can be either one of

                * `numpy.ndarray`.
                * `numpy.matrix`.

        Examples
        --------
        >>> H_dense=H.todense(time=time)

        """

        if _np.array(time).ndim > 0:
            raise TypeError("expecting scalar argument for time")

        if out is None:
            out = _np.zeros(self._shape, dtype=self.dtype)
            out = _np.asmatrix(out)

        if _sp.issparse(self._static):
            self._static.todense(order=order, out=out)
        else:
            out[:] = self._static[:]

        for func, Hd in iteritems(self._dynamic):
            out += Hd * func(time)

        return out

    def toarray(self, time=0, order=None, out=None):
        """Returns copy of a `hamiltonian` object at time `time` as a dense array.

        This function can overflow memory if not used carefully!


        Parameters
        ----------
        time : float, optional
                Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.
        order : str, optional
                Whether to store multi-dimensional data in C (rom-major) or Fortran (molumn-major) order in memory.

                Default is `order = None`, indicating the NumPy default of C-ordered.

                Cannot be specified in conjunction with the `out` argument.
        out : numpy.ndarray
                Array to fill in with the output.

        Returns
        -------
        numpy.ndarray
                Dense array.

        Examples
        --------
        >>> H_dense=H.toarray(time=time)

        """

        if _np.array(time).ndim > 0:
            raise TypeError("expecting scalar argument for time")

        if out is None:
            out = _np.zeros(self._shape, dtype=self.dtype)

        if _sp.issparse(self._static):
            self._static.toarray(order=order, out=out)
        else:
            out[:] = self._static[:]

        for func, Hd in iteritems(self._dynamic):
            out += Hd * func(time)

        return out

    def update_matrix_formats(self, static_fmt, dynamic_fmt):
        """Change the internal structure of the matrices in-place.

        Parameters
        ----------
        static_fmt : str {"csr","csc","dia","dense"}
                Specifies format of static part of Hamiltonian.
        dynamic_fmt: str {"csr","csc","dia","dense"} or  dict, keys: (func,func_args), values: str {"csr","csc","dia","dense"}
                Specifies the format of the dynamic parts of the hamiltonian. To specify a particular dynamic part of the hamiltonian use a tuple (func,func_args) which matches a function+argument pair
                used in the construction of the hamiltonian as a key in the dictionary.
        copy : bool,optional
                Whether to return a deep copy of the original object. Default is `copy = False`.

        Examples
        --------
        make the dynamic part of the `hamiltonian` object to be DIA matrix format and have the static part be CSR matrix format:

        >>> H.update_matrix_formats(static_fmt="csr",dynamic_fmt={(func,func_args):"dia"})


        """
        if static_fmt is not None:
            if type(static_fmt) is not str:
                raise ValueError("Expecting string for 'sparse_fmt'")

            if static_fmt not in ["csr", "csc", "dia", "dense"]:
                raise ValueError(
                    "'{0}' is not a valid sparse format for Hamiltonian class.".format(
                        static_fmt
                    )
                )

            if static_fmt == "dense":
                if _sp.issparse(self._static):
                    self._static = self._static.toarray()
                else:
                    self._static = _np.ascontiguousarray(self._static)
            else:
                sparse_constuctor = getattr(_sp, static_fmt + "_matrix")
                self._static = sparse_constuctor(self._static)

        if dynamic_fmt is not None:
            if type(dynamic_fmt) is str:

                if dynamic_fmt not in ["csr", "csc", "dia", "dense"]:
                    raise ValueError(
                        "'{0}' is not a valid sparse format for Hamiltonian class.".format(
                            dynamic_fmt
                        )
                    )

                if dynamic_fmt == "dense":
                    updates = {
                        func: sparse_constuctor(Hd)
                        for func, Hd in iteritems(self._dynamic)
                        if _sp.issparse(Hd)
                    }
                    updates.update(
                        {
                            func: _np.ascontiguousarray(Hd)
                            for func, Hd in iteritems(self._dynamic)
                            if not _sp.issparse(Hd)
                        }
                    )
                else:
                    updates = {
                        func: sparse_constuctor(Hd)
                        for func, Hd in iteritems(self._dynamic)
                    }

                self._dynamic.update(updates)

            elif type(dynamic_fmt) in [list, tuple]:
                for fmt, (f, f_args) in dynamic_fmt:

                    func = function(f, tuple(f_args))

                    if fmt not in ["csr", "csc", "dia", "dense"]:
                        raise ValueError(
                            "'{0}' is not a valid sparse format for Hamiltonian class.".format(
                                fmt
                            )
                        )

                    try:
                        if fmt == "dense":
                            if _sp.issparse(self._static):
                                self._dynamic[func] = self._dynamic[func].toarray()
                            else:
                                self._dynamic[func] = _np.ascontiguousarray(
                                    self._dynamic[func]
                                )
                        else:
                            sparse_constuctor = getattr(_sp, fmt + "_matrix")
                            self._dynamic[func] = sparse_constuctor(self._dynamic[func])

                    except KeyError:
                        raise ValueError(
                            "({},{}) is not found in dynamic list.".format(f, f_args)
                        )

        self._get_matvecs()

    def as_dense_format(self, copy=False):
        """Casts `hamiltonian` operator to DENSE format.

        Parameters
        ----------
        copy : bool,optional
                Whether to return a deep copy of the original object. Default is `copy = False`.

        Returns
        -------
        obj
                Either one of the following:

                * Shallow copy, if `copy = False`.
                * Deep copy, if `copy = True`.

        Examples
        --------
        >>> H_dense=H.as_dense_format()

        """
        if _sp.issparse(self._static):
            new_static = self._static.toarray()
        else:
            new_static = _np.asarray(self._static, copy=copy)

        dynamic = [
            ([M.toarray(), func] if _sp.issparse(M) else [M, func])
            for func, M in iteritems(self.dynamic)
        ]

        return hamiltonian(
            [new_static], dynamic, basis=self._basis, dtype=self._dtype, copy=copy
        )

    def as_sparse_format(self, static_fmt="csr", dynamic_fmt={}, copy=False):
        """Casts `hamiltonian` operator to SPARSE format(s).

        Parameters
        ----------
        static_fmt : str {"csr","csc","dia","dense"}
                Specifies format of static part of Hamiltonian.
        dynamic_fmt: str {"csr","csc","dia","dense"} or  dict, keys: (func,func_args), values: str {"csr","csc","dia","dense"}
                Specifies the format of the dynamic parts of the hamiltonian. To specify a particular dynamic part of the hamiltonian use a tuple (func,func_args) which matches a function+argument pair
                used in the construction of the hamiltonian as a key in the dictionary.
        copy : bool,optional
                Whether to return a deep copy of the original object. Default is `copy = False`.

        Returns
        -------
        obj
                Either one of the following:

                * whenever possible do not copy data, if `copy = False`.
                * explicitly copy all possible data, if `copy = True`.

        Examples
        --------
        >>> H_dia=H.as_sparse_format(static_fmt="csr",dynamic_fmt={(func,func_args):"dia"})


        """
        dynamic = [[M, func] for func, M in iteritems(self.dynamic)]
        return hamiltonian(
            [self.static],
            dynamic,
            basis=self._basis,
            dtype=self._dtype,
            static_fmt=static_fmt,
            dynamic_fmt=dynamic_fmt,
            copy=copy,
        )

    ### algebra operations

    def transpose(self, copy=False):
        """Transposes `hamiltonian` operator.

        Notes
        -----
        This function does NOT conjugate the operator.

        Returns
        -------
        :obj:`hamiltonian`
                :math:`H_{ij}\\mapsto H_{ji}`

        Examples
        --------

        >>> H_tran = H.transpose()

        """
        dynamic = [[M.T, func] for func, M in iteritems(self.dynamic)]
        return hamiltonian(
            [self.static.T], dynamic, basis=self._basis, dtype=self._dtype, copy=copy
        )

    def conjugate(self):
        """Conjugates `hamiltonian` operator.

        Notes
        -----
        This function does NOT transpose the operator.

        Returns
        -------
        :obj:`hamiltonian`
                :math:`H_{ij}\\mapsto H_{ij}^*`

        Examples
        --------

        >>> H_conj = H.conjugate()

        """
        dynamic = [[M.conj(), func.conj()] for func, M in iteritems(self.dynamic)]
        return hamiltonian(
            [self.static.conj()], dynamic, basis=self._basis, dtype=self._dtype
        )

    def conj(self):
        """Same functionality as :func:`conjugate`."""
        return self.conjugate()

    def getH(self, copy=False):
        """Calculates hermitian conjugate of `hamiltonian` operator.

        Parameters
        ----------
        copy : bool, optional
                Whether to return a deep copy of the original object. Default is `copy = False`.

        Returns
        -------
        :obj:`hamiltonian`
                :math:`H_{ij}\\mapsto H_{ij}^*`

        Examples
        --------

        >>> H_herm = H.getH()

        """
        return self.conj().transpose(copy=copy)

    ### lin-alg operations

    def diagonal(self, time=0):
        """Calculates diagonal of `hamiltonian` operator at time `time`.

        Parameters
        ----------
        time : float, optional
                Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.

        Returns
        -------
        numpy.ndarray
                Diagonal part of operator :math:`H(t=\\texttt{time})`.

        Examples
        --------

        >>> H_diag = H.diagonal(time=0.0)

        """
        if self.Ns <= 0:
            return 0
        if _np.array(time).ndim > 0:
            raise TypeError("expecting scalar argument for time")

        diagonal = self._static.diagonal()
        for func, Hd in iteritems(self._dynamic):
            diagonal += Hd.diagonal() * func(time)

        return diagonal

    def trace(self, time=0):
        """Calculates trace of `hamiltonian` operator at time `time`.

        Parameters
        ----------
        time : float, optional
                Time to evalute the time-dependent part of the operator at (if existent). Default is `time = 0.0`.

        Returns
        -------
        float
                Trace of operator :math:`\\sum_{j=1}^{Ns} H_{jj}(t=\\texttt{time})`.

        Examples
        --------

        >>> H_tr = H.tr(time=0.0)

        """
        if self.Ns <= 0:
            return 0
        if _np.array(time).ndim > 0:
            raise TypeError("expecting scalar argument for time")

        trace = self._static.diagonal().sum()
        for func, Hd in iteritems(self._dynamic):
            trace += Hd.diagonal().sum() * func(time)

        return trace

    def astype(self, dtype, copy=False, casting="unsafe"):
        """Changes data type of `hamiltonian` object.

        Parameters
        ----------
        dtype : 'type'
                The data type (e.g. numpy.float64) to cast the Hamiltonian with.

        Returns
        -------
        `hamiltonian`
                Operator with altered data type.

        Examples
        --------
        `hamiltonian`
                Operator with altered data type.

        >>> H_cpx=H.astype(np.complex128)

        """

        if dtype not in supported_dtypes:
            raise TypeError("hamiltonian does not support type: " + str(dtype))

        static = self.static.astype(dtype, copy=copy, casting=casting)
        dynamic = [
            [M.astype(dtype, copy=copy, casting=casting), func]
            for func, M in iteritems(self.dynamic)
        ]
        return hamiltonian(
            [static], dynamic, basis=self._basis, dtype=dtype, copy=False
        )

    def copy(self):
        """Returns a copy of `hamiltonian` object."""
        dynamic = [[M, func] for func, M in iteritems(self.dynamic)]
        return hamiltonian(
            [self.static], dynamic, basis=self._basis, dtype=self._dtype, copy=True
        )

    ###################
    # special methods #
    ###################

    def __getitem__(self, key):
        if len(key) != 3:
            raise IndexError(
                "invalid number of indices, hamiltonian must be indexed with three indices [time,row,col]."
            )
        try:
            times = iter(key[0])
            iterate = True
        except TypeError:
            time = key[0]
            iterate = False

        key = tuple(key[1:])
        if iterate:
            ME = []
            if self.is_dense:
                for t in times:
                    ME.append(self.todense(time=t)[key])
            else:
                for t in times:
                    ME.append(self.tocsr(time=t)[key])

            ME = tuple(ME)
        else:
            ME = self.tocsr(time=time)[key]

        return ME

    def __str__(self):
        string = "static mat: \n{0}\n\n\ndynamic:\n".format(self._static.__str__())
        for i, (func, Hd) in enumerate(iteritems(self._dynamic)):
            h_str = Hd.__str__()
            func_str = func.__str__()

            string += "{0}) func: {2}, mat: \n{1} \n".format(i, h_str, func_str)

        return string

    def __repr__(self):
        string = "<quspin.operators.hamiltonian:\nstatic mat: {0}\ndynamic:".format(
            self._static.__repr__()
        )
        for i, (func, Hd) in enumerate(iteritems(self._dynamic)):
            h_str = Hd.__repr__()
            func_str = func.__str__()

            string += "\n {0}) func: {2}, mat: {1} ".format(i, h_str, func_str)
        string = string + ">"

        return string

    def __neg__(self):  # -self
        dynamic = [[-M, func] for func, M in iteritems(self.dynamic)]
        return hamiltonian(
            [-self.static], dynamic, basis=self._basis, dtype=self._dtype
        )

    def __call__(self, time=0):  # self(time)
        """Return hamiltonian as a sparse or dense matrix at specific time

        Parameters
        ----------
        time: float
                time to evaluate dynamic parts at.

        Returns
        -------
        if `is_dense` is True:
                `numpy.ndarray`
        else
                `scipy.csr_matrix`

        Examples
        --------
        >>> H_t = H(time)

        """
        if self.is_dense:
            return self.toarray(time)
        else:
            return self.tocsr(time)

    ##################################
    # symbolic arithmetic operations #
    # currently only have +,-,* like #
    # operators implimented.		 #
    ##################################

    def __pow__(self, power):  # self ** power
        if type(power) is not int:
            raise TypeError("hamiltonian can only be raised to integer power.")

        return reduce(mul, (self for i in range(power)))

    def __mul__(self, other):  # self * other
        if ishamiltonian(other):
            return self._mul_hamiltonian(other)

        elif _sp.issparse(other):
            self._mat_checks(other, casting="unsafe")
            return self._mul_sparse(other)

        elif _np.isscalar(other):
            return self._mul_scalar(other)

        elif other.__class__ == _np.ndarray:
            self._mat_checks(other, casting="unsafe")
            return self._mul_dense(other)

        elif other.__class__ == _np.matrix:
            self._mat_checks(other, casting="unsafe")
            return self._mul_dense(other)

        else:
            other = _np.asanyarray(other)
            self._mat_checks(other, casting="unsafe")
            return self._mul_dense(other)

    def __rmul__(self, other):  # other * self
        if ishamiltonian(other):
            self._mat_checks(other, casting="unsafe")
            return self._rmul_hamiltonian(other)

        elif _sp.issparse(other):
            self._mat_checks(other, casting="unsafe")
            return self._rmul_sparse(other)

        elif _np.isscalar(other):

            return self._mul_scalar(other)

        elif other.__class__ == _np.ndarray:
            self._mat_checks(other, casting="unsafe")
            return self._rmul_dense(other)

        elif other.__class__ == _np.matrix:
            self._mat_checks(other, casting="unsafe")
            return self._rmul_dense(other)

        else:
            other = _np.asanyarray(other)
            self._mat_checks(other, casting="unsafe")
            return self._rmul_dense(other)

    def __imul__(self, other):  # self *= other
        if ishamiltonian(other):
            self._mat_checks(other)
            return self._imul_hamiltonian(other)

        elif _sp.issparse(other):
            self._mat_checks(other)
            return self._imul_sparse(other)

        elif _np.isscalar(other):
            return self._imul_scalar(other)

        elif other.__class__ == _np.ndarray:
            self._mat_checks(other)
            return self._imul_dense(other)

        elif other.__class__ == _np.matrix:
            self._mat_checks(other)
            return self._imul_dense(other)

        else:
            other = _np.asanyarray(other)
            self._mat_checks(other)
            return self._imul_dense(other)

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, other):  # self / other
        if ishamiltonian(other):
            return NotImplemented

        elif _sp.issparse(other):
            return NotImplemented

        elif _np.isscalar(other):
            return self._mul_scalar(1.0 / other)

        elif other.__class__ == _np.ndarray:
            return NotImplemented

        elif other.__class__ == _np.matrix:
            return NotImplemented

        else:
            return NotImplemented

    def __rdiv__(self, other):  # other / self
        return NotImplemented

    def __idiv__(self, other):  # self *= other
        if ishamiltonian(other):
            return NotImplemented

        elif _sp.issparse(other):
            return NotImplemented

        elif _np.isscalar(other):
            return self._imul_scalar(1.0 / other)

        elif other.__class__ == _np.ndarray:
            return NotImplemented

        elif other.__class__ == _np.matrix:
            return NotImplemented

        else:
            return NotImplemented

    def __add__(self, other):  # self + other
        if ishamiltonian(other):
            self._mat_checks(other, casting="unsafe")
            return self._add_hamiltonian(other)

        elif _sp.issparse(other):
            self._mat_checks(other, casting="unsafe")
            return self._add_sparse(other)

        elif _np.isscalar(other):
            if other == 0.0:
                return self
            else:
                raise NotImplementedError(
                    "hamiltonian does not support addition by nonzero scalar"
                )

        elif other.__class__ == _np.ndarray:
            self._mat_checks(other, casting="unsafe")
            return self._add_dense(other)

        elif other.__class__ == _np.matrix:
            self._mat_checks(other, casting="unsafe")
            return self._add_dense(other)

        else:
            other = _np.asanyarray(other)
            self._mat_checks(other, casting="unsafe")
            return self._add_dense(other)

    def __radd__(self, other):  # other + self
        return self.__add__(other)

    def __iadd__(self, other):  # self += other
        if ishamiltonian(other):
            self._mat_checks(other)
            return self._iadd_hamiltonian(other)

        elif _sp.issparse(other):
            self._mat_checks(other)
            return self._iadd_sparse(other)

        elif _np.isscalar(other):
            if other == 0.0:
                return self
            else:
                raise NotImplementedError(
                    "hamiltonian does not support addition by nonzero scalar"
                )

        elif other.__class__ == _np.ndarray:
            self._mat_checks(other)
            return self._iadd_dense(other)

        else:
            other = _np.asanyarray(other)
            self._mat_checks(other)
            return self._iadd_dense(other)

    def __sub__(self, other):  # self - other
        if ishamiltonian(other):
            self._mat_checks(other, casting="unsafe")
            return self._sub_hamiltonian(other)

        elif _sp.issparse(other):
            self._mat_checks(other, casting="unsafe")
            return self._sub_sparse(other)

        elif _np.isscalar(other):
            if other == 0.0:
                return self
            else:
                raise NotImplementedError(
                    "hamiltonian does not support subtraction by nonzero scalar"
                )

        elif other.__class__ == _np.ndarray:
            self._mat_checks(other, casting="unsafe")
            return self._sub_dense(other)

        else:
            other = _np.asanyarray(other)
            self._mat_checks(other, casting="unsafe")
            return self._sub_dense(other)

    def __rsub__(self, other):  # other - self
        # NOTE: because we use signed types this is possble
        return self.__sub__(other).__neg__()

    def __isub__(self, other):  # self -= other

        if ishamiltonian(other):
            self._mat_checks(other)
            return self._isub_hamiltonian(other)

        elif _sp.issparse(other):
            self._mat_checks(other)
            return self._isub_sparse(other)

        elif _np.isscalar(other):
            if other == 0.0:
                return self
            else:
                raise NotImplementedError(
                    "hamiltonian does not support subtraction by nonzero scalar"
                )

        elif other.__class__ == _np.ndarray:
            self._mat_checks(other)
            return self._isub_dense(other)

        else:
            other = _np.asanyarray(other)
            self._mat_checks(other)
            return self._isub_dense(other)

    ##########################################################################################
    ##########################################################################################
    # below all of the arithmetic functions are implimented for various combination of types #
    ##########################################################################################
    ##########################################################################################

    # checks
    def _mat_checks(self, other, casting="same_kind"):
        try:
            if other.shape != self._shape:  # only accepts square matricies
                raise ValueError("shapes do not match")
            if not _np.can_cast(other.dtype, self._dtype, casting=casting):
                raise ValueError("cannot cast types")
        except AttributeError:
            if other._shape != self._shape:  # only accepts square matricies
                raise ValueError("shapes do not match")
            if not _np.can_cast(other.dtype, self._dtype, casting=casting):
                raise ValueError("cannot cast types")

    ##########################
    # hamiltonian operations #
    ##########################

    def _add_hamiltonian(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)

        new._is_dense = new._is_dense or other._is_dense

        try:
            new._static += other._static
        except NotImplementedError:
            new._static = new._static + other._static

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        for func, Hd in iteritems(other._dynamic):
            if func in new._dynamic:
                try:
                    new._dynamic[func] += Hd
                except NotImplementedError:
                    new._dynamic[func] = new._dynamic[func] + Hd

                if _check_almost_zero(new._dynamic[func]):
                    new._dynamic.pop(func)
            else:
                new._dynamic[func] = Hd

        new.check_is_dense()
        new._get_matvecs()
        return new

    def _iadd_hamiltonian(self, other):
        self._is_dense = self._is_dense or other._is_dense

        try:
            self._static += other._static
        except NotImplementedError:
            self._static = self._static + other._static

        if _check_almost_zero(self._static):
            self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)

        for func, Hd in iteritems(other._dynamic):
            if func in self._dynamic:
                try:
                    self._dynamic[func] += Hd
                except NotImplementedError:
                    self._dynamic[func] = self._dynamic[func] + Hd

                try:
                    self._dynamic[func].sum_duplicates()
                    self._dynamic[func].eliminate_zeros()
                except:
                    pass

                if _check_almost_zero(self._dynamic[func]):
                    self._dynamic.pop(func)

            else:
                self._dynamic[func] = Hd

        self.check_is_dense()
        self._get_matvecs()

        return self

    def _sub_hamiltonian(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)

        new._is_dense = new._is_dense or other._is_dense

        try:
            new._static -= other._static
        except NotImplementedError:
            new._static = new._static - other._static

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        for func, Hd in iteritems(other._dynamic):
            if func in new._dynamic:
                try:
                    new._dynamic[func] -= Hd
                except NotImplementedError:
                    new._dynamic[func] = new._dynamic[func] - Hd

                if _check_almost_zero(new._dynamic[func]):
                    new._dynamic.pop(func)

            else:
                new._dynamic[func] = -Hd

        new.check_is_dense()
        new._get_matvecs()

        return new

    def _isub_hamiltonian(self, other):
        self._is_dense = self._is_dense or other._is_dense

        try:
            self._static -= other._static
        except NotImplementedError:
            self._static = self._static - other._static

        if _check_almost_zero(self._static):
            self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)

        for func, Hd in iteritems(other._dynamic):
            if func in self._dynamic:
                try:
                    self._dynamic[func] -= Hd
                except NotImplementedError:
                    self._dynamic[func] = self._dynamic[func] - Hd

                if _check_almost_zero(self._dynamic[func]):
                    self._dynamic.pop(func)

            else:
                self._dynamic[func] = -Hd

        self.check_is_dense()
        self._get_matvecs()

        return self

    def _mul_hamiltonian(self, other):
        if self.dynamic and other.dynamic:
            new = self.astype(self._dtype)
            return new._imul_hamiltonian(other)
        elif self.dynamic:
            return self._mul_sparse(other.static)
        elif other.dynamic:
            return other._rmul_sparse(self.static)
        else:
            return self._mul_sparse(other.static)

    def _rmul_hamiltonian(self, other):
        if self.dynamic and other.dynamic:
            new = other.astype(self._dtype)
            return (new.T._imul_hamiltonian(self.T)).T  # lazy implementation
        elif self.dynamic:
            return self._rmul_sparse(other.static)
        elif other.dynamic:
            return other._mul_sparse(self.static)
        else:
            return self._rmul_sparse(other.static)

    def _imul_hamiltonian(self, other):
        if self.dynamic and other.dynamic:
            self._is_dense = self._is_dense or other._is_dense
            new_dynamic_ops = {}
            # create new dynamic operators coming from

            # self.static * other.static
            if _sp.issparse(self.static):
                new_static_op = self.static.dot(other._static)
            elif _sp.issparse(other._static):
                new_static_op = self.static * other._static
            else:
                new_static_op = _np.matmul(self.static, other._static)

            # self.static * other.dynamic
            for func, Hd in iteritems(other._dynamic):
                if _sp.issparse(self.static):
                    Hmul = self.static.dot(Hd)
                elif _sp.issparse(Hd):
                    Hmul = self.static * Hd
                else:
                    Hmul = _np.matmul(self.static, Hd)

                if not _check_almost_zero(Hmul):
                    new_dynamic_ops[func] = Hmul

            # self.dynamic * other.static
            for func, Hd in iteritems(self._dynamic):
                if _sp.issparse(Hd):
                    Hmul = Hd.dot(other._static)
                elif _sp.issparse(other._static):
                    Hmul = Hd * other._static
                else:
                    Hmul = _np.matmul(Hd, other._static)

                if func in new_dynamic_ops:
                    try:
                        new_dynamic_ops[func] += Hmul
                    except NotImplementedError:
                        new_dynamic_ops[func] = new_dynamic_ops[func] + Hmul

                    if _check_almost_zero(new_dynamic_ops[func]):
                        new_dynamic_ops.pop(func)

                else:
                    if not _check_almost_zero(Hmul):
                        new_dynamic_ops[func] = Hmul

            # self.dynamic * other.dynamic
            for func1, H1 in iteritems(self._dynamic):
                for func2, H2 in iteritems(other._dynamic):

                    if _sp.issparse(H1):
                        H12 = H1.dot(H2)
                    elif _sp.issparse(H2):
                        H12 = H1 * H2
                    else:
                        H12 = _np.matmul(H1, H2)

                    func12 = func1 * func2

                    if func12 in new_dynamic_ops:
                        try:
                            new_dynamic_ops[func12] += H12
                        except NotImplementedError:
                            new_dynamic_ops[func12] = new_dynamic_ops[func12] + H12

                        if _check_almost_zero(new_dynamic_ops[func12]):
                            new_dynamic_ops.pop(func12)
                    else:
                        if not _check_almost_zero(H12):
                            new_dynamic_ops[func12] = H12

            self._static = new_static_op
            self._dynamic = new_dynamic_ops
            self._dtype = new_static_op.dtype
            self._get_matvecs()
            return self
        elif self.dynamic:
            return self._imul_sparse(other.static)
        elif other.dynamic:
            return (other.T._imul_sparse(self.static.T)).T
        else:
            return self._imul_sparse(other.static)

    #####################
    # sparse operations #
    #####################

    def _add_sparse(self, other):

        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)

        try:
            new._static += other
        except NotImplementedError:
            new._static = new._static + other

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        new.check_is_dense()
        new._get_matvecs()

        return new

    def _iadd_sparse(self, other):

        try:
            self._static += other
        except NotImplementedError:
            self._static = self._static + other

        if _check_almost_zero(self._static):
            self._static = _sp.csr_matrix(self._shape, dtype=self._dtype)

        self.check_is_dense()
        self._get_matvecs()

        return self

    def _sub_sparse(self, other):

        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)

        try:
            new._static -= other
        except NotImplementedError:
            new._static = new._static - other

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        new.check_is_dense()
        new._get_matvecs()
        return new

    def _isub_sparse(self, other):

        try:
            self._static -= other
        except NotImplementedError:
            self._static = self._static - other

        if _check_almost_zero(self._static):
            self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)

        self.check_is_dense()
        self._get_matvecs()

        return self

    def _mul_sparse(self, other):

        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)

        new._static = new._static * other

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        for func in list(new._dynamic):
            new._dynamic[func] = new._dynamic[func] * other

            if _check_almost_zero(new._dynamic[func]):
                new._dynamic.pop(func)

        new.check_is_dense()
        new._get_matvecs()
        return new

    def _rmul_sparse(self, other):
        # Auxellery function to calculate the right-side multipication with another sparse matrix.

        # find resultant type from product
        result_dtype = _np.result_type(self._dtype, other.dtype)
        # create a copy of the hamiltonian object with the previous dtype
        new = self.astype(result_dtype, copy=True)

        # proform multiplication on all matricies of the new hamiltonian object.

        new._static = other * new._static

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        for func in list(new._dynamic):
            new._dynamic[func] = other.dot(new._dynamic[func])

            try:
                new._dynamic[func].sum_duplicates()
                new._dynamic[func].eliminate_zeros()
            except:
                pass

            if _check_almost_zero(new._dynamic[func]):
                new._dynamic.pop(func)

        new.check_is_dense()
        new._get_matvecs()
        return new

    def _imul_sparse(self, other):

        self._static = self._static * other

        if _check_almost_zero(self._static):
            self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)

        for func in list(self._dynamic):
            self._dynamic[func] = other.dot(self._dynamic[func])

            try:
                self._dynamic[func].sum_duplicates()
                self._dynamic[func].eliminate_zeros()
            except:
                pass

            if _check_almost_zero(self._dynamic[func]):
                self._dynamic.pop(func)

        self.check_is_dense()
        self._get_matvecs()

        return self

    #####################
    # scalar operations #
    #####################

    def _mul_scalar(self, other):
        result_dtype = _np.result_type(self._dtype, other)
        new = self.astype(result_dtype, copy=True)

        try:
            new._static *= other
        except NotImplementedError:
            new._static = new._static * other

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        for func in list(new._dynamic):
            try:
                new._dynamic[func] *= other
            except NotImplementedError:
                new._dynamic[func] = new._dynamic[func] * other

            if _check_almost_zero(new._dynamic[func]):
                new._dynamic.pop(func)

        new.check_is_dense()
        new._get_matvecs()
        return new

    def _imul_scalar(self, other):
        if not _np.can_cast(_np.min_scalar_type(other), self._dtype, casting="same_kind"):
            raise TypeError("cannot cast types")

        try:
            self._static *= other
        except NotImplementedError:
            self._static = self._static * other

        if _check_almost_zero(self._static):
            self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)

        for func in list(self._dynamic):
            try:
                self._dynamic[func] *= other
            except NotImplementedError:
                self._dynamic[func] = self._dynamic[func] * other

            if _check_almost_zero(self._dynamic[func]):
                self._dynamic.pop(func)

        self.check_is_dense()
        self._get_matvecs()

        return self

    ####################
    # dense operations #
    ####################

    def _add_dense(self, other):

        result_dtype = _np.result_type(self._dtype, other.dtype)

        if result_dtype not in supported_dtypes:
            return NotImplemented

        new = self.astype(result_dtype, copy=True)

        if not self._is_dense:
            self._is_dense = True
            warnings.warn(
                "Mixing dense objects will cast internal matrices to dense.",
                HamiltonianEfficiencyWarning,
                stacklevel=3,
            )

        try:
            new._static += other
        except:
            new._static = new._static + other

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        new.check_is_dense()
        new._get_matvecs()
        return new

    def _iadd_dense(self, other):

        if not self._is_dense:
            self._is_dense = True
            warnings.warn(
                "Mixing dense objects will cast internal matrices to dense.",
                HamiltonianEfficiencyWarning,
                stacklevel=3,
            )

        try:
            self._static += other
        except NotImplementedError:
            self._static = self._static + other

        if _check_almost_zero(self._static):
            self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)

        self.check_is_dense()
        self._get_matvecs()

        return self

    def _sub_dense(self, other):

        result_dtype = _np.result_type(self._dtype, other.dtype)

        if result_dtype not in supported_dtypes:
            return NotImplemented

        new = self.astype(result_dtype, copy=True)

        if not self._is_dense:
            self._is_dense = True
            warnings.warn(
                "Mixing dense objects will cast internal matrices to dense.",
                HamiltonianEfficiencyWarning,
                stacklevel=3,
            )

        try:
            new._static -= other
        except:
            new._static = new._static - other

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        new.check_is_dense()
        new._get_matvecs()

        return new

    def _isub_dense(self, other):

        if not self._is_dense:
            self._is_dense = True
            warnings.warn(
                "Mixing dense objects will cast internal matrices to dense.",
                HamiltonianEfficiencyWarning,
                stacklevel=3,
            )

        try:
            self._static -= other
        except:
            self._static = self._static - other

        if _check_almost_zero(self._static):
            self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)

        self.check_is_dense()
        self._get_matvecs()

        return self

    def _mul_dense(self, other):

        result_dtype = _np.result_type(self._dtype, other.dtype)

        if result_dtype not in supported_dtypes:
            return NotImplemented

        new = self.astype(result_dtype, copy=True)

        if not self._is_dense:
            self._is_dense = True
            warnings.warn(
                "Mixing dense objects will cast internal matrices to dense.",
                HamiltonianEfficiencyWarning,
                stacklevel=3,
            )

        new._static = _np.asarray(new._static.dot(other))

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        for func in list(new._dynamic):
            new._dynamic[func] = _np.asarray(new._dynamic[func].dot(other))

            if _check_almost_zero(new._dynamic[func]):
                new._dynamic.pop(func)

        new.check_is_dense()
        new._get_matvecs()

        return new

    def _rmul_dense(self, other):

        result_dtype = _np.result_type(self._dtype, other.dtype)

        if result_dtype not in supported_dtypes:
            return NotImplemented

        new = self.astype(result_dtype, copy=True)

        if not self._is_dense:
            self._is_dense = True
            warnings.warn(
                "Mixing dense objects will cast internal matrices to dense.",
                HamiltonianEfficiencyWarning,
                stacklevel=3,
            )

        if _sp.issparse(new._static):
            new._static = _np.asarray(other * new._static)
        else:
            new._static = _np.asarray(other.dot(new._static))

        if _check_almost_zero(new._static):
            new._static = _sp.dia_matrix(new._shape, dtype=new._dtype)

        for func in list(new._dynamic):
            if _sp.issparse(new._dynamic[func]):
                new._dynamic[func] = _np.asarray(other * new._dynamic[func])
            else:
                new._dynamic[func] = _np.asarray(other.dot(new._dynamic[func]))

            if _check_almost_zero(new._dynamic[func]):
                new._dynamic.pop(func)

        new.check_is_dense()
        new._get_matvecs()

        return new

    def _imul_dense(self, other):

        if not self._is_dense:
            self._is_dense = True
            warnings.warn(
                "Mixing dense objects will cast internal matrices to dense.",
                HamiltonianEfficiencyWarning,
                stacklevel=3,
            )

        self._static = _np.asarray(self._static.dot(other))

        if _check_almost_zero(self._static):
            self._static = _sp.dia_matrix(self._shape, dtype=self._dtype)

        for func in list(self._dynamic):
            self._dynamic[func] = _np.asarray(self._dynamic[func].dot(other))

            if _check_almost_zero(self._dynamic[func]):
                self._dynamic.pop(func)

        self.check_is_dense()
        self._get_matvecs()

        return self

    def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
        # """Method for compatibility with NumPy <= 1.11 ufuncs and dot
        # functions.
        # """
        out = kwargs.get("out")

        if out is not None:
            raise ValueError(
                "quspin hamiltonian class does not support 'out' for numpy.multiply, numpy.dot, or numpy.matmul."
            )

        if (func == _np.dot) or (func == _np.multiply) or (func == _np.matmul):
            if pos == 0:
                return self.__mul__(inputs[1])
            if pos == 1:
                return self.__rmul__(inputs[0])
        elif func == _np.subtract:
            if pos == 0:
                return self.__sub__(inputs[1])
            if pos == 1:
                return self.__rsub__(inputs[0])
        elif func == _np.add:
            if pos == 0:
                return self.__add__(inputs[1])
            if pos == 1:
                return self.__radd__(inputs[0])

        return NotImplemented

    def __array_ufunc__(self, func, method, *inputs, **kwargs):
        # """Method for compatibility with NumPy >= 1.13 ufuncs and dot
        # functions.
        # """
        out = kwargs.get("out")

        if out is not None:
            raise ValueError(
                "quspin hamiltonian class does not support 'out' for numpy.multiply or numpy.dot."
            )

        if (func == _np.dot) or (func == _np.multiply):

            if ishamiltonian(inputs[0]):
                return inputs[0].__mul__(inputs[1])
            elif ishamiltonian(inputs[1]):
                return inputs[1].__rmul__(inputs[0])
        elif func == _np.subtract:
            if ishamiltonian(inputs[0]):
                return inputs[0].__sub__(inputs[1])
            elif ishamiltonian(inputs[1]):
                return inputs[1].__rsub__(inputs[0])
        elif func == _np.add:
            if ishamiltonian(inputs[0]):
                return inputs[0].__add__(inputs[1])
            elif ishamiltonian(inputs[1]):
                return inputs[1].__radd__(inputs[0])

        return NotImplemented


def ishamiltonian(obj):
    """Checks if instance is object of `hamiltonian` class.

    Parameters
    ----------
    obj :
            Arbitraty python object.

    Returns
    -------
    bool
            Can be either of the following:

            * `True`: `obj` is an instance of `hamiltonian` class.
            * `False`: `obj` is NOT an instance of `hamiltonian` class.

    """
    return isinstance(obj, hamiltonian)
