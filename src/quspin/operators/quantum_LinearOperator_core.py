from quspin.operators.hamiltonian_core import ishamiltonian
from quspin.operators.hamiltonian_core import _check_static
from quspin.operators.hamiltonian_core import supported_dtypes
from quspin.operators.hamiltonian_core import hamiltonian

from quspin.operators._make_hamiltonian import _consolidate_static

from quspin.basis import spin_basis_1d as _default_basis
from quspin.basis.base import _is_diagonal, _update_diag
from quspin.basis import isbasis as _isbasis


# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.sparse as _sp
import numpy as _np

from scipy.sparse.linalg import LinearOperator

from six import iteritems

__all__ = ["quantum_LinearOperator", "isquantum_LinearOperator"]


class quantum_LinearOperator(LinearOperator):
    """Applies a quantum operator directly onto a state, without constructing the operator matrix.

    The `quantum_LinearOperator` class uses the `basis.Op()` function to calculate the matrix vector product on the
    fly, greatly reducing the amount of memory needed for a calculation at the cost of speed.

    This object is useful for doing large scale Lanczos calculations using the `eigsh` method.

    Notes
    -----
    The class does NOT yet support time-dependent operators.

    Examples
    --------

    The following example shows how to construct and use `quantum_LinearOperator` objects.

    .. literalinclude:: ../../doc_examples/quantum_LinearOperator-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(
        self,
        static_list,
        N=None,
        basis=None,
        diagonal=None,
        check_symm=True,
        check_herm=True,
        check_pcon=True,
        dtype=_np.complex128,
        copy=False,
        **basis_args,
    ):
        """Intializes the `quantum_LinearOperator` object.

        Parameters
        ----------

        static_list : list
                Contains list of objects to calculate the static part of a `quantum_LinearOperator` operator. Same as
                the `static` argument of the `quantum_operator` class. The format goes like:

                >>> static_list=[[opstr_1,[indx_11,...,indx_1m]],matrix_2,...]

        N : int, optional
                number of sites to create the default spin basis with.
        basis : :obj:`basis`, optional
                basis object to construct quantum operator with.
        diagonal : array_like
                array containing diagonal matrix elements precalculated by other means.
        dtype : 'type'
                Data type (e.g. numpy.float64) to construct the operator with.
        check_symm : bool, optional
                Enable/Disable symmetry check on `static_list` and `dynamic_list`.
        check_herm : bool, optional
                Enable/Disable hermiticity check on `static_list` and `dynamic_list`.
        check_pcon : bool, optional
                Enable/Disable particle conservation check on `static_list` and `dynamic_list`.
        basis_args : dict
                Optional additional arguments to pass to the `basis` class, if not already using a `basis` object
                to create the operator.

        """

        if type(static_list) in [list, tuple]:
            for ele in static_list:
                if not _check_static(ele):
                    raise ValueError(
                        "quantum_LinearOperator only supports operator string representations."
                    )
        else:
            raise TypeError(
                "expecting list/tuple of lists/tuples containing opstr and list of indx"
            )

        if dtype not in supported_dtypes:
            raise TypeError("hamiltonian does not support type: " + str(dtype))
        else:
            self._dtype = dtype

        if N == []:
            raise ValueError(
                "second argument of `quantum_LinearOperator()` canNOT be an empty list."
            )
        elif type(N) is int and basis is None:
            self._basis = _default_basis(N, **basis_args)
        elif N is None and _isbasis(basis):
            self._basis = basis
        else:
            raise ValueError("expecting integer for N or basis object for basis.")

        self._unique_me = self.basis._unique_me
        self._transposed = False
        self._conjugated = False
        self._scale = _np.array(1.0, dtype=dtype)
        self._dtype = dtype
        self._ndim = 2
        self._shape = (self._basis.Ns, self._basis.Ns)

        if check_herm:
            self.basis.check_hermitian(static_list, [])

        if check_symm:
            self.basis.check_symm(static_list, [])

        if check_pcon:
            self.basis.check_pcon(static_list, [])

        if diagonal is not None:
            self.set_diagonal(diagonal, copy=copy)
        else:
            self._diagonal = None

        self._public_static_list = list(static_list)
        static_list = _consolidate_static(static_list)

        self._static_list = []
        for opstr, indx, J in static_list:
            ME, row, col = self.basis.Op(opstr, indx, J, self._dtype)
            if _is_diagonal(row, col):
                if self._diagonal is None:
                    self._diagonal = _np.zeros((self.Ns,), dtype=ME.dtype)

                _update_diag(self._diagonal, row, ME)

            else:
                self._static_list.append((opstr, indx, J))

    @property
    def shape(self):
        """tuple: shape of linear operator."""
        return self._shape

    @property
    def basis(self):
        """:obj:`basis`: basis used to build the `hamiltonian` object.

        Defaults to `None` if operator has  no basis (i.e. was created externally and passed as a precalculated array).
        """
        return self._basis

    @property
    def ndim(self):
        """int: number of dimensions, always equal to 2."""
        return self._ndim

    @property
    def static_list(self):
        """list: operator list used to create this object."""
        return self._public_static_list

    @property
    def get_shape(self):
        """tuple: shape of the `quantum_LinearOperator` object, always equal to `(Ns,Ns)`."""
        return self._shape

    @property
    def Ns(self):
        """int: number of states in the (symmetry-reduced) Hilbert space spanned by `basis`."""
        return self._shape[0]

    @property
    def dtype(self):
        """type: data type of `quantum_LinearOperator` object."""
        return _np.dtype(self._dtype)

    @property
    def T(self):
        """:obj:`quantum_LinearOperator`: transposes the operator matrix, :math:`H_{ij}\\mapsto H_{ji}`."""
        return self.transpose(copy=False)

    @property
    def H(self):
        """:obj:`quantum_LinearOperator`: transposes and conjugates the operator matrix, :math:`H_{ij}\\mapsto H_{ji}^*`."""
        return self.getH(copy=False)

    @property
    def diagonal(self):
        """numpy.ndarray: static diagonal part of the linear operator."""
        if self._diagonal is not None:
            diagonal_view = self._diagonal[:]
            diagonal_view.setflags(write=0, uic=0)
            return diagonal_view
        else:
            return None

    def set_diagonal(self, diagonal, copy=True):
        """Sets the diagonal part of the quantum_LinearOperator.

        Parameters
        ----------
        diagonal: array_like
                array_like object containing the new diagonal.

        """
        if diagonal.__class__ != _np.ndarray:
            diagonal = _np.asanyarray(diagonal)
        if diagonal.ndim != 1:
            raise ValueError("diagonal must be 1-d array.")
        if diagonal.shape[0] != self.Ns:
            raise ValueError("length of diagonal must be equal to dimension of matrix")

        if copy:
            self._diagonal = diagonal.copy()
        else:
            self._diagonal = diagonal

    ### state manipulation/observable routines

    # def dot(self,other):
    # 	"""Matrix-vector multiplication of `quantum_LinearOperator` operator, with state `V`.

    # 	.. math::
    # 		H|V\\rangle

    # 	Parameters
    # 	----------
    # 	other : numpy.ndarray
    # 		Vector (quantums tate) to multiply the `quantum_LinearOperator` operator with.

    # 	Returns
    # 	-------
    # 	numpy.ndarray
    # 		Vector corresponding to the `hamiltonian` operator applied on the state `V`.

    # 	Examples
    # 	--------
    # 	>>> B = H.dot(A,check=True)

    # 	corresponds to :math:`B = HA`.

    # 	"""
    # 	return self.__mul__(other)

    # def rdot(self,other):
    # 	"""Vector-matrix multiplication of `quantum_LinearOperator` operator, with state `V`.

    # 	.. math::
    # 		\\langle V|H

    # 	Parameters
    # 	----------
    # 	other : numpy.ndarray
    # 		Vector (quantums tate) to multiply the `quantum_LinearOperator` operator with.

    # 	Returns
    # 	-------
    # 	numpy.ndarray
    # 		Vector corresponding to the `hamiltonian` operator applied on the state `V`.

    # 	Examples
    # 	--------
    # 	>>> B = H.dot(A,check=True)

    # 	corresponds to :math:`B = AH`.

    # 	"""
    # 	return self.__rmul__(other)

    def dot(self, other, out=None, a=1.0):
        """Matrix-vector multiplication of `quantum_LinearOperator` operator, with state `V`.

        .. math::
                aH|V\\rangle

        Parameters
        ----------
        other : numpy.ndarray
                Vector (quantums tate) to multiply the `quantum_LinearOperator` operator with.
        out : array_like, optional
                specify the output array for the the result.
        a : scalar, optional
                scalar to multiply the final product with: :math:`B = aHA`.

        Returns
        -------
        numpy.ndarray
                Vector corresponding to the `hamiltonian` operator applied on the state `V`.

        Examples
        --------
        >>> B = H.dot(A,check=True)

        corresponds to :math:`B = HA`.

        """
        if out is not None:
            other = _np.asanyarray(other)
            out = _np.asanyarray(out)

            result_dtype = _np.result_type(self._dtype, other.dtype)

            if not out.flags["CARRAY"] or (out.dtype, out.shape) != (
                result_dtype,
                other.shape,
            ):
                raise ValueError(
                    "out must be C-congituous writable array \
					with dtype {} and shape {}.".format(
                        result_dtype, out.shape
                    )
                )

            if self.diagonal is not None:
                _np.multiply(other.T, self.diagonal, out=out.T)

            self.basis.inplace_Op(
                other,
                self._static_list,
                self._dtype,
                self._conjugated,
                self._transposed,
                v_out=out,
                a=a,
            )

            return out
        else:
            return a * (self * other)

    def quant_fluct(self, V, enforce_pure=False, check=True, time=0):
        """Calculates the quantum fluctuations (variance) of `quantum_LinearOperator` object in state `V`.

        .. math::
                \\langle V|H^2|V\\rangle - \\langle V|H|V\\rangle^2

        Parameters
        ----------
        V : numpy.ndarray
                Depending on the shape, can be a single state or a collection of pure or mixed states
                [see `enforce_pure`].
        enforce_pure : bool, optional
                Flag to enforce pure expectation value of `V` is a square matrix with multiple pure states
                in the columns.

        Returns
        -------
        float
                Quantum fluctuations of `hamiltonian` operator in state `V`.

        Examples
        --------
        >>> H_fluct = H.quant_fluct(V,time=0,check=True)

        corresponds to :math:`\\left(\\Delta H\\right)^2 = \\langle V|H^2(t=\\texttt{time})|V\\rangle - \\langle V|H(t=\\texttt{time})|V\\rangle^2`.

        """
        from quspin.operators.exp_op_core import isexp_op
        from quspin.operators.hamiltonian_core import ishamiltonian

        if self.Ns <= 0:
            return _np.asarray([])

        if ishamiltonian(V):
            raise TypeError("Can't take expectation value of hamiltonian")

        if isexp_op(V):
            raise TypeError("Can't take expectation value of exp_op")

        # fluctuations =  expctH2 - expctH^2
        kwargs = dict(enforce_pure=enforce_pure)
        V_dot = self.dot(V, check=check)
        expt_value_sq = self._expt_value_core(V, V_dot, **kwargs) ** 2

        if V.shape[0] != V.shape[1] or enforce_pure:
            sq_expt_value = self._expt_value_core(V_dot, V_dot, **kwargs)
        else:
            V_dot = self.dot(V_dot, time=time, check=check)
            sq_expt_value = self._expt_value_core(V, V_dot, **kwargs)

        return sq_expt_value - expt_value_sq

    def expt_value(self, V, enforce_pure=False):
        """Calculates expectation value of `quantum_LinearOperator` object in state `V`.

        .. math::
                \\langle V|H|V\\rangle

        Parameters
        ----------
        V : numpy.ndarray
                Depending on the shape, can be a single state or a collection of pure or mixed states
                [see `enforce_pure` argument of `basis.ent_entropy`].
        enforce_pure : bool, optional
                Flag to enforce pure expectation value of `V` is a square matrix with multiple pure states
                in the columns.

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
        from quspin.operators.hamiltonian_core import ishamiltonian

        if self.Ns <= 0:
            return _np.asarray([])

        if ishamiltonian(V):
            raise TypeError("Can't take expectation value of hamiltonian")

        if isexp_op(V):
            raise TypeError("Can't take expectation value of exp_op")

        V_dot = self.dot(V)
        return self._expt_value_core(V, V_dot, enforce_pure=enforce_pure)

    def _expt_value_core(self, V_left, V_right, enforce_pure=False):
        if _sp.issparse(V_right):
            if V_left.shape[0] != V_left.shape[1] or enforce_pure:  # pure states
                return _np.asscalar((V_left.T.conj().dot(V_right)).toarray())
            else:  # density matrix
                return V_right.diagonal().sum()
        else:
            V_right = _np.asarray(V_right).squeeze()
            if V_right.ndim == 1:  # pure state
                return _np.vdot(V_left, V_right)
            elif (
                V_left.shape[0] != V_left.shape[1] or enforce_pure
            ):  # multiple pure states
                return _np.einsum("ij,ij->j", V_left.conj(), V_right)
            else:  # density matrix
                return V_right.trace()

    def matrix_ele(self, Vl, Vr, diagonal=False):
        """Calculates matrix element of `quantum_LinearOperator` object between states `Vl` and `Vr`.

        .. math::
                \\langle V_l|H|V_r\\rangle

        Notes
        -----
        Taking the conjugate or transpose of the state `Vl` is done automatically.

        Parameters
        ----------
        Vl : numpy.ndarray
                Vector(s)/state(s) to multiple with on left side.
        Vl : numpy.ndarray
                Vector(s)/state(s) to multiple with on right side.
        diagonal : bool, optional
                When set to `True`, returs only diagonal part of expectation value. Default is `diagonal = False`.

        Returns
        -------
        float
                Matrix element of `quantum_operator` quantum_operator between the states `Vl` and `Vr`.

        Examples
        --------
        >>> H_lr = H.expt_value(Vl,Vr,pars=pars,diagonal=False,check=True)

        corresponds to :math:`H_{lr} = \\langle V_l|H(\\lambda=0)|V_r\\rangle`.

        """
        Vr = self.dot(Vr)

        try:
            shape = Vl.shape
        except AttributeError:
            Vl = _np.asanyarray(Vl)
            shape = Vl.shape

        if shape[0] != self._shape[1]:
            raise ValueError(
                "matrix dimension mismatch with shapes: {0} and {1}.".format(
                    Vl.shape, self._shape
                )
            )

        if Vl.ndim > 2:
            raise ValueError("Expecting  0< V.ndim < 3.")

        if _sp.issparse(Vl):
            if diagonal:
                return Vl.T.conj().dot(Vr).diagonal()
            else:
                return Vl.T.conj().dot(Vr)
        else:
            if diagonal:
                return _np.einsum("ij,ij->j", Vl.conj(), Vr)
            else:
                return Vl.T.conj().dot(Vr)

    def _matvec(self, other):

        other = _np.asanyarray(other)
        result_dtype = _np.result_type(self._dtype, other.dtype)

        other = other.astype(result_dtype, copy=False, order="C")
        new_other = _np.zeros_like(other)

        if self._diagonal is not None:
            _np.multiply(other.T, self._diagonal, out=new_other.T)

        self.basis.inplace_Op(
            other,
            self._static_list,
            self._dtype,
            self._conjugated,
            self._transposed,
            v_out=new_other,
        )

        return new_other

    def _rmatvec(self, other):
        return self.T.conj()._matvec(other)

    def _matmat(self, other):
        return self._matvec(other)

    ### Diagonalisation routines

    def eigsh(self, **eigsh_args):
        """Computes SOME eigenvalues and eigenvectors of hermitian `quantum_LinearOperator` operator using SPARSE hermitian methods.

        This function method solves for eigenvalues and eigenvectors, but can only solve for a few of them accurately.
        It calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_, which is a wrapper for ARPACK.

        Notes
        -----
        Assumes the operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
        to reassure themselves of the hermiticity properties before use.

        Parameters
        ----------
        eigsh_args :
                For all additional arguments see documentation of `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_.

        Returns
        -------
        tuple
                Tuple containing the `(eigenvalues, eigenvectors)` of the `quantum_LinearOperator` operator.

        Examples
        --------
        >>> eigenvalues,eigenvectors = H.eigsh(**eigsh_args)

        """
        return _sla.eigsh(self, **eigsh_args)

    ### algebra operations

    def transpose(self, copy=False):
        """Transposes `quantum_LinearOperator` operator.

        Notes
        -----
        This function does NOT conjugate the operator.

        Returns
        -------
        :obj:`quantum_LinearOperator`
                :math:`H_{ij}\\mapsto H_{ji}`

        Examples
        --------

        >>> H_tran = H.transpose()

        """
        if copy:
            return self.copy().transpose()
        else:
            self._transposed = not self._transposed
            return self

    def conjugate(self):
        """Conjugates `quantum_LinearOperator` operator.

        Notes
        -----
        This function does NOT transpose the operator.

        Returns
        -------
        :obj:`quantum_LinearOperator`
                :math:`H_{ij}\\mapsto H_{ij}^*`

        Examples
        --------

        >>> H_conj = H.conj()

        """
        self._conjugated = not self._conjugated
        return self

    def conj(self):
        """Conjugates `quantum_LinearOperator` operator.

        Notes
        -----
        This function does NOT transpose the operator.

        Returns
        -------
        :obj:`quantum_LinearOperator`
                :math:`H_{ij}\\mapsto H_{ij}^*`

        Examples
        --------

        >>> H_conj = H.conj()

        """
        return self.conjugate()

    def getH(self, copy=False):
        """Calculates hermitian conjugate of `quantum_LinearOperator` operator.

        Parameters
        ----------
        copy : bool, optional
                Whether to return a deep copy of the original object. Default is `copy = False`.

        Returns
        -------
        :obj:`quantum_LinearOperator`
                :math:`H_{ij}\\mapsto H_{ji}^*`

        Examples
        --------

        >>> H_herm = H.getH()

        """
        if copy:
            return self.copy().get_H()
        else:
            return self.conj().transpose()

    ### special methods

    def copy(self):
        """Returns a deep copy of `quantum_LinearOperator` object."""
        return quantum_LinearOperator(
            list(self._static_list),
            basis=self._basis,
            diagonal=self._diagonal,
            dtype=self._dtype,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
            copy=True,
        )

    def __repr__(self):
        return "<{0}x{1} quspin quantum_LinearOperator of type '{2}'>".format(
            *(self._shape[0], self._shape[1], self._dtype)
        )

    def __neg__(self):
        return self.__mul__(-1)

    def __add__(self, other):
        if other.__class__ in [_np.ndarray, _np.matrix]:
            dense = True
        elif _sp.issparse(other):
            dense = False
        elif ishamiltonian(other):
            return self._add_hamiltonian(other)
        elif isinstance(other, LinearOperator):
            return LinearOperator.__add__(self, other)
        elif _np.isscalar(other):
            return LinearOperator.__add__(self, other)
        else:
            dense = True
            other = _np.asanyarray(other)

        if self._shape != other.shape:
            raise ValueError(
                "dimension mismatch with shapes {0} and {1}".format(
                    self._shape, other.shape
                )
            )

        if dense:
            return self._add_dense(other)
        else:
            return self._add_sparse(other)

    def __iadd__(self, other):
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if other.__class__ in [_np.ndarray, _np.matrix]:
            dense = True
        elif _sp.issparse(other):
            dense = False
        elif ishamiltonian(other):
            return self._sub_hamiltonian(other)
        elif isinstance(other, LinearOperator):
            return LinearOperator.__sub__(self, other)
        elif _np.isscalar(other):
            return LinearOperator.__sub__(self, other)
        else:
            dense = False
            other = _np.asanyarray(other)

        if self._shape != other.shape:
            raise ValueError(
                "dimension mismatch with shapes {0} and {1}".format(
                    self._shape, other.shape
                )
            )

        if dense:
            return self._sub_dense(other)
        else:
            return self._sub_sparse(other)

    def __isub__(self, other):
        return NotImplemented

    def __rsub__(self, other):
        return -(self.__sub__(other))

    def __imul__(self, other):
        if _np.isscalar(other):
            return self._mul_scalar(other)
        else:
            return NotImplemented

    def __mul__(self, other):
        if other.__class__ in [_np.ndarray, _np.matrix]:
            dense = True
        elif _sp.issparse(other):
            dense = False
        elif ishamiltonian(other):
            return self._mul_hamiltonian(other)
        elif isinstance(other, LinearOperator):
            return LinearOperator.dot(self, other)
        elif _np.asarray(other).ndim == 0:
            return self._mul_scalar(other)
        else:
            dense = True
            other = _np.asanyarray(other)

        if self.get_shape[1] != other.shape[0]:
            raise ValueError(
                "dimension mismatch with shapes {} and {}".format(
                    self._shape, other.shape
                )
            )

        if dense:
            if other.ndim == 1:
                return self._matvec(other)
            elif other.ndim == 2:
                return self._matmat(other)
            else:
                raise ValueError
        else:
            return self._mul_sparse(other)

    def __rmul__(self, other):
        if other.__class__ in [_np.ndarray, _np.matrix]:
            dense = True
        elif _sp.issparse(other):
            dense = False
        elif ishamiltonian(other):
            return self._rmul_hamiltonian(other)
        elif isinstance(other, LinearOperator):
            return LinearOperator.dot(other.transpose(), self.transpose()).transpose()
        elif _np.isscalar(other):
            return self._mul_scalar(other)
        else:
            dense = True
            other = _np.asanyarray(other)

        if dense:
            if other.ndim == 1:
                return self.T._matvec(other)
            elif other.ndim == 2:
                if self._shape[0] != other.shape[1]:
                    raise ValueError(
                        "dimension mismatch with shapes {0} and {1}".format(
                            self._shape, other.shape
                        )
                    )
                return (self.T._matmat(other.T)).T
            else:
                raise ValueError
        else:
            if self._shape[0] != other.shape[1]:
                raise ValueError(
                    "dimension mismatch with shapes {0} and {1}".format(
                        self._shape, other.shape
                    )
                )
            return (self.T._mul_sparse(other.T)).T

    def _mul_scalar(self, other):
        self._dtype = _np.result_type(self._dtype, other)
        self._scale *= other

    def _mul_hamiltonian(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        static = self.__mul__(other._static_list)
        dynamic = [[self.__mul__(Hd), func] for func, Hd in iteritems(other.dynamic)]
        return hamiltonian(
            [static], dynamic, basis=self._basis, dtype=result_dtype, copy=False
        )

    def _mul_sparse(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        if self.diagonal is not None:
            new_other = _sp.dia_matrix(
                (_np.asarray([self._diagonal]), _np.array([0])), shape=self._shape
            ).dot(other)
            if new_other.dtype != result_dtype:
                new_other = new_other.astype(result_dtype)
        else:
            new_other = _sp.csr_matrix(other.shape, dtype=result_dtype)

        for opstr, indx, J in self.static_list:
            if not self._transposed:
                ME, row, col = self.basis.Op(opstr, indx, J, self._dtype)
            else:
                ME, col, row = self.basis.Op(opstr, indx, J, self._dtype)

            if self._conjugated:
                ME = ME.conj()

            new_other = new_other + _sp.csr_matrix(
                (ME, (row, col)), shape=self._shape
            ).dot(other)

        return new_other

    def _rmul_hamiltonian(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        static = self.__rmul__(other._static_list)
        dynamic = [[self.__rmul__(Hd), func] for func, Hd in iteritems(other.dynamic)]
        return hamiltonian(
            [static], dynamic, basis=self._basis, dtype=result_dtype, copy=False
        )

    def _add_hamiltonian(self, other):
        return NotImplemented

    def _add_sparse(self, other):
        return NotImplemented

    def _add_dense(self, other):
        return NotImplemented

    def _sub_sparse(self, other):
        return NotImplemented

    def _sub_hamiltonian(self, other):
        return NotImplemented

    def _sub_dense(self, other):
        return NotImplemented

    def __numpy_ufunc__(self, func, method, pos, inputs, **kwargs):
        # """Method for compatibility with NumPy's ufuncs and dot
        # functions.
        # """

        if (func == _np.dot) or (func == _np.multiply):
            if pos == 0:
                return self.__mul__(inputs[1])
            if pos == 1:
                return self.__rmul__(inputs[0])
            else:
                return NotImplemented


def isquantum_LinearOperator(obj):
    """Checks if instance is object of `quantum_LinearOperator` class.

    Parameters
    ----------
    obj :
            Arbitraty python object.

    Returns
    -------
    bool
            Can be either of the following:

            * `True`: `obj` is an instance of `quantum_LinearOperator` class.
            * `False`: `obj` is NOT an instance of `quantum_LinearOperator` class.

    """
    return isinstance(obj, quantum_LinearOperator)
