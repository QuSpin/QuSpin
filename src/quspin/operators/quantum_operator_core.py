from quspin.basis import spin_basis_1d as _default_basis
from quspin.basis import isbasis as _isbasis

# from quspin.operators._oputils import _get_matvec_function, matvec as _matvec

from quspin.tools.matvec import _matvec
from quspin.tools.matvec import _get_matvec_function

from quspin.operators._make_hamiltonian import make_static
from quspin.operators._make_hamiltonian import _check_almost_zero

from quspin.operators import hamiltonian_core

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

import functools
from six import iteritems, itervalues, viewkeys

from zipfile import ZipFile
from tempfile import TemporaryDirectory
import os, pickle


__all__ = ["quantum_operator", "isquantum_operator", "save_zip", "load_zip"]


# function used to create Linearquantum_operator with fixed set of parameters.
def _quantum_operator_dot(op, pars, v):
    return op.dot(v, pars=pars, check=False)


class quantum_operator(object):
    """Constructs parameter-dependent (hermitian and nonhermitian) operators.

    The `quantum_operator` class maps quantum operators to keys of a dictionary. When calling various methods
    of `quantum_operator`, it allows one to 'dynamically' specify the pre-factors of these operators.

    Examples
    --------

    It is often required to be able to handle a parameter-dependent Hamiltonian :math:`H(\\lambda)=H_0 + \\lambda H_1`, e.g.

    .. math::
            H_0=\\sum_j J_{zz}S^z_jS^z_{j+2} + h_xS^x_j, \\qquad H_1=\\sum_j S^z_j

    The following code snippet shows how to use the `quantum_operator` class to vary the parameter :math:`\\lambda`
    without having to re-build the Hamiltonian every time.

    .. literalinclude:: ../../doc_examples/quantum_operator-example.py
            :linenos:
            :language: python
            :lines: 7-

    """

    def __init__(
        self,
        input_dict,
        N=None,
        basis=None,
        shape=None,
        copy=True,
        check_symm=True,
        check_herm=True,
        check_pcon=True,
        matrix_formats={},
        dtype=_np.complex128,
        **basis_args,
    ):
        """Intializes the `quantum_operator` object (parameter dependent quantum quantum_operators).

        Parameters
        ----------
        input_dict : dict
                The `values` of this dictionary contain quantum_operator lists, in the same format as the `static_list`
                argument of the `hamiltonian` class.

                The `keys` of this dictionary correspond to the parameter values, e.g. :math:`J_{zz},h_x`, and are
                used to specify the coupling strength during calls of the `quantum_operator` class methods.

                >>> # use "Jzz" and "hx" keys to specify the zz and x coupling strengths, respectively
                >>> input_dict = { "Jzz": [["zz",Jzz_bonds]], "hx" : [["x" ,hx_site ]] }

        N : int, optional
                Number of lattice sites for the `hamiltonian` object.
        dtype : 'type'
                Data type (e.g. numpy.float64) to construct the quantum_operator with.
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
        matrix_formats: dict, optional
                Dictionary of key,value pairs which, given a key associated with an operator in `input_dict`, the value of this key
                specifies the sparse matrix format {"csr","csc","dia","dense"}.
        kw_args : dict
                Optional additional arguments to pass to the `basis` class, if not already using a `basis` object
                to create the quantum_operator.

        """
        self._is_dense = False
        self._ndim = 2
        self._basis = basis

        if not (dtype in hamiltonian_core.supported_dtypes):
            raise TypeError("hamiltonian does not support type: " + str(dtype))
        else:
            self._dtype = dtype

        opstr_dict = {}
        other_dict = {}
        self._quantum_operator = {}
        if isinstance(input_dict, dict):
            for key, op in iteritems(input_dict):
                if type(key) is not str:
                    raise ValueError("keys to input_dict must be strings.")

                if type(op) not in [list, tuple]:
                    raise ValueError(
                        "input_dict must contain values which are lists/tuples."
                    )

                opstr_list = []
                other_list = []
                for ele in op:
                    if hamiltonian_core._check_static(ele):
                        opstr_list.append(ele)
                    else:
                        other_list.append(ele)

                if opstr_list:
                    opstr_dict[key] = opstr_list
                if other_list:
                    other_dict[key] = other_list
        else:
            raise ValueError(
                "input_dict must be dictionary or another quantum_operator quantum_operators"
            )

        if opstr_dict:
            # check if user input basis

            if basis is not None:
                if len(basis_args) > 0:
                    wrong_keys = set(basis_args.keys())
                    temp = ", ".join(["{}" for key in wrong_keys])
                    raise ValueError(
                        ("unexpected optional argument(s): " + temp).format(*wrong_keys)
                    )

            # if not
            if basis is None:
                if N is None:  # if L is missing
                    raise Exception(
                        "if opstrs in use, argument N needed for basis class"
                    )

                if type(N) is not int:  # if L is not int
                    raise TypeError("argument N must be integer")

                basis = _default_basis(N, **basis_args)

            elif not _isbasis(basis):
                raise TypeError("expecting instance of basis class for argument: basis")

            static_opstr_list = []
            for key, opstr_list in iteritems(opstr_dict):
                static_opstr_list.extend(opstr_list)

            if check_herm:
                basis.check_hermitian(static_opstr_list, [])

            if check_symm:
                basis.check_symm(static_opstr_list, [])

            if check_pcon:
                basis.check_pcon(static_opstr_list, [])

            self._shape = (basis.Ns, basis.Ns)

            for key, opstr_list in iteritems(opstr_dict):
                O = make_static(basis, opstr_list, dtype)
                self._quantum_operator[key] = O

        if other_dict:
            if not hasattr(self, "_shape"):
                found = False
                if (
                    shape is None
                ):  # if no shape argument found, search to see if the inputs have shapes.
                    for key, O_list in iteritems(other_dict):
                        for O in O_list:
                            try:  # take the first shape found
                                shape = O.shape
                                found = True
                                break
                            except AttributeError:
                                continue
                else:
                    found = True

                if not found:
                    raise ValueError("no dictionary entries have shape attribute.")
                if shape[0] != shape[1]:
                    raise ValueError("quantum_operator must be square matrix")

                self._shape = shape

            for key, O_list in iteritems(other_dict):
                for i, O in enumerate(O_list):
                    if _sp.issparse(O):
                        self._mat_checks(O)
                        if i == 0:
                            self._quantum_operator[key] = O
                        else:
                            try:
                                self._quantum_operator[key] += O
                            except NotImplementedError:
                                self._quantum_operator[key] = (
                                    self._quantum_operator[key] + O
                                )

                    elif O.__class__ is _np.ndarray:
                        self._mat_checks(O)
                        self._is_dense = True
                        if i == 0:
                            self._quantum_operator[key] = O
                        else:
                            try:
                                self._quantum_operator[key] += O
                            except NotImplementedError:
                                self._quantum_operator[key] = (
                                    self._quantum_operator[key] + O
                                )

                    elif O.__class__ is _np.matrix:
                        self._mat_checks(O)
                        self._is_dense = True
                        if i == 0:
                            self._quantum_operator[key] = O
                        else:
                            try:
                                self._quantum_operator[key] += O
                            except NotImplementedError:
                                self._quantum_operator[key] = (
                                    self._quantum_operator[key] + O
                                )

                    else:
                        O = _np.asanyarray(O)
                        self._mat_checks(O)
                        if i == 0:
                            self._quantum_operator[key] = O
                        else:
                            try:
                                self._quantum_operator[key] += O
                            except NotImplementedError:
                                self._quantum_operator[key] = (
                                    self._quantum_operator[key] + O
                                )

        else:
            if not hasattr(self, "_shape"):
                if shape is None:
                    # check if user input basis
                    basis = basis_args.get("basis")

                    # if not
                    if basis is None:
                        if N is None:  # if N is missing
                            raise Exception(
                                "argument N or shape needed to create empty quantum_operator"
                            )

                        if type(N) is not int:  # if L is not int
                            raise TypeError("argument N must be integer")

                        basis = _default_basis(N, **basis_args)

                    elif not _isbasis(basis):
                        raise TypeError(
                            "expecting instance of basis class for argument: basis"
                        )

                    shape = (basis.Ns, basis.Ns)

                else:
                    basis = basis_args.get("basis")
                    if not basis is None:
                        raise ValueError(
                            "empty hamiltonian only accepts basis or shape, not both"
                        )

                if len(shape) != 2:
                    raise ValueError("expecting ndim = 2")
                if shape[0] != shape[1]:
                    raise ValueError("hamiltonian must be square matrix")

                self._shape = shape

        if basis is not None:
            self._basis = basis

        self._Ns = self._shape[0]

        keys = list(self._quantum_operator.keys())
        for key in keys:
            if _check_almost_zero(self._quantum_operator[key]):
                self._quantum_operator.pop(key)

        self.update_matrix_formats(matrix_formats)

    def get_operators(self, key):
        return self._quantum_operator[key]

    @property
    def shape(self):
        """tuple: shape of the `quantum_operator` object, always equal to `(Ns,Ns)`."""
        return self._shape

    @property
    def basis(self):
        """:obj:`basis`: basis used to build the `hamiltonian` object. Defaults to `None` if quantum_operator has
        no basis (i.e. was created externally and passed as a precalculated array).

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
    def get_shape(self):
        """tuple: shape of the `quantum_operator` object, always equal to `(Ns,Ns)`."""
        return self._shape

    @property
    def shape(self):
        """tuple: shape of the `quantum_operator` object, always equal to `(Ns,Ns)`."""
        return self._shape

    @property
    def is_dense(self):
        """bool: `True` if `quantum_operator` contains a dense matrix as a componnent of either
        the static or dynamic list.

        """
        return self._is_dense

    @property
    def dtype(self):
        """type: data type of `quantum_operator` object."""
        return _np.dtype(self._dtype).name

    @property
    def T(self):
        """:obj:`quantum_operator`: transposes the operator matrix, :math:`H_{ij}\\mapsto H_{ji}`."""
        return self.transpose()

    @property
    def H(self):
        """:obj:`quantum_operator`: transposes and conjugates the operator matrix, :math:`H_{ij}\\mapsto H_{ji}^*`."""
        return self.getH()

    ### state manipulation/observable routines

    def matvec(self, x):
        """Matrix-vector multiplication.

        Performs the operation y=A*x where A is an MxN linear operator and x is a column vector or 1-d array.

        Notes
        -----
        This matvec wraps the user-specified matvec routine or overridden _matvec method to ensure that y has the correct shape and type.

        Parameters
        ----------
        x : {matrix, ndarray}
                An array with shape (N,) or (N,1).

        Returns
        -------
        y : {matrix, ndarray}
                A matrix or ndarray with shape (M,) or (M,1) depending on the type and shape of the x argument.

        """

        return self.dot(x)

    def rmatvec(self, x):
        """Adjoint matrix-vector multiplication.

        Performs the operation y = A^H * x where A is an MxN linear operator and x is a column vector or 1-d array.

        Notes
        -----
        This rmatvec wraps the user-specified rmatvec routine or overridden _rmatvec method to ensure that y has the correct shape and type.

        Parameters
        ----------
        x : {matrix, ndarray}
                An array with shape (M,) or (M,1).

        Returns
        -------
        y : {matrix, ndarray}
                A matrix or ndarray with shape (N,) or (N,1) depending on the type and shape of the x argument.

        """
        return self.T.conj().dot(x)

    def matmat(self, X):
        """Matrix-matrix multiplication.

        Performs the operation y=A*X where A is an MxN linear operator and X dense N*K matrix or ndarray.

        Notes
        -----
        This matmat wraps any user-specified matmat routine or overridden _matmat method to ensure that y has the correct type.

        Parameters
        ----------
        X : {matrix, ndarray}
                An array with shape (N,K).

        Returns
        -------
        Y : {matrix, ndarray}
                A matrix or ndarray with shape (M,K) depending on the type of the X argument.

        """
        return self.dot(X)

    def dot(self, V, pars={}, check=True, out=None, overwrite_out=True, a=1.0):
        """Matrix-vector multiplication of `quantum_operator` quantum_operator for parameters `pars`, with state `V`.

        .. math::
                aH(\\lambda)|V\\rangle

        Notes
        -----

        Parameters
        ----------
        V : numpy.ndarray
                Vector (quantums tate) to multiply the `quantum_operator` quantum_operator with.
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
        check : bool, optional
                Whether or not to do checks for shape compatibility.
        out : array_like, optional
                specify the output array for the the result. This is not supported if `V` is a sparse matrix.
        overwrite_out : bool, optional
                flag used to toggle between two different ways to treat `out`. If set to `True` all values in `out` will be overwritten with the result of the dot product.
                If `False` the result of the dot product will be added to the values of `out`.
        a : scalar, optional
                scalar to multiply the final product with: :math:`B = aHV`.

        Returns
        -------
        numpy.ndarray
                Vector corresponding to the `quantum_operator` quantum_operator applied on the state `V`.

        Examples
        --------
        >>> B = H.dot(A,pars=pars,check=True)

        corresponds to :math:`B = HA`.

        """

        pars = self._check_scalar_pars(pars)

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

        result_dtype = _np.result_type(V.dtype, self._dtype)

        if not (result_dtype in hamiltonian_core.supported_dtypes):
            raise TypeError("hamiltonian does not support type: " + str(dtype))

        if V.ndim == 3:
            eps = _np.finfo(self.dtype).eps

            if V.shape[0] != V.shape[1]:
                raise ValueError("Density matrices must be square!")

            # allocate C-contiguous array to output results in.
            out = _np.zeros(V.shape[-1:] + V.shape[:-1], dtype=result_dtype)

            for i in range(V.shape[2]):
                v = _np.ascontiguousarray(V[..., i], dtype=result_dtype)
                for key, J in pars.items():
                    if _np.abs(J) > eps:
                        kwargs = dict(overwrite_out=False, a=a * J, out=out[i, ...])
                        self._matvec_functions[key](
                            self._quantum_operator[key], v, **kwargs
                        )

            return out.transpose((1, 2, 0))

        if _sp.issparse(V):
            if out is not None:
                raise TypeError("'out' option does not apply for sparse inputs.")

            sparse_constuctor = getattr(_sp, V.getformat() + "_matrix")
            out = sparse_constuctor(V.shape, dtype=result_dtype)
            for key, J in pars.items():
                out = out + J * self._quantum_operator[key].dot(V)

            out = a * out

        else:
            if out is not None:
                try:
                    if out.dtype != result_dtype:
                        raise TypeError(
                            "'out' must be array with correct dtype and dimensions for output array."
                        )
                    if out.shape != V.shape:
                        raise ValueError(
                            "'out' must be array with correct dtype and dimensions for output array."
                        )
                except AttributeError:
                    raise TypeError(
                        "'out' must be C-contiguous array with correct dtype and dimensions for output array."
                    )

                if overwrite_out:
                    out[...] = 0
            else:
                out = _np.zeros_like(V, dtype=result_dtype)

            eps = _np.finfo(self.dtype).eps
            V = _np.asarray(V, dtype=result_dtype)

            for key, J in pars.items():
                if _np.abs(J) > eps:
                    self._matvec_functions[key](
                        self._quantum_operator[key],
                        V,
                        overwrite_out=False,
                        a=a * J,
                        out=out,
                    )

        return out

    def rdot(self, V, pars={}, check=False, out=None, overwrite_out=True, a=1.0):
        """Vector-matrix multiplication of `quantum_operator` quantum_operator for parameters `pars`, with state `V`.

        .. math::
                a\\langle V]H(\\lambda)


        Parameters
        ----------
        V : numpy.ndarray
                Vector (quantums tate) to multiply the `quantum_operator` quantum_operator with.
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
        check : bool, optional
                Whether or not to do checks for shape compatibility.
        out : array_like, optional
                specify the output array for the the result. This is not supported if `V` is a sparse matrix.
        overwrite_out : bool, optional
                flag used to toggle between two different ways to treat `out`. If set to `True` all values in `out` will be overwritten with the result.
                If `False` the result of the dot product will be added to the values of `out`.
        a : scalar, optional
                scalar to multiply the final product with: :math:`B = aVH`.


        Returns
        -------
        numpy.ndarray
                Vector corresponding to the `quantum_operator` quantum_operator applied on the state `V`.

        Examples
        --------
        >>> B = H.dot(A,pars=pars,check=True)

        corresponds to :math:`B = AH`.

        """
        return (
            self.transpose()
            .dot(
                V.transpose(),
                pars=pars,
                check=check,
                out=out.T,
                overwrite_out=overwrite_out,
                a=a,
            )
            .transpose()
        )

    def quant_fluct(self, V, pars={}, check=True, enforce_pure=False):
        """Calculates the quantum fluctuations (variance) of `quantum_operator` object for parameters `pars`, in state `V`.

        .. math::
                \\langle V|H(\\lambda)^2|V\\rangle - \\langle V|H(\\lambda)|V\\rangle^2

        Parameters
        ----------
        V : numpy.ndarray
                Depending on the shape, can be a single state or a collection of pure or mixed states
                [see `enforce_pure`].
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.

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

        if hamiltonian_core.ishamiltonian(V):
            raise TypeError("Can't take expectation value of hamiltonian")

        if isexp_op(V):
            raise TypeError("Can't take expectation value of exp_op")

        # fluctuations =  expctH2 - expctH^2
        kwargs = dict(enforce_pure=enforce_pure)
        V_dot = self.dot(V, pars=pars, check=check)
        expt_value_sq = self._expt_value_core(V, V_dot, **kwargs) ** 2

        if V.ndim == 1 or (V.shape[0] != V.shape[1] or enforce_pure):
            sq_expt_value = self._expt_value_core(V_dot, V_dot, **kwargs)
        else:
            V_dot = self.dot(V_dot, pars=pars, check=check)
            sq_expt_value = self._expt_value_core(V, V_dot, **kwargs)

        return sq_expt_value - expt_value_sq

    def expt_value(self, V, pars={}, check=True, enforce_pure=False):
        """Calculates expectation value of of `quantum_operator` object for parameters `pars`, in state `V`.

        .. math::
                \\langle V|H(\\lambda)|V\\rangle

        Parameters
        ----------
        V : numpy.ndarray
                Depending on the shape, can be a single state or a collection of pure or mixed states
                [see `enforce_pure` argument of `basis.ent_entropy`].
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
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

        if hamiltonian_core.ishamiltonian(V):
            raise TypeError("Can't take expectation value of hamiltonian")

        if isexp_op(V):
            raise TypeError("Can't take expectation value of exp_op")

        V_dot = self.dot(V, check=check, pars=pars)
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

    def matrix_ele(self, Vl, Vr, pars={}, diagonal=False, check=True):
        """Calculates matrix element of `quantum_operator` object for parameters `pars` in states `Vl` and `Vr`.

        .. math::
                \\langle V_l|H(\\lambda)|V_r\\rangle

        Notes
        -----
        Taking the conjugate or transpose of the state `Vl` is done automatically.

        Parameters
        ----------
        Vl : numpy.ndarray
                Vector(s)/state(s) to multiple with on left side.
        Vl : numpy.ndarray
                Vector(s)/state(s) to multiple with on right side.
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
        diagonal : bool, optional
                When set to `True`, returs only diagonal part of expectation value. Default is `diagonal = False`.
        check : bool,

        Returns
        -------
        float
                Matrix element of `quantum_operator` quantum_operator between the states `Vl` and `Vr`.

        Examples
        --------
        >>> H_lr = H.expt_value(Vl,Vr,pars=pars,diagonal=False,check=True)

        corresponds to :math:`H_{lr} = \\langle V_l|H(\\lambda=0)|V_r\\rangle`.

        """
        Vr = self.dot(Vr, pars=pars, check=check)

        if check:
            try:
                shape = Vl.shape
            except AttributeError:
                Vl = _np.asanyarray(Vl)
                shape = Vl.shape

            if shape[0] != self._shape[1]:
                raise ValueError(
                    "matrix dimension mismatch with shapes: {0} and {1}.".format(
                        V.shape, self._shape
                    )
                )

            if Vl.ndim > 2:
                raise ValueError("Expecting  0< V.ndim < 3.")

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

    ### Diagonalisation routines

    def eigsh(self, pars={}, **eigsh_args):
        """Computes SOME eigenvalues and eigenvectors of hermitian `quantum_operator` quantum_operator using SPARSE hermitian methods.

        This function method solves for eigenvalues and eigenvectors, but can only solve for a few of them accurately.
        It calls `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_, which is a wrapper for ARPACK.

        Notes
        -----
        Assumes the quantum_operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
        to reassure themselves of the hermiticity properties before use.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
        eigsh_args :
                For all additional arguments see documentation of `scipy.sparse.linalg.eigsh <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_.

        Returns
        -------
        tuple
                Tuple containing the `(eigenvalues, eigenvectors)` of the `quantum_operator` quantum_operator.

        Examples
        --------
        >>> eigenvalues,eigenvectors = H.eigsh(pars=pars,**eigsh_args)

        """
        if self.Ns == 0:
            return _np.array([]), _np.array([[]])

        return _sla.eigsh(self.tocsr(pars), **eigsh_args)

    def eigh(self, pars={}, **eigh_args):
        """Computes COMPLETE eigensystem of hermitian `quantum_operator` quantum_operator using DENSE hermitian methods.

        This function method solves for all eigenvalues and eigenvectors. It calls
        `numpy.linalg.eigh <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html>`_,
        and uses wrapped LAPACK functions which are contained in the module py_lapack.

        Notes
        -----
        Assumes the quantum_operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
        to reassure themselves of the hermiticity properties before use.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
        eigh_args :
                For all additional arguments see documentation of `numpy.linalg.eigh <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html>`_.

        Returns
        -------
        tuple
                Tuple containing the `(eigenvalues, eigenvectors)` of the `quantum_operator` quantum_operator.

        Examples
        --------
        >>> eigenvalues,eigenvectors = H.eigh(pars=pars,**eigh_args)

        """
        eigh_args["overwrite_a"] = True

        if self.Ns <= 0:
            return _np.asarray([]), _np.asarray([[]])

        # fill dense array with hamiltonian
        H_dense = self.todense(pars=pars)
        # calculate eigh
        E, H_dense = _la.eigh(H_dense, **eigh_args)
        return E, H_dense

    def eigvalsh(self, pars={}, **eigvalsh_args):
        """Computes ALL eigenvalues of hermitian `quantum_operator` quantum_operator using DENSE hermitian methods.

        This function method solves for all eigenvalues. It calls
        `numpy.linalg.eigvalsh <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html>`_,
        and uses wrapped LAPACK functions which are contained in the module py_lapack.

        Notes
        -----
        Assumes the quantum_operator is hermitian! If the flat `check_hermiticity = False` is used, we advise the user
        to reassure themselves of the hermiticity properties before use.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
        eigvalsh_args :
                For all additional arguments see documentation of `numpy.linalg.eigvalsh <https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html>`_.

        Returns
        -------
        numpy.ndarray
                Eigenvalues of the `quantum_operator` quantum_operator.

        Examples
        --------
        >>> eigenvalues = H.eigvalsh(pars=pars,**eigvalsh_args)

        """

        if self.Ns <= 0:
            return _np.asarray([])

        H_dense = self.todense(pars=pars)
        E = _np.linalg.eigvalsh(H_dense, **eigvalsh_args)
        # eigvalsh_args["overwrite_a"] = True
        # E = _la.eigvalsh(H_dense,**eigvalsh_args)
        return E

    ### routines to change object type

    def tocsr(self, pars={}):
        """Returns copy of a `quantum_operator` object for parameters `pars` as a `scipy.sparse.csr_matrix`.

        Casts the `quantum_operator` object as a
        `scipy.sparse.csr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_
        object.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.

        Returns
        -------
        :obj:`scipy.sparse.csr_matrix`

        Examples
        --------
        >>> H_csr=H.tocsr(pars=pars)

        """
        pars = self._check_scalar_pars(pars)

        H = _sp.csr_matrix(self.get_shape, dtype=self._dtype)

        for key, J in pars.items():
            try:
                H += J * _sp.csr_matrix(self._quantum_operator[key])
            except:
                H = H + J * _sp.csr_matrix(self._quantum_operator[key])

        return H

    def tocsc(self, pars={}):
        """Returns copy of a `quantum_operator` object for parameters `pars` as a `scipy.sparse.csc_matrix`.

        Casts the `quantum_operator` object as a
        `scipy.sparse.csc_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csc_matrix.html>`_
        object.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.

        Returns
        -------
        :obj:`scipy.sparse.csc_matrix`

        Examples
        --------
        >>> H_csc=H.tocsc(pars=pars)

        """
        pars = self._check_scalar_pars(pars)

        H = _sp.csc_matrix(self.get_shape, dtype=self._dtype)

        for key, J in pars.items():
            try:
                H += J * _sp.csc_matrix(self._quantum_operator[key])
            except:
                H = H + J * _sp.csc_matrix(self._quantum_operator[key])

        return H

    def todense(self, pars={}, out=None):
        """Returns copy of a `quantum_operator` object for parameters `pars` as a dense array.

        This function can overflow memory if not used carefully!

        Notes
        -----
        If the array dimension is too large, scipy may choose to cast the `quantum_operator` quantum_operator as a
        `numpy.matrix` instead of a `numpy.ndarray`. In such a case, one can use the `quantum_operator.toarray()`
        method.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
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
        >>> H_dense=H.todense(pars=pars)

        """
        pars = self._check_scalar_pars(pars)

        if out is None:
            out = _np.zeros(self._shape, dtype=self.dtype)
            out = _np.asmatrix(out)

        for key, J in pars.items():
            out += J * self._quantum_operator[key]

        return out

    def toarray(self, pars={}, out=None):
        """Returns copy of a `quantum_operator` object for parameters `pars` as a dense array.

        This function can overflow memory if not used carefully!


        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.
        out : numpy.ndarray
                Array to fill in with the output.

        Returns
        -------
        numpy.ndarray
                Dense array.

        Examples
        --------
        >>> H_dense=H.toarray(pars=pars)

        """

        pars = self._check_scalar_pars(pars)

        if out is None:
            out = _np.zeros(self._shape, dtype=self.dtype)

        for key, J in pars.items():
            out += J * self._quantum_operator[key]

        return out

    def aslinearoperator(self, pars={}):
        """Returns copy of a `quantum_operator` object for parameters `pars` as a `scipy.sparse.linalg.Linearquantum_operator`.

        Casts the `quantum_operator` object as a
        `scipy.sparse.linalg.Linearquantum_operator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>`_
        object.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.

        Returns
        -------
        :obj:`scipy.sparse.linalg.Linearquantum_operator`

        Examples
        --------
        >>> H_aslinop=H.aslinearquantum_operator(pars=pars)

        """
        pars = self._check_scalar_pars(pars)
        matvec = functools.partial(_quantum_operator_dot, self, pars)
        rmatvec = functools.partial(_quantum_operator_dot, self.T.conj(), pars)
        return _sla.LinearOperator(
            self.get_shape, matvec, rmatvec=rmatvec, matmat=matvec, dtype=self._dtype
        )

    def tohamiltonian(self, pars={}, copy=True):
        """Returns copy of a `quantum_operator` object for parameters `pars` as a `hamiltonian` object.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.

        copy : bool, optional
                Explicitly copy matrices when constructing the `hamiltonian` object, default is `True`.

        Returns
        -------
        :obj:`hamiltonian`

        Examples
        --------
        >>> H_ham=H.tohamiltonian(pars=pars)

        """
        pars = self._check_hamiltonian_pars(pars)

        static = []
        dynamic = []

        for key, J in pars.items():
            if type(J) is tuple and len(J) == 2:
                dynamic.append([self._quantum_operator[key], J[0], J[1]])
            else:
                if J == 1.0:
                    static.append(self._quantum_operator[key])
                else:
                    static.append(J * self._quantum_operator[key])

        return hamiltonian_core.hamiltonian(
            static, dynamic, dtype=self._dtype, copy=copy
        )

    def update_matrix_formats(self, matrix_formats):
        """Change the internal structure of the matrices in-place.

        Parameters
        ----------
        matrix_formats: dict, optional
                Dictionary of key,value pairs which, given a key associated with an operator in `input_dict`, the value of this key
                specifies the sparse matrix format {"csr","csc","dia","dense"}.

        Examples
        --------
        Given `O` which has two operators defined by strings: 'Hx' for transverse field, and 'Hising' is the Ising part. The Ising part must be diagonal
        therefore it is useful to cast it to a DIA matrix format, while the transverse field is not diagonal so it is most efficient to use CSR matrix format.
        This can be accomplished by the following:

        >>> O.update_matrix_formats(dict(Hx="csr",Hising="dia"))

        """
        if type(matrix_formats) is not dict:
            raise ValueError(
                "matrix_formats must be a dictionary with the formats of the matrices being values and keys being the operator keys."
            )

        extra = set(matrix_formats.keys()) - set(self._quantum_operator.keys())
        if extra:
            raise ValueError("unexpected couplings: {}".format(extra))

        for key in self._quantum_operator.keys():
            if key in matrix_formats:
                fmt = matrix_formats[key]
                if fmt not in ["dia", "csr", "csc", "dense"]:
                    raise TypeError(
                        "sparse formats must be either 'csr','csc', 'dia' or 'dense'."
                    )

                if fmt == "dense":
                    O = self._quantum_operator[key]
                    try:
                        self._quantum_operator[key] = O.toarray()
                    except AttributeError:
                        self._quantum_operator[key] = _np.ascontiguousarray(O)
                else:
                    sparse_constuctor = getattr(_sp, fmt + "_matrix")
                    O = self._quantum_operator[key]
                    if _sp.issparse(O):
                        self._quantum_operator[key] = sparse_constuctor(O)

        self._update_matvecs()

    ### algebra operations

    def transpose(self, copy=False):
        """Transposes `quantum_operator` quantum_operator.

        Notes
        -----
        This function does NOT conjugate the quantum_operator.

        Returns
        -------
        :obj:`quantum_operator`
                :math:`H_{ij}\\mapsto H_{ji}`

        Examples
        --------

        >>> H_tran = H.transpose()

        """

        new_dict = {
            key: [op.transpose()] for key, op in iteritems(self._quantum_operator)
        }
        return quantum_operator(
            new_dict, basis=self._basis, dtype=self._dtype, shape=self._shape, copy=copy
        )

    def conjugate(self):
        """Conjugates `quantum_operator` quantum_operator.

        Notes
        -----
        This function does NOT transpose the quantum_operator.

        Returns
        -------
        :obj:`quantum_operator`
                :math:`H_{ij}\\mapsto H_{ij}^*`

        Examples
        --------

        >>> H_conj = H.conj()

        """
        new_dict = {
            key: [op.conjugate()] for key, op in iteritems(self._quantum_operator)
        }
        return quantum_operator(
            new_dict,
            basis=self._basis,
            dtype=self._dtype,
            shape=self._shape,
            copy=False,
        )

    def conj(self):
        """Conjugates `quantum_operator` quantum_operator.

        Notes
        -----
        This function does NOT transpose the quantum_operator.

        Returns
        -------
        :obj:`quantum_operator`
                :math:`H_{ij}\\mapsto H_{ij}^*`

        Examples
        --------

        >>> H_conj = H.conj()

        """
        return self.conjugate()

    def getH(self, copy=False):
        """Calculates hermitian conjugate of `quantum_operator` quantum_operator.

        Parameters
        ----------
        copy : bool, optional
                Whether to return a deep copy of the original object. Default is `copy = False`.

        Returns
        -------
        :obj:`quantum_operator`
                :math:`H_{ij}\\mapsto H_{ij}^*`

        Examples
        --------

        >>> H_herm = H.getH()

        """
        return self.conjugate().transpose(copy=copy)

    def copy(self):
        """Returns a deep copy of `quantum_operator` object."""
        new_dict = {key: [op] for key, op in iteritems(self._quantum_operator)}
        return quantum_operator(
            new_dict, basis=self._basis, dtype=self._dtype, shape=self._shape, copy=True
        )

    def astype(self, dtype, copy=False, casting="unsafe"):
        """Changes data type of `quantum_operator` object.

        Parameters
        ----------
        dtype : 'type'
                The data type (e.g. numpy.float64) to cast the Hamiltonian with.

        Returns
        `quantum_operator`
                quantum_operator with altered data type.

        Examples
        --------
        >>> H_cpx=H.astype(np.complex128)

        """
        if dtype not in hamiltonian_core.supported_dtypes:
            raise ValueError(
                "quantum_operator can only be cast to floating point types"
            )

        new_dict = {
            key: [op.astype(dtype, copy=copy, casting=casting)]
            for key, op in iteritems(self._quantum_operator)
        }

        return quantum_operator(
            new_dict, basis=self._basis, dtype=dtype, shape=self._shape, copy=copy
        )

    ### lin-alg operations

    def diagonal(self, pars={}):
        """Returns diagonal of `quantum_operator` quantum_operator for parameters `pars`.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.

        Returns
        -------
        numpy.ndarray
                array containing the diagonal part of the operator :math:`diag_j = H_{jj}(\\lambda)`.

        Examples
        --------

        >>> H_diagonal = H.diagonal(pars=pars)

        """
        pars = self._check_scalar_pars(pars)
        diag = _np.zeros(self.Ns, dtype=self._dtype)
        for key, value in iteritems(self._quantum_operator):
            diag += pars[key] * value.diagonal()
        return diag

    def trace(self, pars={}):
        """Calculates trace of `quantum_operator` quantum_operator for parameters `pars`.

        Parameters
        ----------
        pars : dict, optional
                Dictionary with same `keys` as `input_dict` and coupling strengths as `values`. Any missing `keys`
                are assumed to be set to unity.

        Returns
        -------
        float
                Trace of quantum_operator :math:`\\sum_{j=1}^{Ns} H_{jj}(\\lambda)`.

        Examples
        --------

        >>> H_tr = H.trace(pars=pars)

        """
        pars = self._check_scalar_pars(pars)
        tr = 0.0
        for key, value in iteritems(self._quantum_operator):
            try:
                tr += pars[key] * value.trace()
            except AttributeError:
                tr += pars[key] * value.diagonal().sum()
        return tr

    def __str__(self):
        s = ""
        for key, op in iteritems(self._quantum_operator):
            s = s + ("{}:\n{}\n".format(key, op))

        return s

    def __repr__(self):
        return "<{} x {} quspin.operator.quantum_operator with {} operator(s)>".format(
            self.shape[0], self.shape[1], len(self._quantum_operator)
        )

    def __call__(self, **pars):
        pars = self._check_scalar_pars(pars)
        if self.is_dense:
            return self.todense(pars)
        else:
            return self.tocsr(pars)

    def __neg__(self):
        return self.__imul__(-1)

    def __iadd__(self, other):
        self._is_dense = self._is_dense or other._is_dense
        if isinstance(other, quantum_operator):
            for key, value in iteritems(other._quantum_operator):
                if key in self._quantum_operator:
                    self._quantum_operator[key] = self._quantum_operator[key] + value
                else:
                    self._quantum_operator[key] = value

                if _check_almost_zero(self._quantum_operator[key]):
                    self._quantum_operator.pop(key)

            self._update_matvecs()
            return self
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __add__(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)
        new += other
        return new

    def __isub__(self, other):
        self._is_dense = self._is_dense or other._is_dense
        if isinstance(other, quantum_operator):
            for key, value in iteritems(other._quantum_operator):
                if key in self._quantum_operator:
                    self._quantum_operator[key] = self._quantum_operator[key] - value
                else:
                    self._quantum_operator[key] = -value

                if _check_almost_zero(self._quantum_operator[key]):
                    self._quantum_operator.pop(key)

            self._update_matvecs()
            return self
        elif other == 0:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)
        new -= other
        return new

    def __imul__(self, other):
        if isinstance(other, quantum_operator):
            return NotImplemented
        elif not _np.isscalar(other):
            return NotImplemented
        else:
            for op in itervalues(self._quantum_operator):
                op *= other

            self._update_matvecs()
            return self

    def __mul__(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)
        new *= other
        return new

    def __idiv__(self, other):
        if isinstance(other, quantum_operator):
            return NotImplemented
        elif not _np.isscalar(other):
            return NotImplemented
        else:
            for op in itervalues(self._quantum_operator):
                op /= other
            self._update_matvecs()
            return self

    def __div__(self, other):
        result_dtype = _np.result_type(self._dtype, other.dtype)
        new = self.astype(result_dtype, copy=True)
        new /= other
        return new

    def _check_hamiltonian_pars(self, pars):

        if not isinstance(pars, dict):
            raise ValueError("expecing dictionary for parameters.")

        pars = dict(pars)

        extra = set(pars.keys()) - set(self._quantum_operator.keys())
        if extra:
            raise ValueError("unexpected couplings: {}".format(extra))

        missing = set(self._quantum_operator.keys()) - set(pars.keys())
        for key in missing:
            pars[key] = _np.array(1, dtype=_np.int32)

        for key, J in pars.items():
            if type(J) is tuple:
                if len(J) != 2:
                    raise ValueError(
                        "expecting parameters to be either scalar or tuple of function and arguements of function."
                    )
            else:
                J = _np.array(J)
                if J.ndim > 0:
                    raise ValueError(
                        "expecting parameters to be either scalar or tuple of function and arguements of function."
                    )

        return pars

    def _check_scalar_pars(self, pars):

        if not isinstance(pars, dict):
            raise ValueError("expecing dictionary for parameters.")

        pars = dict(pars)

        extra = set(pars.keys()) - set(self._quantum_operator.keys())

        if extra:
            raise ValueError("unexpected couplings: {}".format(extra))

        missing = set(self._quantum_operator.keys()) - set(pars.keys())
        for key in missing:
            pars[key] = 1.0

        return pars

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

    def _update_matvecs(self):
        self._matvec_functions = {}

        for key in self._quantum_operator.keys():
            self._matvec_functions[key] = _get_matvec_function(
                self._quantum_operator[key]
            )


def isquantum_operator(obj):
    """Checks if instance is object of `quantum_operator` class.

    Parameters
    ----------
    obj :
            Arbitraty python object.

    Returns
    -------
    bool
            Can be either of the following:

            * `True`: `obj` is an instance of `quantum_operator` class.
            * `False`: `obj` is NOT an instance of `quantum_operator` class.

    """
    return isinstance(obj, quantum_operator)


def save_zip(archive, op, save_basis=True):
    """Save a `quantum_operator` to a zip archive to be used later.

    Parameters
    ----------
    archive : str
            name of archive, including path.

    op : `quantum_operator` object
            operator which you would like to save to disk

    save_basis : bool
            flag which tells code whether to save `basis` attribute of `op`, if it has such an attribute.
            some basis objects may not be able to be pickled, therefore attempting to save them will fail
            if this is the case, then set this flag to be `False`.


    Notes
    -----
    In order to keep formatting consistent, this function will always overwrite any file with the same name `archive`.
    This means that you can not append data to an existing archive. If you would like to combine data, either
    construct everything together and save or combine different `quantum_oeprator` objects using the `+` operator in python.

    """
    if not isquantum_operator(op):
        raise ValueError("this function can only save quantum_operator objects")

    with TemporaryDirectory() as tmpdirname:
        with ZipFile(archive, "w") as arch:

            if save_basis and op._basis is not None:
                file = os.path.join(tmpdirname, "basis.pickle")
                with open(file, "wb") as IO:
                    pickle.dump(op._basis, IO)

                arch.write(file, "basis.pickle")

            for key, matrix in iteritems(op._quantum_operator):
                if _sp.isspmatrix(matrix):
                    filename = "sparse_" + key + ".npz"
                    file = os.path.join(tmpdirname, filename)
                    _sp.save_npz(file, matrix)
                else:
                    filename = "dense_" + key + ".npz"
                    file = os.path.join(tmpdirname, filename)
                    _np.savez_compressed(file, matrix=matrix)

                if filename in arch.namelist():
                    raise ValueError(
                        "duplicate operator key '{}'' entry in archive.".format(key)
                    )
                arch.write(file, arcname=filename)


def load_zip(archive):
    """Load quantum_operator object from a zip archive.

    Parameters
    ----------
    archive : str
            name of archive, including path.

    Returns
    -------
    operator:  `quantum_operator`
            an object with matrix data extracted from the archive.

    """
    ops_dict = {}
    dtype = None
    basis = None
    with ZipFile(archive, "r") as arch:
        for file in arch.namelist():
            if file == "basis.pickle":
                with arch.open(file) as basisfile:
                    basis = pickle.load(basisfile)

                continue

            elif "sparse" in file:
                key = file.replace("sparse_", "").replace(".npz", "")
                with arch.open(file) as matfile:
                    matrix = _sp.load_npz(matfile)
            else:
                key = file.replace("dense_", "").replace(".npz", "")
                with arch.open(file) as matfile:
                    f = _np.load(matfile)
                    matrix = f["matrix"]

            if dtype is None:
                dtype = matrix.dtype
            else:
                dtype = _np.result_type(dtype, matrix.dtype)

            ops_dict[key] = [matrix]

    return quantum_operator(ops_dict, dtype=dtype, copy=False, basis=basis)
