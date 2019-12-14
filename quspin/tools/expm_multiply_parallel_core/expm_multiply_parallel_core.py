from scipy.sparse.linalg import LinearOperator,onenormest,aslinearoperator
from .expm_multiply_parallel_wrapper import (_wrapper_expm_multiply,
	_wrapper_csr_trace,_wrapper_csr_1_norm)
from scipy.sparse.construct import eye
from scipy.sparse.linalg._expm_multiply import _fragment_3_1,_exact_1_norm
import scipy.sparse as _sp
import numpy as _np

class expm_multiply_parallel(object):
    """Implements `scipy.sparse.linalg.expm_multiply()` for *openmp*.

    Notes
    -----
    This is a wrapper over custom c++ code.

    Examples
    --------

    This example shows how to construct the `expm_multiply_parallel` object.

    Further code snippets can be found in the examples for the function methods of the class.
    The code snippet below initiates the class, and is required to run the example codes for the function methods.
    
    .. literalinclude:: ../../doc_examples/expm_multiply_parallel-example.py
        :linenos:
        :language: python
        :lines: 7-30
    
    """
    def __init__(self,A,a=1.0,dtype=None,copy=False):
        """Initializes `expm_multiply_parallel`. 

        Parameters
        -----------
        A : {array_like, scipy.sparse matrix}
            The operator (matrix) whose exponential is to be calculated.
        a : scalar, optional
            scalar value multiplying generator matrix :math:`A` in matrix exponential: :math:`\\mathrm{e}^{aA}`.
        dtype: numpy.dtype, optional
            dtype specified for this operator. Default is: result_type(A.dtype,min_scalar_type(a),float32)
        copy: bool, optional
            if `True` the matrix is copied otherwise the matrix is stored by reference. 

        Note
        ----
        The `dtype` need not be the same dtype of `A` or `a`, however it must be possible to cast the result of a*A to this dtype. 

        """
        if _np.array(a).ndim == 0:
            self._a = a
        else:
            raise ValueError("a must be scalar value.")

        self._A = _sp.csr_matrix(A,copy=copy)

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix.")

        a_dtype_min = _np.min_scalar_type(self._a)
        min_dtype = _np.result_type(A.dtype,a_dtype_min,_np.float32)

        if dtype is None:
            self._dtype = min_dtype
        else:
            if not _np.can_cast(min_dtype,dtype):
                raise ValueError("dtype not sufficient to represent a*A to at least float32 precision.")

            self._dtype = dtype

        tol = _np.finfo(self._dtype).eps/2
        tol_dtype = _np.finfo(self._dtype).eps.dtype
        self._tol = _np.array(tol,dtype=tol_dtype)

        mu = _wrapper_csr_trace(self._A.indptr,self._A.indices,self._A.data)/self._A.shape[0]
        self._mu = _np.array(mu,dtype=self._dtype)
        self._A_1_norm = _wrapper_csr_1_norm(self._A.indptr,self._A.indices,self._A.data,self._mu)
        self._calculate_partition()

        # shift = eye(A.shape[0],format="csr",dtype=A.dtype)
        # shift.data *= mu
        # self._A = self._A - shift


    @property
    def a(self):
        """scalar: value multiplying generator matrix :math:`A` in matrix exponential: :math:`\\mathrm{e}^{aA}`"""
        return self._a

    @property
    def A(self):
        """scipy.sparse.csr_matrix: csr_matrix to be exponentiated."""
        return self._A


    def set_a(self,a):
        """Sets the value of the property `a`.

        Examples
        --------

        .. literalinclude:: ../../doc_examples/expm_multiply_parallel-example.py
            :linenos:
            :language: python
            :lines: 32-35
            
        Parameters
        -----------
        a : scalar
            new value of `a`.

        """

        if _np.array(a).ndim == 0:
            self._a = a
            self._calculate_partition()
        else:
            raise ValueError("expecting 'a' to be scalar.")

    def dot(self,v,work_array=None,overwrite_v=False):
        """Calculates the action of :math:`\\mathrm{e}^{aA}` on a vector :math:`v`. 

        Examples
        --------

        .. literalinclude:: ../../doc_examples/expm_multiply_parallel-example.py
            :linenos:
            :language: python
            :lines: 37-

        Parameters
        -----------
        v : contiguous numpy.ndarray
            array to apply :math:`\\mathrm{e}^{aA}` on.
        work_array : contiguous numpy.ndarray, optional
            array of `shape = (2*len(v),)` which is used as work_array space for the underlying c-code. This saves extra memory allocation for function operations.
        overwrite_v : bool
            if set to `True`, the data in `v` is overwritten by the function. This saves extra memory allocation for the results.

        Returns
        --------
        numpy.ndarray
            result of :math:`\\mathrm{e}^{aA}v`. 

            If `overwrite_v = True` the dunction returns `v` with the data overwritten, otherwise the result is stored in a new array.  

        """
        v = _np.asarray(v)
            
        if v.ndim != 1:
            raise ValueError("array must have ndim of 1.")
        
        if v.shape[0] != self._A.shape[1]:
            raise ValueError("dimension mismatch {}, {}".format(self._A.shape,v.shape))



        v_dtype = _np.result_type(self._dtype,v.dtype)


        if overwrite_v:
            if v_dtype != v.dtype:
                raise ValueError("if overwrite_v is True, the input array must match correct output dtype for matrix multiplication.")

            if not v.flags["CARRAY"]:
                raise TypeError("input array must a contiguous and writable.")

            if v.ndim != 1:
                raise ValueError("array must have ndim of 1.")
        else:
            v = v.astype(v_dtype,order="C",copy=True)

        if work_array is None:
            work_array = _np.zeros((2*self._A.shape[0],),dtype=v.dtype)
        else:
            work_array = _np.ascontiguousarray(work_array)
            if work_array.shape != (2*self._A.shape[0],):
                raise ValueError("work_array array must be an array of shape (2*v.shape[0],) with same dtype as v.")
            if work_array.dtype != v_dtype:
                raise ValueError("work_array must be array of dtype which matches the result of the matrix-vector multiplication.")

        a = _np.array(self._a,dtype=v_dtype)
        mu = _np.array(self._mu,dtype=v_dtype)
        tol = _np.array(self._tol,dtype=mu.real.dtype)
        _wrapper_expm_multiply(self._A.indptr,self._A.indices,self._A.data,
                    self._s,self._m_star,a,tol,mu,v,work_array)

        return v

    def _calculate_partition(self):
        if _np.abs(self._a)*self._A_1_norm == 0:
            self._m_star, self._s = 0, 1
        else:
            ell = 2
            norm_info = LazyOperatorNormInfo(self._A, self._A_1_norm, self._a, self._mu, self._dtype, ell=ell)
            self._m_star, self._s = _fragment_3_1(norm_info, 1, self._tol, ell=ell)


##### code below is copied from scipy.sparse.linalg._expm_multiply_core and modified slightly.


def matvec_p(v,A,a,mu,p):
    for i in range(p):
        v = a * (A.dot(v) - mu*v)

    return v


class LazyOperatorNormInfo:
    """
    Information about an operator is lazily computed.

    The information includes the exact 1-norm of the operator,
    in addition to estimates of 1-norms of powers of the operator.
    This uses the notation of Computing the Action (2011).
    This class is specialized enough to probably not be of general interest
    outside of this module.

    """
    def __init__(self, A, A_1_norm, a, mu, dtype, ell=2):
        """
        Provide the operator and some norm-related information.

        Parameters
        -----------
        A : linear operator
            The operator of interest.
        A_1_norm : float
            The exact 1-norm of A.
        ell : int, optional
            A technical parameter controlling norm estimation quality.

        """
        self._A = A
        self._a = a
        self._mu = mu
        self._dtype = dtype
        self._A_1_norm = A_1_norm
        self._ell = ell
        self._d = {}

    def onenorm(self):
        """
        Compute the exact 1-norm.
        """
        return _np.abs(self._a) * self._A_1_norm

    def d(self, p):
        """
        Lazily estimate d_p(A) ~= || A^p ||^(1/p) where ||.|| is the 1-norm.
        """
        if p not in self._d:
            matvec = lambda v: self._a * (self._A.dot(v) - self._mu*v)
            rmatvec = lambda v: self._a * (self._A.H.dot(v) - self._mu*v)
            LO = LinearOperator(self._A.shape,dtype=self._dtype,matvec=matvec,rmatvec=rmatvec)

            est = onenormest(LO**p)

            # est = onenormest((self._a * aslinearoperator(self._A))**p)
            self._d[p] = est ** (1.0 / p)

        return self._d[p]

    def alpha(self, p):
        """
        Lazily compute max(d(p), d(p+1)).
        """
        return max(self.d(p), self.d(p+1))


