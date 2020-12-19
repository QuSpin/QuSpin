from scipy.sparse.linalg import LinearOperator,onenormest,aslinearoperator
from .expm_multiply_parallel_wrapper import (_wrapper_expm_multiply,
	_wrapper_expm_multiply_batch,_wrapper_csr_trace,_wrapper_csr_1_norm)
from scipy.sparse.construct import eye
from scipy.sparse.linalg._expm_multiply import _fragment_3_1,_exact_1_norm
import scipy.sparse as _sp
import numpy as _np

class expm_multiply_parallel(object):
    """Implements `scipy.sparse.linalg.expm_multiply()` for *openmp*.

    Notes
    -----
    * this is a wrapper over custom c++ code.
    * the `dtype` input need not be the same dtype as `A` or `a`; however, it must be possible to cast the result of `a*A` to this `dtype`. 
    * consider the special case of real-time evolution with a purely-imaginary Hamiltonian, in which case `a=-1j*time` and `A` are both complex-valued, while the resulting matrix exponential is real-valued: in such cases, one can use either one of
    
    >>> expm_multiply_parallel( (1j*H.tocsr()).astype(np.float64), a=-1.0, dtype=np.float64)`
    
    and
    
    >>> expm_multiply_parallel( H.tocsr(), a=-1.0j, dtype=np.complex128)
    
    The more efficient way to compute the matrix exponential in this case is to use a real-valued `dtype`. 


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
        dtype : numpy.dtype, optional
            data type specified for the total operator :math:`\\mathrm{e}^{aA}`. Default is: `numpy.result_type(A.dtype,min_scalar_type(a),float64)`.
        copy : bool, optional
            if `True` the matrix is copied otherwise the matrix is stored by reference. 

        """
        if _np.array(a).ndim == 0:
            self._a = a
        else:
            raise ValueError("a must be scalar value.")

        self._A = _sp.csr_matrix(A,copy=copy)

        if A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square matrix.")

        a_dtype_min = _np.min_scalar_type(self._a)

        # use double precision by default. 
        if dtype is None:
            self._dtype = _np.result_type(A.dtype,a_dtype_min,_np.float64)
        else:
            min_dtype = _np.result_type(A.dtype,a_dtype_min,_np.float32)
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


    def set_a(self,a,dtype=None):
        """Sets the value of the property `a`.

        Parameters
        ----------
        a : scalar
            new value of `a`.
        dtype : numpy.dtype, optional
            dtype specified for this operator. Default is: result_type(A.dtype,min_scalar_type(a),float64)

        Examples
        --------

        .. literalinclude:: ../../doc_examples/expm_multiply_parallel-example.py
            :linenos:
            :language: python
            :lines: 32-35
            
        """

        if _np.array(a).ndim == 0:
            self._a = a

            a_dtype_min = _np.min_scalar_type(self._a)

            # use double precision by default. 
            if dtype is None:
                self._dtype = _np.result_type(self._A.dtype,a_dtype_min,_np.float64)
            else:
                min_dtype = _np.result_type(A.dtype,a_dtype_min,_np.float32)
                if not _np.can_cast(min_dtype,dtype):
                    raise ValueError("dtype not sufficient to represent a*A to at least float32 precision.")

                self._dtype = dtype

            tol = _np.finfo(self._dtype).eps/2
            tol_dtype = _np.finfo(self._dtype).eps.dtype
            self._tol = _np.array(tol,dtype=tol_dtype)
            self._mu = _np.array(self._mu,dtype=self._dtype)

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
        v : contiguous numpy.ndarray, 1d or 2d array
            array to apply :math:`\\mathrm{e}^{aA}` on.
        work_array : contiguous numpy.ndarray, optional
            array can be any shape but must contain 2*v.size contiguous elements. 
            This array is used as temporary memory space for the underlying c-code. This saves extra memory allocation for function operations.
        overwrite_v : bool
            if set to `True`, the data in `v` is overwritten by the function. This saves extra memory allocation for the results.

        Returns
        --------
        numpy.ndarray
            result of :math:`\\mathrm{e}^{aA}v`. 

            If `overwrite_v = True` the dunction returns `v` with the data overwritten, otherwise the result is stored in a new array.  

        """
        v = _np.asarray(v)
            
        if v.ndim > 2:
            raise ValueError("array must have ndim <= 2.")
        
        if v.shape[0] != self._A.shape[1]:
            raise ValueError("dimension mismatch {}, {}".format(self._A.shape,v.shape))



        v_dtype = _np.result_type(self._dtype,v.dtype)


        if overwrite_v:
            if v_dtype != v.dtype:
                raise ValueError("if overwrite_v is True, the input array must match correct output dtype for matrix multiplication.")

            if not v.flags["CARRAY"]:
                raise TypeError("input array must a contiguous and writable.")

        else:
            v = v.astype(v_dtype,order="C",copy=True)

        if work_array is None:
            if v.ndim == 1:
                work_array = _np.zeros((2*self._A.shape[0],),dtype=v.dtype)
            else:
                work_array = _np.zeros((2*self._A.shape[0],v.shape[1]),dtype=v.dtype)
        else:
            work_array = work_array.ravel(order="A")

            if work_array.size == 2*v.size:
                raise ValueError("work_array must have twice the number of elements as in v.")
        
            if work_array.dtype != v_dtype:
                raise ValueError("work_array must be array of dtype which matches the result of the matrix-vector multiplication.")



        a = _np.array(self._a,dtype=v_dtype)
        mu = _np.array(self._mu,dtype=v_dtype)
        tol = _np.array(self._tol,dtype=mu.real.dtype)
        if v.ndim == 1:
            _wrapper_expm_multiply(self._A.indptr,self._A.indices,self._A.data,
                        self._s,self._m_star,a,tol,mu,v,work_array.ravel())
        else:
            work_array = work_array.reshape((-1,v.shape[1]))
            _wrapper_expm_multiply_batch(self._A.indptr,self._A.indices,self._A.data,
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
            rmatvec = lambda v: _np.conj(self._a) * (self._A.H.dot(v) - _np.conj(self._mu)*v)
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


