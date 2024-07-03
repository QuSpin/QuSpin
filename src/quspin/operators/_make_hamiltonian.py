# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse as _sp
import warnings
import numpy as _np
from quspin.operators._functions import function
from six import iteritems


def _check_almost_zero(matrix):
    """Check if matrix is almost zero."""
    atol = 100 * _np.finfo(matrix.dtype).eps

    if _sp.issparse(matrix):
        return _np.allclose(matrix.data, 0, atol=atol)
    else:
        return _np.allclose(matrix, 0, atol=atol)


def _consolidate_static(static_list):
    eps = 10 * _np.finfo(_np.float64).eps

    static_dict = {}
    for opstr, bonds in static_list:
        if opstr not in static_dict:
            static_dict[opstr] = {}

        for bond in bonds:
            J = bond[0]
            indx = tuple(bond[1:])
            if indx in static_dict[opstr]:
                static_dict[opstr][indx] += J
            else:
                static_dict[opstr][indx] = J

    static_list = []
    for opstr, opstr_dict in static_dict.items():
        for indx, J in opstr_dict.items():
            if _np.abs(J) > eps:
                static_list.append((opstr, indx, J))

    return static_list


def _consolidate_dynamic(dynamic_list):
    eps = 10 * _np.finfo(_np.float64).eps

    dynamic_dict = {}
    for opstr, bonds, f, f_args in dynamic_list:
        f_args = tuple(f_args)
        if (opstr, f, f_args) not in dynamic_dict:
            dynamic_dict[(opstr, f, f_args)] = {}

        for bond in bonds:
            J = bond[0]
            indx = tuple(bond[1:])
            if indx in dynamic_dict[(opstr, f, f_args)]:
                dynamic_dict[(opstr, f, f_args)][indx] += J
            else:
                dynamic_dict[(opstr, f, f_args)][indx] = J

    dynamic_list = {}
    for (opstr, f, f_args), opstr_dict in dynamic_dict.items():
        func = (f, f_args)

        if func not in dynamic_list:
            dynamic_list[func] = []

        dynamic_list[func].extend(
            [(opstr, indx, J) for indx, J in iteritems(opstr_dict) if _np.abs(J) > eps]
        )

    return dynamic_list


def test_function(func, func_args, dtype):
    t = _np.cos((_np.pi / _np.exp(0)) ** (1.0 / _np.euler_gamma))
    func_val = func(t, *func_args)
    func_val = _np.array(func_val, dtype=dtype)
    if func_val.ndim > 0:
        raise ValueError("function must return 0-dim numpy array or scalar value.")


def make_static(basis, static_list, dtype):
    """
    args:
            static=[[opstr_1,indx_1],...,[opstr_n,indx_n]], list of opstr,indx to add up for static piece of Hamiltonian.
            dtype = the low level C-type which the matrix should store its values with.
    returns:
            H: a csr_matrix representation of the list static

    description:
            this function takes the list static and creates a list of matrix elements is coordinate format. it does
            this by calling the basis method Op which takes a state in the basis, acts with opstr and returns a matrix
            element and the state which it is connected to. This function is called for every opstr in list static and for every
            state in the basis until the entire hamiltonian is mapped out. It takes those matrix elements (which need not be
            sorted or even unique) and creates a coo_matrix from the scipy.sparse library. It then converts this coo_matrix
            to a csr_matrix class which has optimal sparse matrix vector multiplication.
    """
    static_list = _consolidate_static(static_list)
    return basis._make_matrix(static_list, dtype)


def make_dynamic(basis, dynamic_list, dtype):
    """
    args:
    dynamic=[[opstr_1,indx_1,func_1,func_1_args],...,[opstr_n,indx_n,func_n,func_n_args]], list of opstr,indx and functions to drive with
    dtype = the low level C-type which the matrix should store its values with.

    returns:
    tuple((func_1,func_1_args,H_1),...,(func_n_func_n_args,H_n))

    H_i: a csr_matrix representation of opstr_i,indx_i
    func_i: callable function of time which is the drive term in front of H_i

    description:
            This function works the same as static, but instead of adding all of the elements
            of the dynamic list together, it returns a tuple which contains each individual csr_matrix
            representation of all the different driven parts. This way one can construct the time dependent
            Hamiltonian simply by looping over the tuple returned by this function.
    """
    Ns = basis.Ns
    dynamic = {}
    dynamic_list = _consolidate_dynamic(dynamic_list)
    for (f, f_args), ops_list in iteritems(dynamic_list):
        if _np.isscalar(f_args):
            raise TypeError("function arguments must be array type")
        test_function(f, f_args, dtype)

        func = function(f, f_args)
        Hd = basis._make_matrix(ops_list, dtype)
        if not _check_almost_zero(Hd):
            dynamic[func] = Hd
    return dynamic
