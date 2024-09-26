import numpy as _np
import scipy.sparse as _sp
from quspin_extensions.basis._basis_utils import _shuffle_sites


####################################################
# set of helper functions to implement the partial #
# trace of lattice density matrices. They do not   #
# have any checks and states are assumed to be     #
# in the non-symmetry reduced basis.               #
####################################################


def _lattice_partial_trace_pure(psi, sub_sys_A, L, sps, return_rdm="A"):
    """
    This function computes the partial trace of a dense pure state psi over set of sites sub_sys_A and returns
    reduced DM. Vectorisation available.
    """

    psi_v = _lattice_reshape_pure(psi, sub_sys_A, L, sps)

    if return_rdm == "A":
        return _np.squeeze(_np.einsum("...ij,...kj->...ik", psi_v, psi_v.conj())), None
    elif return_rdm == "B":
        return None, _np.squeeze(_np.einsum("...ji,...jk->...ik", psi_v.conj(), psi_v))
    elif return_rdm == "both":
        return _np.squeeze(
            _np.einsum("...ij,...kj->...ik", psi_v, psi_v.conj())
        ), _np.squeeze(_np.einsum("...ji,...jk->...ik", psi_v.conj(), psi_v))


def _lattice_partial_trace_mixed(rho, sub_sys_A, L, sps, return_rdm="A"):
    """
    This function computes the partial trace of a set of dense mixed states rho over set of sites sub_sys_A
    and returns reduced DM. Vectorisation available.
    """
    rho_v = _lattice_reshape_mixed(rho, sub_sys_A, L, sps)
    if return_rdm == "A":
        return _np.einsum("...jlkl->...jk", rho_v), None
    elif return_rdm == "B":
        return None, _np.einsum("...ljlk->...jk", rho_v.conj())
    elif return_rdm == "both":
        return _np.einsum("...jlkl->...jk", rho_v), _np.einsum(
            "...ljlk->...jk", rho_v.conj()
        )


def _lattice_partial_trace_sparse_pure(psi, sub_sys_A, L, sps, return_rdm="A"):
    """
    This function computes the partial trace of a sparse pure state psi over set of sites sub_sys_A and returns
    reduced DM.
    """
    psi = _lattice_reshape_sparse_pure(psi, sub_sys_A, L, sps)

    if return_rdm == "A":
        return psi.dot(psi.T.conj()), None
    elif return_rdm == "B":
        return None, psi.T.conj().dot(psi)
    elif return_rdm == "both":
        return psi.dot(psi.T.conj()), psi.T.conj().dot(psi)


def _lattice_reshape_pure(psi, sub_sys_A, L, sps):
    """
    This function reshapes the dense pure state psi over the Hilbert space defined by sub_sys_A and its complement.
    Vectorisation available.
    """
    extra_dims = psi.shape[:-1]
    n_dims = len(extra_dims)
    sub_sys_B = set(range(L)) - set(sub_sys_A)

    sub_sys_A = tuple(sub_sys_A)
    sub_sys_B = tuple(sub_sys_B)

    L_A = len(sub_sys_A)
    L_B = len(sub_sys_B)

    Ns_A = sps**L_A
    Ns_B = sps**L_B
    T_tup = sub_sys_A + sub_sys_B
    psi_v = _shuffle_sites(sps, T_tup, psi)
    psi_v = psi_v.reshape(extra_dims + (Ns_A, Ns_B))

    return psi_v


'''
def _lattice_reshape_pure(psi,sub_sys_A,L,sps):
	"""
	This function reshapes the dense pure state psi over the Hilbert space defined by sub_sys_A and its complement. 
	Vectorisation available. 
	"""
	extra_dims = psi.shape[:-1]
	n_dims = len(extra_dims)
	sub_sys_B = set(range(L))-set(sub_sys_A)

	sub_sys_A = tuple(sub_sys_A)
	sub_sys_B = tuple(sub_sys_B)

	L_A = len(sub_sys_A)
	L_B = len(sub_sys_B)

	Ns_A = (sps**L_A)
	Ns_B = (sps**L_B)

	T_tup = sub_sys_A+sub_sys_B
	T_tup = tuple(range(n_dims)) + tuple(n_dims + s for s in T_tup)
	R_tup = extra_dims + tuple(sps for i in range(L))
	psi_v = psi.reshape(R_tup) # DM where index is given per site as rho_v[i_1,...,i_L,j_1,...j_L]
	psi_v = psi_v.transpose(T_tup) # take transpose to reshuffle indices
	psi_v = psi_v.reshape(extra_dims+(Ns_A,Ns_B))
	return psi_v
'''


def _lattice_reshape_mixed(rho, sub_sys_A, L, sps):
    """
    This function reshapes the dense mixed state psi over the Hilbert space defined by sub_sys_A and its complement.
    Vectorisation available.
    """
    extra_dims = rho.shape[:-2]
    n_dims = len(extra_dims)
    sub_sys_B = set(range(L)) - set(sub_sys_A)

    sub_sys_A = tuple(sub_sys_A)
    sub_sys_B = tuple(sub_sys_B)

    L_A = len(sub_sys_A)
    L_B = len(sub_sys_B)

    Ns_A = sps**L_A
    Ns_B = sps**L_B

    # T_tup tells numpy how to reshuffle the indices such that when I reshape the array to the
    # 4-_tensor rho_{ik,jl} i,j are for sub_sys_A and k,l are for sub_sys_B
    # which means I need (sub_sys_A,sub_sys_B,sub_sys_A+L,sub_sys_B+L)

    T_tup = sub_sys_A + sub_sys_B
    T_tup = tuple(T_tup) + tuple(L + s for s in T_tup)
    rho = rho.reshape(extra_dims + (-1,))
    rho_v = _shuffle_sites(sps, T_tup, rho)

    return rho_v.reshape(extra_dims + (Ns_A, Ns_B, Ns_A, Ns_B))


'''
def _lattice_reshape_mixed(rho,sub_sys_A,L,sps):
	"""
	This function reshapes the dense mixed state psi over the Hilbert space defined by sub_sys_A and its complement.
	Vectorisation available. 
	"""
	extra_dims = rho.shape[:-2]
	n_dims = len(extra_dims)
	sub_sys_B = set(range(L))-set(sub_sys_A)

	sub_sys_A = tuple(sub_sys_A)
	sub_sys_B = tuple(sub_sys_B)

	L_A = len(sub_sys_A)
	L_B = len(sub_sys_B)

	Ns_A = (sps**L_A)
	Ns_B = (sps**L_B)

	# T_tup tells numpy how to reshuffle the indices such that when I reshape the array to the 
	# 4-_tensor rho_{ik,jl} i,j are for sub_sys_A and k,l are for sub_sys_B
	# which means I need (sub_sys_A,sub_sys_B,sub_sys_A+L,sub_sys_B+L)

	T_tup = sub_sys_A+sub_sys_B
	T_tup = tuple(range(n_dims)) + tuple(s+n_dims for s in T_tup) + tuple(L+n_dims+s for s in T_tup)

	R_tup = extra_dims + tuple(sps for i in range(2*L))

	rho_v = rho.reshape(R_tup) # DM where index is given per site as rho_v[i_1,...,i_L,j_1,...j_L]
	rho_v = rho_v.transpose(T_tup) # take transpose to reshuffle indices
	
	return rho_v.reshape(extra_dims+(Ns_A,Ns_B,Ns_A,Ns_B))

'''


def _lattice_reshape_sparse_pure(psi, sub_sys_A, L, sps):
    """
    This function reshapes the sparse pure state psi over the Hilbert space defined by sub_sys_A and its complement.
    """
    sub_sys_B = set(range(L)) - set(sub_sys_A)

    sub_sys_A = tuple(sub_sys_A)
    sub_sys_B = tuple(sub_sys_B)

    L_A = len(sub_sys_A)
    L_B = len(sub_sys_B)

    Ns_A = sps**L_A
    Ns_B = sps**L_B
    psi = psi.tocoo()

    T_tup = sub_sys_A + sub_sys_B
    # reshuffle indices for the sub-systems.
    # j = sum( j[i]*(sps**i) for i in range(L))
    # this reshuffles the j[i] similar to the transpose operation
    # on the dense arrays psi_v.transpose(T_tup)

    if T_tup != tuple(range(L)):
        indx = _np.zeros(psi.col.shape, dtype=psi.col.dtype)
        for i_old, i_new in enumerate(T_tup):
            indx += ((psi.col // (sps ** (L - i_new - 1))) % sps) * (
                sps ** (L - i_old - 1)
            )
    else:
        indx = psi.col

    # A = _np.array([0,1,2,3,4,5,6,7,8,9,10,11])
    # print("make shift way of reshaping array")
    # print("A = {}".format(A))
    # print("A.reshape((3,4)): \n {}".format(A.reshape((3,4))))
    # print("rows: A.reshape((3,4))/4: \n {}".format(A.reshape((3,4))/4))
    # print("cols: A.reshape((3,4))%4: \n {}".format(A.reshape((3,4))%4))

    psi._shape = (Ns_A, Ns_B)
    psi.row[:] = indx / Ns_B
    psi.col[:] = indx % Ns_B

    return psi.tocsr()


def _tensor_reshape_pure(psi, sub_sys_A, Ns_l, Ns_r):
    extra_dims = psi.shape[:-1]
    if sub_sys_A == "left":
        return psi.reshape(extra_dims + (Ns_l, Ns_r))
    else:
        n_dims = len(extra_dims)
        T_tup = tuple(range(n_dims)) + (n_dims + 1, n_dims)
        psi_v = psi.reshape(extra_dims + (Ns_l, Ns_r))
        return psi_v.transpose(T_tup)


def _tensor_reshape_sparse_pure(psi, sub_sys_A, Ns_l, Ns_r):
    psi = psi.tocoo()
    # make shift way of reshaping array
    # j = j_l + Ns_r * j_l
    # j_l = j / Ns_r
    # j_r = j % Ns_r
    if sub_sys_A == "left":
        psi._shape = (Ns_l, Ns_r)
        psi.row[:] = psi.col / Ns_r
        psi.col[:] = psi.col % Ns_r
        return psi.tocsr()
    else:
        psi._shape = (Ns_l, Ns_r)
        psi.row[:] = psi.col / Ns_r
        psi.col[:] = psi.col % Ns_r
        return psi.T.tocsr()


def _tensor_reshape_mixed(rho, sub_sys_A, Ns_l, Ns_r):
    extra_dims = rho.shape[:-2]
    if sub_sys_A == "left":
        return rho.reshape(extra_dims + (Ns_l, Ns_r, Ns_l, Ns_r))
    else:
        n_dims = len(extra_dims)
        T_tup = tuple(range(n_dims)) + (n_dims + 1, n_dims) + (n_dims + 3, n_dims + 2)
        rho_v = rho.reshape(extra_dims + (Ns_l, Ns_r, Ns_l, Ns_r))
        return rho_v.transpose(T_tup)


def _tensor_partial_trace_pure(psi, sub_sys_A, Ns_l, Ns_r, return_rdm="A"):
    psi_v = _tensor_reshape_pure(psi, sub_sys_A, Ns_l, Ns_r)

    if return_rdm == "A":
        return _np.squeeze(_np.einsum("...ij,...kj->...ik", psi_v, psi_v.conj())), None
    elif return_rdm == "B":
        return None, _np.squeeze(_np.einsum("...ji,...jk->...ik", psi_v.conj(), psi_v))
    elif return_rdm == "both":
        return _np.squeeze(
            _np.einsum("...ij,...kj->...ik", psi_v, psi_v.conj())
        ), _np.squeeze(_np.einsum("...ji,...jk->...ik", psi_v.conj(), psi_v))


def _tensor_partial_trace_sparse_pure(psi, sub_sys_A, Ns_l, Ns_r, return_rdm="A"):
    psi = _tensor_reshape_sparse_pure(psi, sub_sys_A, Ns_l, Ns_r)

    if return_rdm == "A":
        return psi.dot(psi.T.conj()), None
    elif return_rdm == "B":
        return None, psi.T.conj().dot(psi)
    elif return_rdm == "both":
        return psi.dot(psi.T.conj()), psi.T.conj().dot(psi)


def _tensor_partial_trace_mixed(rho, sub_sys_A, Ns_l, Ns_r, return_rdm="A"):
    rho_v = _tensor_reshape_mixed(rho, sub_sys_A, Ns_l, Ns_r)
    if return_rdm == "A":
        return _np.squeeze(_np.einsum("...ijkj->...ik", rho_v)), None
    elif return_rdm == "B":
        return None, _np.squeeze(_np.einsum("...jijk->...ik", rho_v.conj()))
    elif return_rdm == "both":
        return _np.squeeze(_np.einsum("...ijkj->...ik", rho_v)), _np.squeeze(
            _np.einsum("...jijk->...ik", rho_v.conj())
        )
