# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse as _sp
import warnings
import numpy as _np



def _consolidate_bonds(bonds):
	l = len(bonds)
	i=0
	while(i < l):
		j = 0
		while(j < l):
			if i != j:
				if bonds[i][1:] == bonds[j][1:]:
					bonds[i][0] += bonds[j][0]
					del bonds[j]
					if bonds[i][0] == 0:
						del bonds[i]

					l = len(bonds)
			j += 1
		i += 1
					



def _consolidate_static(static_list):
	l = len(static_list)
	i=0
	while(i < l):
		j = 0
		while(j < l):
			if i != j:
				opstr1,bonds1 = tuple(static_list[i])
				opstr2,bonds2 = tuple(static_list[j])
				if opstr1 == opstr2:
					del static_list[j]
					static_list[i][1].extend(bonds2)
					_consolidate_bonds(static_list[i][1])
					l = len(static_list)
			j += 1
		i += 1


def _consolidate_dynamic(dynamic_list):
	l = len(dynamic_list)
	i = 0

	while(i < l):
		j = 0
		while(j < l):
			if i != j:
				opstr1,bonds1,f1,f1_args = tuple(dynamic_list[i])
				opstr2,bonds2,f2,f2_args = tuple(dynamic_list[j])
				if (opstr1 == opstr2) and (f1 == f2) and (f1_args == f2_args):
					del dynamic_list[j]
					dynamic_list[i][1].extend(bonds2)
					_consolidate_bonds(dynamic_list[i][1])
					l = len(dynamic_list)
			j += 1
		i += 1



def test_function(func,func_args):
	t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )
	func_val=func(t,*func_args)
	if not _np.isscalar(func_val):
		raise TypeError("function must return scaler values")




def make_static(basis,static_list,dtype):
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
	Ns=basis.Ns
	H = _sp.csr_matrix((Ns,Ns),dtype=dtype)
	_consolidate_static(static_list)
	for opstr,bonds in static_list:
		for bond in bonds:
			J=bond[0]
			indx=bond[1:]
#			indx = _np.asarray(indx,_np.int32)
			ME,row,col = basis.Op(opstr,indx,J,dtype)
			Ht=_sp.csr_matrix((ME,(row,col)),shape=(Ns,Ns),dtype=dtype) 
			H=H+Ht
			del Ht
			H.sum_duplicates() # sum duplicate matrix elements
			H.eliminate_zeros() # remove all zero matrix elements

	return H 





def make_dynamic(basis,dynamic_list,dtype):
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
	Ns=basis.Ns
	dynamic=[]
	_consolidate_dynamic(dynamic_list)
	if dynamic_list:
		for opstr,bonds,f,f_args in dynamic_list:
			H=_sp.csr_matrix(([],([],[])),shape=(Ns,Ns),dtype=dtype)
			if _np.isscalar(f_args): raise TypeError("function arguments must be array type")
			test_function(f,f_args)
			for bond in bonds:
				J=bond[0]
				indx=bond[1:]
#				indx = _np.asarray(indx,_np.int32)
				ME,row,col = basis.Op(opstr,indx,J,dtype)
				Ht=_sp.csr_matrix((ME,(row,col)),shape=(Ns,Ns),dtype=dtype) 
				H=H+Ht
				del Ht
				H.sum_duplicates() # sum duplicate matrix elements
				H.eliminate_zeros() # remove all zero matrix elements
			dynamic.append((H,f,f_args))
	return tuple(dynamic)
