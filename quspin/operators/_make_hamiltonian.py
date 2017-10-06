from __future__ import print_function, division

# the final version the sparse matrices are stored as, good format for dot produces with vectors.
import scipy.sparse as _sp
import warnings
import numpy as _np
from ._functions import function




def _consolidate_static(static_list):
	eps = 10 * _np.finfo(_np.float64).eps

	static_dict={}
	for opstr,bonds in static_list:
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
	for opstr,opstr_dict in static_dict.items():
		for indx,J in opstr_dict.items():
			if _np.abs(J) > eps:
				static_list.append((opstr,indx,J))


	return static_list


def _consolidate_dynamic(dynamic_list):
	eps = 10 * _np.finfo(_np.float64).eps
	
	dynamic_dict={}
	for opstr,bonds,f,f_args in dynamic_list:
		f_args = tuple(f_args)
		if (opstr,f,f_args) not in dynamic_dict:
			dynamic_dict[(opstr,f,f_args)] = {}

		for bond in bonds:
			J = bond[0]
			indx = tuple(bond[1:])
			if indx in dynamic_dict[(opstr,f,f_args)]:
				dynamic_dict[(opstr,f,f_args)][indx] += J
			else:
				dynamic_dict[(opstr,f,f_args)][indx] = J


	dynamic_list = []
	for (opstr,f,f_args),opstr_dict in dynamic_dict.items():
		for indx,J in opstr_dict.items():
			if _np.abs(J) > eps:
				dynamic_list.append((opstr,indx,J,f,f_args))


	return dynamic_list



# def _consolidate_bonds(bonds):
# 	eps = _np.finfo(_np.float64).eps
# 	l = len(bonds)
# 	i=0
# 	while(i < l):
# 		j = 0
# 		while(j < l):
# 			if i != j:
# 				if bonds[i][1:] == bonds[j][1:]:
# 					bonds[i][0] += bonds[j][0]
# 					del bonds[j]
# 					if j < i: i -= 1
# 					l = len(bonds)
# 			j += 1
# 		i += 1


# 	i=0
# 	while(i < l):
# 		if abs(bonds[i][0]) < 10 * eps:
# 			del bonds[i]
# 			l = len(bonds)
# 			continue

# 		i += 1
					



# def _consolidate_static(static_list):
# 	l = len(static_list)
# 	i=0
# 	while(i < l):
# 		j = 0
# 		while(j < l):
# 			if i != j:
# 				opstr1,bonds1 = tuple(static_list[i])
# 				opstr2,bonds2 = tuple(static_list[j])
# 				if opstr1 == opstr2:
# 					del static_list[j]
# 					static_list[i][1].extend(bonds2)
# 					_consolidate_bonds(static_list[i][1])
# 					l = len(static_list)
# 			j += 1
# 		i += 1


# def _consolidate_dynamic(dynamic_list):
# 	l = len(dynamic_list)
# 	i = 0

# 	while(i < l):
# 		j = 0
# 		while(j < l):
# 			if i != j:
# 				opstr1,bonds1,f1,f1_args = tuple(dynamic_list[i])
# 				opstr2,bonds2,f2,f2_args = tuple(dynamic_list[j])
# 				if (opstr1 == opstr2) and (f1 == f2) and (f1_args == f2_args):
# 					del dynamic_list[j]
# 					dynamic_list[i][1].extend(bonds2)
# 					_consolidate_bonds(dynamic_list[i][1])
# 					l = len(dynamic_list)
# 			j += 1
# 		i += 1



def test_function(func,func_args):
	t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )
	func_val=func(t,*func_args)
	func_val=_np.array(func_val)
	if func_val.ndim > 0:
		raise ValueError("function must return 0-dim numpy array or scalar value.")




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
	static_list = _consolidate_static(static_list)
	for opstr,indx,J in static_list:
		# print(opstr,bond)
		ME,row,col = basis.Op(opstr,indx,J,dtype)
		Ht=_sp.csr_matrix((ME,(row,col)),shape=(Ns,Ns),dtype=dtype) 
		H=H+Ht
		del Ht
		H.sum_duplicates() # sum duplicate matrix elements
		H.eliminate_zeros() # remove all zero matrix elements
	# print()
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
	dynamic={}
	dynamic_list = _consolidate_dynamic(dynamic_list)
	for opstr,indx,J,f,f_args in dynamic_list:
		if _np.isscalar(f_args): raise TypeError("function arguments must be array type")
		test_function(f,f_args)

		#indx = _np.asarray(indx,_np.int32)
		ME,row,col = basis.Op(opstr,indx,J,dtype)
		Ht =_sp.csr_matrix((ME,(row,col)),shape=(Ns,Ns),dtype=dtype) 

		func = function(f,tuple(f_args))
		if func in dynamic:
			try:
				dynamic[func] += Ht
			except:
				dynamic[func] = dynamic[func] + Ht
		else:
			dynamic[func] = Ht


	return dynamic





def make_op(basis,opstr,bonds,dtype):
	Ns=basis.Ns
	H=_sp.csr_matrix(([],([],[])),shape=(Ns,Ns),dtype=dtype)
	for bond in bonds:
		J=bond[0]
		indx=bond[1:]
		ME,row,col = basis.Op(opstr,indx,J,dtype)
		Ht=_sp.csr_matrix((ME,(row,col)),shape=(Ns,Ns),dtype=dtype) 
		H=H+Ht
		del Ht
		H.sum_duplicates() # sum duplicate matrix elements
		H.eliminate_zeros() # remove all zero matrix elements
	
	return H
