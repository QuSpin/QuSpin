import basis_ops
from numpy import dtype as _dtype
from numpy import int32 as _index_type
from numpy import array as _array
import numpy as _np

__all__=['op','op_m','op_z','op_p','op_pz','op_p_z','op_t','op_t_z','op_t_p','op_t_pz','op_t_p_z']

_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}

_basis_op_errors={1:"opstr character not recognized.",
									-1:"attemping to use real hamiltonian with complex matrix elements."}



class FortranError(Exception):
	# this class defines an exception which can be raised whenever there is some sort of error which we can
	# see will obviously break the code. 
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message



def op(opstr,indx,J,dtype,pauli,basis,**blocks):

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_spinop"]
	col,ME,error = fortran_op(basis,opstr,indx)

	if error != 0: raise FortranError(_basis_op_errors[error])
	row=_array(xrange(len(basis)),dtype=_index_type)
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))


	return ME,row,col



def op_m(opstr,indx,J,dtype,pauli,basis,**blocks):

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_m_op"]
	col,ME,error = fortran_op(basis,opstr,indx)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))


	return ME,row,col
		



def op_z(opstr,indx,J,dtype,pauli,basis,L,**blocks):
	zblock=blocks.get("zblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_z_op"]

	col,ME,error = fortran_op(basis,opstr,indx,L,zblock)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))


	return ME,row,col




def op_p(opstr,indx,J,dtype,pauli,N,basis,L,**blocks):
	pblock=blocks.get("pblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_p_op"]
	col,ME,error = fortran_op(N,basis,opstr,indx,L,pblock)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))


	return ME,row,col




def op_pz(opstr,indx,J,dtype,pauli,N,basis,L,**blocks):
	pzblock=blocks.get("pzblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_pz_op"]
	col,ME,error = fortran_op(N,basis,opstr,indx,L,pzblock)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))


	return ME,row,col



def op_p_z(opstr,indx,J,dtype,pauli,N,basis,L,**blocks):
	zblock=blocks.get("zblock")
	pblock=blocks.get("pblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_p_z_op"]
	col,ME,error = fortran_op(N,basis,opstr,indx,L,pblock,zblock)
	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))


	return ME,row,col





def op_t(opstr,indx,J,dtype,pauli,N,basis,L,**blocks):
	a=blocks.get("a")
	kblock=blocks.get("kblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_t_op"]
	col,ME,error = fortran_op(N,basis,opstr,indx,L,kblock,a)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))


	return ME,row,col




def op_t_z(opstr,indx,J,dtype,pauli,N,m,basis,L,**blocks):
	a=blocks.get("a")
	kblock=blocks.get("kblock")
	zblock=blocks.get("zblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_t_z_op"]
	col,ME,error = fortran_op(N,m,basis,opstr,indx,L,zblock,kblock,a)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))

	return ME,row,col


def op_t_p(opstr,indx,J,dtype,pauli,N,m,basis,L,**blocks):
	a=blocks.get("a")
	kblock=blocks.get("kblock")
	pblock=blocks.get("pblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_t_p_op"]
	col,ME,error = fortran_op(N,m,basis,opstr,indx,L,pblock,kblock,a)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	row=_np.concatenate((row,row))

	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))

	return ME,row,col



def op_t_pz(opstr,indx,J,dtype,pauli,N,m,basis,L,**blocks):
	a=blocks.get("a")
	kblock=blocks.get("kblock")
	pzblock=blocks.get("pzblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_t_pz_op"]
	col,ME,error = fortran_op(N,m,basis,opstr,indx,L,pzblock,kblock,a)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	row=_np.concatenate((row,row))

	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
#	print col,ME
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))

	return ME,row,col


def op_t_p_z(opstr,indx,J,dtype,pauli,N,m,basis,L,**blocks):
	a=blocks.get("a")
	kblock=blocks.get("kblock")
	pblock=blocks.get("pblock")
	zblock=blocks.get("zblock")

	dtype = _dtype(dtype)
	char = _type_conv[dtype.char]
	fortran_op = basis_ops.__dict__[char+"_t_p_z_op"]
	col,ME,error = fortran_op(N,m,basis,opstr,indx,L,pblock,zblock,kblock,a)

	if error != 0: raise FortranError(_basis_op_errors[error])

	row=_array(xrange(len(basis)),dtype=_index_type)
	row=_np.concatenate((row,row))

	mask = col >= 0
	row = row[ mask ]
	col = col[ mask ]
	ME = ME[ mask ]
	col -= 1 #convert from fortran index to c index.
	if pauli:
		ME*=J
	else:
		ME*=(J*0.5**(len(opstr)))

	return ME,row,col


	
