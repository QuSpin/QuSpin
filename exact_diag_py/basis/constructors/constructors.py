import fortran_wrap
from fortran_wrap import *
from numpy import dtype as _dtype


# open boundary conditions
fortran_op_m={"float32":fortran_wrap.s_m_op,
							"float64":fortran_wrap.d_m_op,
							"complex64":fortran_wrap.c_m_op,
							"complex128":fortran_wrap.z_m_op}

fortran_op_p={"float32":fortran_wrap.s_p_op,
							"float64":fortran_wrap.d_p_op,
							"complex64":fortran_wrap.c_p_op,
							"complex128":fortran_wrap.z_p_op}

fortran_op_z={"float32":fortran_wrap.s_z_op,
							"float64":fortran_wrap.d_z_op,
							"complex64":fortran_wrap.c_z_op,
							"complex128":fortran_wrap.z_z_op}

fortran_op_pz={"float32":fortran_wrap.s_pz_op,
							 "float64":fortran_wrap.d_pz_op,
							 "complex64":fortran_wrap.c_pz_op,
							 "complex128":fortran_wrap.z_pz_op}

fortran_op_p_z={"float32":fortran_wrap.s_p_z_op,
								"float64":fortran_wrap.d_p_z_op,
								"complex64":fortran_wrap.c_p_z_op,
								"complex128":fortran_wrap.z_p_z_op}

# periodic boundary conditions

fortran_op_t={"float32":fortran_wrap.s_t_op,
							"float64":fortran_wrap.d_t_op,
							"complex64":fortran_wrap.c_t_op,
							"complex128":fortran_wrap.z_t_op}



#SpinOp for no symmetries

fortran_SpinOp={"float32":s_spinop,
								"float64":d_spinop,
								"complex64":c_spinop,
								"complex128":z_spinop}


__all__=['op_m','op_z','op_p','op_pz','op_p_z','op_t','SpinOp']







class FortranError(Exception):
	# this class defines an exception which can be raised whenever there is some sort of error which we can
	# see will obviously break the code. 
	def __init__(self,message):
		self.message=message
	def __str__(self):
		return self.message




def op_m(basis,opstr,indx,dtype):
	dtype=str(_dtype(dtype))
	col,ME,error = fortran_op_m[dtype](basis,opstr,indx)
	if error != 0:
		if error == 1:
			raise FortranError("opstr character not recognized.")
		elif error == -1:
			raise FortranError("attemping to use real hamiltonian with complex matrix elements.")


	return ME,col
		



def op_z(N,basis,opstr,indx,L,dtype,**blocks):
	zblock=blocks.get("zblock")
	dtype=str(_dtype(dtype))
	col,ME,error = fortran_op_z[dtype](N,basis,opstr,indx,L,zblock)
	if error != 0:
		if error == 1:
			raise FortranError("opstr character not recognized.")
		elif error == -1:
			raise FortranError("attemping to use real hamiltonian with complex matrix elements.")

	return ME,col



def op_p(N,basis,opstr,indx,L,dtype,**blocks):
	pblock=blocks.get("pblock")
	dtype=str(_dtype(dtype))
	col,ME,error = fortran_op_p[dtype](N,basis,opstr,indx,L,pblock)
	if error != 0:
		if error == 1:
			raise FortranError("opstr character not recognized.")
		elif error == -1:
			raise FortranError("attemping to use real hamiltonian with complex matrix elements.")

	return ME,col



def op_pz(N,basis,opstr,indx,L,dtype,**blocks):
	pzblock=blocks.get("pzblock")
	dtype=str(_dtype(dtype))
	col,ME,error = fortran_op_pz[dtype](N,basis,opstr,indx,L,pzblock)
	if error != 0:
		if error == 1:
			raise FortranError("opstr character not recognized.")
		elif error == -1:
			raise FortranError("attemping to use real hamiltonian with complex matrix elements.")

	return ME,col



def op_p_z(N,basis,opstr,indx,L,dtype,**blocks):
	zblock=blocks.get("zblock")
	pblock=blocks.get("pblock")
	dtype=str(_dtype(dtype))
	col,ME,error = fortran_op_p_z[dtype](N,basis,opstr,indx,L,pblock,zblock)
	if error != 0:
		if error == 1:
			raise FortranError("opstr character not recognized.")
		elif error == -1:
			raise FortranError("attemping to use real hamiltonian with complex matrix elements.")

	return ME,col





def op_t(N,basis,opstr,indx,L,dtype,**blocks):
	a=blocks.get("a")
	kblock=blocks.get("kblock")
	dtype=str(_dtype(dtype))
	col,ME,error = fortran_op_t[dtype](N,basis,opstr,indx,L,kblock,a)
	if error != 0:
		if error == 1:
			raise FortranError("opstr character not recognized.")
		elif error == -1:
			raise FortranError("attemping to use real hamiltonian with complex matrix elements.")

	return ME,col



def SpinOp(basis,opstr,indx,dtype):
	dtype=str(_dtype(dtype))
	ME,col,error=fortran_SpinOp[dtype](basis,opstr,indx)
	if error != 0:
		if error == 1:
			raise FortranError("opstr character not regcognized.")
		elif error == -1:
			raise FortranError("attemping to use real hamiltonian with complex matrix elements.")

	return ME,col


	
