from ._functions import function,_quspin_ident_function


import scipy.sparse as _sp
import numpy as _np

allowed_dtypes = set([_np.float32,_np.complex64,_np.float64,_np.complex128])



class _base_operator(object):
	def __init__(self,static_matrix,dynamic_matrix,default_args,dtype=_np.complex128,copy=True,shape=None):

		if dtype not in allowed_dtypes:
			raise ValueError("dtype must be numpy dtypes: float32, float64, complex64 or complex128.")

		self._default_args = default_args
		self._dtype=dtype
		self._shape = None

		ifunc = function()
		self._ops = {}


class operator(_base_operator):
	def __init__(static,dynamic,default_args,**kwargs):

		static_opstr = []
		dynamic_opstr = []

		static_matrix = []
		dynamic_matrix = []

		for ele in indx_list:
			if type(ele) is list:
				if len(ele) == 2:
					static_opstr.append(ele)
				elif len(ele) == 3:
					dynamic_matrix.append(ele)
				elif len(ele) == 4:
					dynamic_opstr.append(ele)
			else:
				static_matrix.append(ele)

		if static_opstr or dynamic_opstr:
			# check basis
			if basis is None:
				raise ValueError("basis required for operator strings")
			else:
				shape = (basis.Ns,basis.Ns)
				self._shape = shape
				self._basis = basis

			for opstr,indx_list in static_opstr:
				matrix = _sp.sparse(self._shape,dtype=dtype)

				for indx in indx_list:
					J = indx[0]
					indx = indx[1:]
					ME,row,col = basis.Op(opstr,indx,J,dtype)
					matrix += _sp.csr_matrix((ME,(row,col)),shape=shape,dtype=dtype)

				static_matrix.append(matrix)

			for opstr,indx_list,f,args in dynamic_opstr:
				matrix = _sp.sparse(self._shape,dtype=dtype)
				for indx in indx_list:
					J = indx[0]
					indx = indx[1:]
					ME,row,col = basis.Op(opstr,indx,J,dtype)
					matrix += _sp.csr_matrix((ME,(row,col)),shape=shape,dtype=dtype)

				dynamic_matrix.append((matrix,f,args))


		_base_operator.__init__(self,static_matrix,dynamic_matrix,default_args,**kwargs)










