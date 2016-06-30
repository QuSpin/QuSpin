import numpy as _np
from scipy import sparse as _sp
from scipy.sparse import linalg as _sla
from scipy import linalg as _la

MAXPRINT = 50
# this file stores

class basis(object):


	def __init__(self):
		self._Ns = 0
		self._basis = _np.asarray([])
		self._operators = ""
		if self.__class__.__name__ == 'basis':
			raise ValueError("This class is not intended"
                             " to be instantiated directly.")


	def __str__(self):
		
		string = "reference states: \n"
		if self._Ns == 0:
			return string
		
		
		str_list = self._get__str__()
		if self._Ns > MAXPRINT:
			L_str = len(str_list[0])
			t = (" ".join(["" for i in xrange(L_str/2)]))+":"
			str_list.insert(MAXPRINT//2,t)
		
		string += "\n".join(str_list)

		return string


	@property
	def Ns(self):
		return self._Ns

	@property
	def operators(self):
		return self._operators

	def __repr__(self):
		return "< exact_diag_py basis with {0} states >".format(self._Ns)





def isbasis(x):
	return isinstance(x,basis)


