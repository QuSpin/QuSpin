


# this file stores

class basis(object):
	def __init__(self):
		self._Ns = 0
		self._basis = []
		self._operators = ""
		if self.__class__.__name__ == 'basis':
			raise ValueError("This class is not intended"
                             " to be instantiated directly.")



	@property
	def Ns(self):
		return self._Ns

	@property
	def operators(self):
		return self._operators

	def __repr__(self):
		return "< exact_diag_py basis with {0} states >".format(self._Ns)


	def __iter__(self):
		for b in self._basis:
			yield b

	def __bool__(self):
		return self._Ns > 0 






def isbasis(x):
	return isinstance(x,basis)


