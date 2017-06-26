
def _quspin_ident_function(*args):
	return 1.0


class function(object):
	def __init__(self,f=_quspin_ident_function,args=()):
		self._f = f
		self._args = args

	def conjugate(self):
		return conjugate_function(function(self._f,*self._args))

	def conj(self):
		return self.conjugate()

	def __neg__(self):
		return neg_function(function(self._f,*self._args))

	def __eq__(self,other):
		if not isinstance(other,self.__class__):
			return False
		else:
			return ((self._f == other._f) and (self._args == other._args))

	def __call__(self,*args):
		return self._f(*(args+self._args))

	def __mul__(self,other):
		return mul_function(self,other)

	def __div__(self,other):
		return div_function(self,other)

	def __add__(self,other):
		return add_function(self,other)

	def __sub__(self,other):
		return sub_function(other,self)


class noncomm_binary_function(function):
	def __init__(self,function1,function2):
		if not isinstance(function1,function):
			raise ValueError("func1 must be function object.")

		if not isinstance(function2,function):
			raise ValueError("func2 must be function object.")

		self._function1 = function1
		self._function2 = function2

	def __eq__(self,other):
		if not isinstance(other,self.__class__):
			return False
		else:
			return ((self._function1==other._function1) and (self._function2==other._function2))


class comm_binary_function(function):
	def __init__(self,function1,function2):
		if not isinstance(function1,function):
			raise ValueError("func1 must be function object.")

		if not isinstance(function2,function):
			raise ValueError("func2 must be function object.")

		self._function1 = function1
		self._function2 = function2


	def __eq__(self,other):
		if not isinstance(other,self.__class__):
			return False
		else:
			ord1 = ((self._function1==other._function1) and (self._function2==other._function2))
			ord2 = ((self._function1==other._function2) and (self._function2==other._function1))
			return (ord1 or ord2)


class mul_function(comm_binary_function):
	def __init__(self,*args,**kwargs):
		comm_binary_function.__init__(self,*args,**kwargs)

	def __call__(self,*args):
		return self._function1(*args)*self._function2(*args)


class add_function(comm_binary_function):
	def __init__(self,*args,**kwargs):
		comm_binary_function.__init__(self,*args,**kwargs)

	def __call__(self,*args):
		return self._function1(*args)+self._function2(*args)			


class div_function(noncomm_binary_function):
	def __init__(self,*args,**kwargs):
		noncomm_binary_function.__init__(self,*args,**kwargs)

	def __call__(self,*args):
		return self._function1(*args)/self._function2(*args)


class sub_function(noncomm_binary_function):
	def __init__(self,*args,**kwargs):
		noncomm_binary_function.__init__(self,*args,**kwargs)

	def __call__(self,*args):
		return self._function1(*args)-self._function2(*args)




class neg_function(function):
	def __init__(self,function1):
		if not isinstance(function1,function):
			raise ValueError("func1 must be function object.")

		self._function1 = function1

	def __call__(self,*args):
		return -self._function1(*args)	

	def __neg__(self):
		return self._function1


class conjugate_function(function):
	def __init__(self,function1):
		if not isinstance(function1,function):
			raise ValueError("func1 must be function object.")

		self._function1 = function1

	def __call__(self,*args):
		return self._function1(*args).conjugate()

	def conjugate(self):
		return self._function1



if __name__ =="__main__":
	import numpy as np

	f1 = function()
	f2 = function(lambda x:x)
	
	A = {f1:np.array([[0,-1j],[1j,0]])}

	print A
	A = {f1.conj():matrix.conj() for f1,matrix in A.items()}
	print A
	A = {f1.conj():matrix.conj() for f1,matrix in A.items()}
	print A


		

