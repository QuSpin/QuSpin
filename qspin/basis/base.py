import numpy as _np
import warnings

MAXPRINT = 50
# this file stores the base class for all basis classes

class basis(object):


	def __init__(self):
		self._Ns = 0
		self._basis = _np.asarray([])
		self._operators = "no operators for base."
		if self.__class__.__name__ == 'basis':
			raise ValueError("This class is not intended"
                             " to be instantiated directly.")


	def __str__(self):
		
		string = "reference states: \n"
		if self._Ns == 0:
			return string
		
		if hasattr(self,"_get__str__"):
			str_list = self._get__str__()
			if self._Ns > MAXPRINT:
				L_str = len(str_list[0])
				t = (" ".join(["" for i in xrange(L_str/2)]))+":"
				str_list.insert(MAXPRINT//2,t)
		
			string += "\n".join(str_list)
			return string
		else:
			warnings.warn("basis class {0} missing _get__str__ function, can not print out basis representatives.".format(type(self)),UserWarning,stacklevel=3)
			return "reference states: \n\t not availible"


	@property
	def Ns(self):
		return self._Ns

	@property
	def operators(self):
		return self._operators

	def __repr__(self):
		return "< instance of 'qspin.basis.base' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.base'>"




	def Op(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of 'Op' required for for creating hamiltonians!".format(self.__class__))
		
	def get_vec(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of 'get_vec'!".format(self.__class__))

	def get_proj(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of 'get_proj'!".format(self.__class__))




	def _hc_opstr(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '_hc_opstr' required for hermiticity check!".format(self.__class__))

	def _sort_opstr(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '_sort_opstr' required for symmetry and hermiticity checks!".format(self.__class__))

	def _expand_opstr(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '_expand_opstr' required for particle conservation check!".format(self.__class__))

	def _non_zero(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '_non_zero' required for particle conservation check!".format(self.__class__))


	# These methods are required for every basis class.
	# for examples see spin.py
	def hc_opstr(self,op):
		return self.__class__._hc_opstr(self,op)

	def sort_opstr(self,op):
		return self.__class__._sort_opstr(self,op)

	def expand_opstr(self,op,num):
		return self.__class__._expand_opstr(self,op,num)

	def non_zero(self,op):
		return self.__class__._non_zero(self,op)


	# these methods can be overriden in the even the ones implimented below do not work
	# for whatever reason
	def sort_list(self,op_list):
		return self.__class__._sort_list(self,op_list)

	def get_lists(self,static,dynamic):
		return self.__class__._get_lists(self,static,dynamic)

	def get_hc_lists(self,static_list,dynamic_list):
		return self.__class__._get_hc_lists(self,static_list,dynamic_list)

	def consolidate_lists(self,static_list,dynamic_list):
		return self.__class__._consolidate_lists(self,static_list,dynamic_list)

	def expand_list(self,op_list):
		return self.__class__._expand_list(self,op_list)






	def _sort_list(self,op_list):
		sorted_op_list = []
		for op in op_list:
			sorted_op_list.append(self.sort_opstr(op))
		sorted_op_list = tuple(sorted_op_list)

		return sorted_op_list



	# this function gets overridden in photon_basis because the index must be extended to include the photon index.
	def _get_lists(self,static,dynamic):
		static_list = []
		for opstr,bonds in static:
			for bond in bonds:
				indx = list(bond[1:])
				J = complex(bond[0])
				static_list.append((opstr,indx,J))

		dynamic_list = []
		for opstr,bonds,f,f_args in dynamic:
			for bond in bonds:
				indx = list(bond[1:])
				J = complex(bond[0])
				dynamic_list.append((opstr,indx,J,f,f_args))

		return self.sort_list(static_list),self.sort_list(dynamic_list)


	def _get_hc_lists(self,static_list,dynamic_list):
		static_list_hc = []
		for op in static_list:
			static_list_hc.append(self.hc_opstr(op))

		static_list_hc = tuple(static_list_hc)


		# define arbitrarily complicated weird-ass number
		t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )

		dynamic_list_hc = []
		dynamic_list_eval = []
		for opstr,indx,J,f,f_args in dynamic_list:
			J *= f(t,*f_args)
			op = (opstr,indx,J)
			dynamic_list_hc.append(self.hc_opstr(op))
			dynamic_list_eval.append(self.sort_opstr(op))

		dynamic_list_hc = tuple(dynamic_list_hc)
		
		return static_list,static_list_hc,dynamic_list_eval,dynamic_list_hc



	def _expand_list(self,op_list):
		op_list_exp = []
		for i,op in enumerate(op_list):
			new_ops = self.expand_opstr(op,[i])
			for new_op in new_ops:
#				print new_op
				if self.non_zero(new_op):
					op_list_exp.append(new_op)

		return op_list_exp



	def _consolidate_lists(self,static_list,dynamic_list):
		l = len(static_list)
		i = 0
		while (i < l):
			j = 0
			while (j < l):
				if i != j:
					opstr1,indx1,J1,i1 = tuple(static_list[i]) 
					opstr2,indx2,J2,i2 = tuple(static_list[j])
					if opstr1 == opstr2 and indx1 == indx2:
						
						del static_list[j]
						if j < i: 
							i -= 1
						else:
							j -= 1

						if J1 == -J2: 
							del static_list[i]
							i -= 1
						else:
							static_list[i] = list(static_list[i])
							static_list[i][2] += J2
							static_list[i][3].extend(i2)
							static_list[i] = tuple(static_list[i])
						
						l = len(static_list)

				if i >= l: break
				j += 1
			i += 1
			
		l = len(dynamic_list)
		i = 0

		while (i < l):
			j = 0
			while (j < l):
				if i != j:
					opstr1,indx1,J1,f1,f1_args,i1 = tuple(dynamic_list[i]) 
					opstr2,indx2,J2,f2,f2_args,i2 = tuple(dynamic_list[j])
					if opstr1 == opstr2 and indx1 == indx2 and f1 == f2 and f1_args == f2_args:

						del dynamic_list[j]
						if J1 == -J2: 
							if j < i: i -= 1
							del dynamic_list[i]
						else:
							dynamic_list[i] = list(dynamic_list[i])
							dynamic_list[i][2] += J2
							dynamic_list[i][3].extend(i2)
							dynamic_list[i] = tuple(dynamic_list[i])
						
						l = len(dynamic_list)

						if i >= l: break
				j += 1
			i += 1


		return static_list,dynamic_list
















	def check_hermitian(self,static_list,dynamic_list):

		static_list,dynamic_list = self.get_lists(static_list,dynamic_list)
		static_expand,static_expand_hc,dynamic_expand,dynamic_expand_hc = self.get_hc_lists(static_list,dynamic_list)

		# calculate non-hermitian elements
		diff = set( tuple(static_expand) ) - set( tuple(static_expand_hc) )
		if diff:
			unique_opstrs = list(set( zip(*tuple(diff))[0]) )
			warnings.warn("The following static operator strings contain non-hermitian couplings: {}".format(unique_opstrs),UserWarning,stacklevel=3)
			user_input = raw_input("Display all {} non-hermitian couplings? (y or n) ".format(len(diff)) )
			if user_input == 'y':
				print "   (opstr, indices, coupling)"
				for i in xrange(len(diff)):
					print "{}. {}".format(i+1, list(diff)[i])
			raise TypeError("Hamiltonian not hermitian! To turn this check off set check_herm=False in hamiltonian.")
			
			
		# define arbitrarily complicated weird-ass number
		t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )


		# calculate non-hermitian elements
		diff = set( tuple(dynamic_expand) ) - set( tuple(dynamic_expand_hc) )
		if diff:
			unique_opstrs = list(set( zip(*tuple(diff))[0]) )
			warnings.warn("The following dynamic operator strings contain non-hermitian couplings: {}".format(unique_opstrs),UserWarning,stacklevel=3)
			user_input = raw_input("Display all {} non-hermitian couplings at time t = {}? (y or n) ".format( len(diff), _np.round(t,5)))
			if user_input == 'y':
				print "   (opstr, indices, coupling(t))"
				for i in xrange(len(diff)):
					print "{}. {}".format(i+1, list(diff)[i])
			raise TypeError("Hamiltonian not hermitian! To turn this check off set check_herm=False in hamiltonian.")

		print "Hermiticity check passed!"




	






	def check_pcon(self,static,dynamic):
		if not hasattr(self,"_check_pcon"):
			warnings.warn("Test for particle conservation not implemented for {0}, to turn off this warning set check_pcon=Flase in hamiltonian".format(type(self)),UserWarning,stacklevel=3)
			return

		if self._check_pcon:
			static_list,dynamic_list = self.get_lists(static,dynamic)
			static_list_exp = self.expand_list(static_list)
			dynamic_list_exp = self.expand_list(dynamic_list)
			static_list_exp,dynamic_list_exp = self.consolidate_lists(static_list_exp,dynamic_list_exp)

			con = ""

			odd_ops = []
			for opstr,indx,J,ii in static_list_exp:	
				p = opstr.count("+")
				m = opstr.count("-")

				if (p-m) != 0:
					for i in ii:
						if static_list[i] not in odd_ops:
							odd_ops.append(static_list[i])


	
			if odd_ops:
				unique_opstrs = list(set( zip(*tuple(odd_ops))[0]) )
				unique_odd_ops = []
				[ unique_odd_ops.append(ele) for ele in odd_ops if ele not in unique_odd_ops]
				warnings.warn("The following static operator strings do not conserve particle number{1}: {0}".format(unique_opstrs,con),UserWarning,stacklevel=4)
				user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(odd_ops)) )
				if user_input == 'y':
					print " these operators do not conserve particle number{0}:".format(con)
					print "   (opstr, indices, coupling)"
					for i,op in enumerate(unique_odd_ops):
						print "{0}. {1}".format(i+1, op)
				raise TypeError("Hamiltonian does not conserve particle number{0} To turn off check, use check_pcon=False in hamiltonian.".format(con))

			


			odd_ops = []
			for opstr,indx,J,f,f_args,ii in dynamic_list_exp:	
				p = opstr.count("+")
				m = opstr.count("-")
				if (p-m) != 0:
					for i in ii:
						if dynamic_list[i] not in odd_ops:
							odd_ops.append(dynamic_list[i])

	
			if odd_ops:
				unique_opstrs = list(set( zip(*tuple(odd_ops))[0]) )
				unique_odd_ops = []
				[ unique_odd_ops.append(ele) for ele in odd_ops if ele not in unique_odd_ops]
				warnings.warn("The following static operator strings do not conserve particle number{1}: {0}".format(unique_opstrs,con),UserWarning,stacklevel=4)
				user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(odd_ops)) )
				if user_input == 'y':
					print " these operators do not conserve particle number{0}:".format(con)
					print "   (opstr, indices, coupling)"
					for i,op in enumerate(unique_odd_ops):
						print "{0}. {1}".format(i+1, op)
				raise TypeError("Hamiltonian does not conserve particle number{0} To turn off check, use check_pcon=False in hamiltonian.".format(con))

			print "Particle conservation check passed!"




	def check_symm(self,static,dynamic):
		if not hasattr(self,"_check_symm"):
			warnings.warn("Test for symmetries not implemented for {0}, to turn off this warning set check_pcon=Flase in hamiltonian".format(type(self)),UserWarning,stacklevel=3)
			return

		static_blocks,dynamic_blocks = self._check_symm(static,dynamic)

		# define arbitrarily complicated weird-ass number
		t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )

		for symm in static_blocks.keys():
			if len(static_blocks[symm]) == 2:
				odd_ops,missing_ops = static_blocks[symm]
				ops = list(missing_ops)
				ops.extend(odd_ops)
				unique_opstrs = list(set( zip(*tuple(ops))[0]) )
				if unique_opstrs:
					unique_missing_ops = []
					unique_odd_ops = []
					[ unique_missing_ops.append(ele) for ele in missing_ops if ele not in unique_missing_ops]
					[ unique_odd_ops.append(ele) for ele in odd_ops if ele not in unique_odd_ops]
					warnings.warn("The following static operator strings do not obey {0}: {1}".format(symm,unique_opstrs),UserWarning,stacklevel=4)
					user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(unique_missing_ops) + len(unique_odd_ops)) )
					if user_input == 'y':
						print " these operators are needed for {}:".format(symm)
						print "   (opstr, indices, coupling)"
						for i,op in enumerate(unique_missing_ops):
							print "{0}. {1}".format(i+1, op)
						print 
						print " these do not obey the {}:".format(symm)
						print "   (opstr, indices, coupling)"
						for i,op in enumerate(unique_odd_ops):
							print "{0}. {1}".format(i+1, op)
					raise TypeError("Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(symm))


			elif len(static_blocks[symm]) == 1:
				missing_ops, = static_blocks[symm]
				unique_opstrs = list(set( zip(*tuple(missing_ops))[0]) )
				if unique_opstrs:
					unique_missing_ops = []
					[ unique_missing_ops.append(ele) for ele in missing_ops if ele not in unique_missing_ops]
					warnings.warn("The following static operator strings do not obey {0}: {1}".format(symm,unique_opstrs),UserWarning,stacklevel=4)
					user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(unique_missing_ops)) )
					if user_input == 'y':
						print " these operators are needed for {}:".format(symm)
						print "   (opstr, indices, coupling)"
						for i,op in enumerate(unique_missing_ops):
							print "{0}. {1}".format(i+1, op)
					raise TypeError("Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(symm))
			else:
				continue


		for symm in dynamic_blocks.keys():
			if len(dynamic_blocks[symm]) == 2:
				odd_ops,missing_ops = dynamic_blocks[symm]
				ops = list(missing_ops)
				ops.extend(odd_ops)
				unique_opstrs = list(set( zip(*tuple(ops))[0]) )
				if unique_opstrs:
					unique_missing_ops = []
					unique_odd_ops = []
					[ unique_missing_ops.append(ele) for ele in missing_ops if ele not in unique_missing_ops]
					[ unique_odd_ops.append(ele) for ele in odd_ops if ele not in unique_odd_ops]
					warnings.warn("The following dynamic operator strings do not obey {0}: {1}".format(symm,unique_opstrs),UserWarning,stacklevel=4)
					user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(unique_missing_ops) + len(unique_odd_ops)) )
					if user_input == 'y':
						print " these operators are missing for {}:".format(symm)
						print "   (opstr, indices, coupling)"
						for i,op in enumerate(unique_missing_ops):
							print "{0}. {1}".format(i+1, op)
						print 
						print " these do not obey {}:".format(symm)
						print "   (opstr, indices, coupling)"
						for i,op in enumerate(unique_odd_ops):
							print "{0}. {1}".format(i+1, op)
					raise TypeError("Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(symm))


			elif len(dynamic_blocks[symm]) == 1:
				missing_ops, = dynamic_blocks[symm]
				unique_opstrs = list(set( zip(*tuple(missing_ops))[0]) )
				if unique_opstrs:
					unique_missing_ops = []
					[ unique_missing_ops.append(ele) for ele in missing_ops if ele not in unique_missing_ops]
					warnings.warn("The following dynamic operator strings do not obey {0}: {1}".format(symm,unique_opstrs),UserWarning,stacklevel=4)
					user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(unique_missing_ops)) )
					if user_input == 'y':
						print " these operators are needed for {}:".format(symm)
						print "   (opstr, indices, coupling)"
						for i,op in enumerate(unique_missing_ops):
							print "{0}. {1}".format(i+1, op)
					raise TypeError("Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(symm))
			else:
				continue

		print "Symmetry checks passed!"







def isbasis(x):
	return isinstance(x,basis)


