import numpy as _np
import scipy.sparse as _sp
import warnings

MAXPRINT = 50
# this file stores the base class for all basis classes

class basis(object):


	def __init__(self):
		self._Ns = 0
		self._basis = _np.asarray([])
		self._operators = "no operators for base."
		self._unique_me = True
		if self.__class__.__name__ == 'basis':
			raise ValueError("This class is not intended"
							 " to be instantiated directly.")


	def __str__(self):
		
		string = "reference states: \n"
		if self._Ns == 0:
			return string
		
		str_list = list(self._get__str__())
		if self._Ns > MAXPRINT:
			L_str = len(str_list[0])
			t = (" ".join(["" for i in range(L_str//2)]))+":"
			str_list.insert(MAXPRINT//2,t)
		
		string += "\n".join(str_list)
		return string



	@property
	def unique_me(self):
		return self._unique_me

	@property
	def Ns(self):
		return self._Ns

	@property
	def operators(self):
		return self._operators

	def _get__str__(self):
		temp1 = "\t{0:"+str(len(str(self.Ns)))+"d}.  "
		n_space = len(str(self.sps))
		temp2 = "|"+(" ".join(["{:"+str(n_space)+"d}" for i in range(self.N)]))+">"

		if self._Ns > MAXPRINT:
			half = MAXPRINT // 2
			str_list = [(temp1.format(i))+(temp2.format(*[int(b//self.sps**i)%self.sps for i in range(self.N)])) for i,b in zip(range(half),self._basis[:half])]
			str_list.extend([(temp1.format(i))+(temp2.format(*[int(b//self.sps**i)%self.sps for i in range(self.N)])) for i,b in zip(range(self._Ns-half,self._Ns,1),self._basis[-half:])])
		else:
			str_list = [(temp1.format(i))+(temp2.format(*[int(b//self.sps**i)%self.sps for i in range(self.N)])) for i,b in enumerate(self._basis)]

		return tuple(str_list)


	# this methods are optional and are not required for main functions:
	def __iter__(self):
		raise NotImplementedError("basis class: {0} missing implimentation of '__iter__' required for for iterating over basis!".format(self.__class__))

	def __getitem__(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '__getitem__' required for for '[]' operator!".format(self.__class__))

	def index(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of 'index' function!".format(self.__class__))

#	def _get__str__(self,*args,**kwargs):
#		raise NotImplementedError("basis class: {0} missing implimentation of '_get__str__' required to print basis!".format(self.__class__))		


	# this method is required in order to create manybody hamiltonians/operators
	def Op(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of 'Op' required for for creating hamiltonians!".format(self.__class__))

	# this method is required in order to use entanglement entropy functions		
	def get_vec(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of 'get_vec' required for entanglement entropy calculations!".format(self.__class__))

	@property
	def sps(self):
		raise NotImplementedError("basis class: {0} missing local number of degrees of freedom per site 'm' required for entanglement entropy calculations!".format(self.__class__))

	# this method is required for the block_tools functions
	def get_proj(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of 'get_proj' required for entanglement block_tools calculations!".format(self.__class__))

	# thes methods are required for the symmetry, particle conservation, and hermiticity checks
	def _hc_opstr(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '_hc_opstr' required for hermiticity check! turn this check off by setting test_herm=False".format(self.__class__))

	def _sort_opstr(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '_sort_opstr' required for symmetry and hermiticity checks! turn this check off by setting check_herm=False".format(self.__class__))

	def _expand_opstr(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '_expand_opstr' required for particle conservation check! turn this check off by setting check_pcon=False".format(self.__class__))

	def _non_zero(self,*args,**kwargs):
		raise NotImplementedError("basis class: {0} missing implimentation of '_non_zero' required for particle conservation check! turn this check off by setting check_pcon=False".format(self.__class__))


	# these methods can be overwritten in the event that the ones implimented below do not work
	# for whatever reason
	def sort_local_list(self,op_list):
		return self.__class__._sort_local_list(self,op_list)

	def get_local_lists(self,static,dynamic):
		return self.__class__._get_local_lists(self,static,dynamic)

	def get_hc_local_lists(self,static_list,dynamic_list):
		return self.__class__._get_hc_local_lists(self,static_list,dynamic_list)

	def consolidate_local_lists(self,static_list,dynamic_list):
		return self.__class__._consolidate_local_lists(self,static_list,dynamic_list)

	def expand_local_list(self,op_list):
		return self.__class__._expand_local_list(self,op_list)

	def expanded_form(self,static_list,dynamic_list):
		return self.__class__._expanded_form(self,static_list,dynamic_list)



	def _sort_local_list(self,op_list):
		sorted_op_list = []
		for op in op_list:
			sorted_op_list.append(self._sort_opstr(op))
		sorted_op_list = tuple(sorted_op_list)

		return sorted_op_list


	# this function flattens out the static and dynamics lists to: 
	# [[opstr1,indx11,J11,...],[opstr1,indx12,J12,...],...,[opstrn,indxnm,Jnm,...]]
	# this function gets overridden in photon_basis because the index must be extended to include the photon index.
	def _get_local_lists(self,static,dynamic):
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

		return self.sort_local_list(static_list),self.sort_local_list(dynamic_list)

	# takes the list from the format given by get_local_lists and takes the hermitian conjugate of operators.
	def _get_hc_local_lists(self,static_list,dynamic_list):
		static_list_hc = []
		for op in static_list:
			static_list_hc.append(self._hc_opstr(op))

		static_list_hc = tuple(static_list_hc)


		# define arbitrarily complicated weird-ass number
		t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )

		dynamic_list_hc = []
		dynamic_list_eval = []
		for opstr,indx,J,f,f_args in dynamic_list:
			J *= f(t,*f_args)
			op = (opstr,indx,J)
			dynamic_list_hc.append(self._hc_opstr(op))
			dynamic_list_eval.append(self._sort_opstr(op))

		dynamic_list_hc = tuple(dynamic_list_hc)
		
		return static_list,static_list_hc,dynamic_list_eval,dynamic_list_hc


	# this function takes the list format giveb by get_local_lists and expands any operators into the most basic components
	# 'n'(or 'z' for spins),'+','-' If by default one doesn't need to do this then _expand_opstr must do nothing. 
	def _expand_local_list(self,op_list):
		op_list_exp = []
		for i,op in enumerate(op_list):
			new_ops = self._expand_opstr(op,[i])
			for new_op in new_ops:
				if self._non_zero(new_op):
					op_list_exp.append(new_op)

		return self.sort_local_list(op_list_exp)


	def _consolidate_local_lists(self,static_list,dynamic_list):

		static_dict={}
		for opstr,indx,J,ii in static_list:
			if opstr in static_dict:
				if indx in static_dict[opstr]:
					static_dict[opstr][indx][0] += J
					static_dict[opstr][indx][1].extend(ii)
				else:
					static_dict[opstr][indx] = [J,ii]
			else:
				static_dict[opstr] = {indx:[J,ii]}

		static_list = []
		for opstr,opstr_dict in static_dict.items():
			for indx,(J,ii) in opstr_dict.items():
				if J != 0:
					static_list.append((opstr,indx,J,ii))


		dynamic_dict={}
		for opstr,indx,J,f,f_args,ii in dynamic_list:
			if opstr in dynamic_dict:
				if indx in dynamic_dict[opstr]:
					dynamic_dict[opstr][indx][0] += J
					dynamic_dict[opstr][indx][3].extend(ii)
				else:
					dynamic_dict[opstr][indx] = [J,f,f_args,ii]
			else:
				dynamic_dict[opstr] = {indx:[J,f,f_args,ii]}


		dynamic_list = []
		for opstr,opstr_dict in dynamic_dict.items():
			for indx,(J,f,f_args,ii) in opstr_dict.items():
				if J != 0:
					dynamic_list.append((opstr,indx,J,f,f_args,ii))


		return static_list,dynamic_list


	def _expanded_form(self,static,dynamic):
		static_list,dynamic_list = self.get_local_lists(static,dynamic)
		static_list = self.expand_local_list(static_list)
		dynamic_list = self.expand_local_list(dynamic_list)
		static_list,dynamic_list = self.consolidate_local_lists(static_list,dynamic_list)

		static_dict={}
		for opstr,indx,J,ii in static_list:
			indx = list(indx)
			indx.insert(0,J)
			if opstr in static_dict:
				static_dict[opstr].append(indx)
			else:
				static_dict[opstr] = [indx]

		static = [[str(key),list(value)] for key,value in static_dict.items()]

		dynamic_dict={}
		for opstr,indx,J,f,f_args,ii in dynamic_list:
			indx = list(indx)
			indx.insert(0,J)
			if opstr in dynamic_dict:
				dynamic_dict[opstr].append(indx)
			else:
				dynamic_dict[opstr] = [indx]

		dynamic = [[str(key),list(value)] for key,value in dynamic_dict.items()]

		return static,dynamic








	def check_hermitian(self,static_list,dynamic_list):

		static_list,dynamic_list = self.get_local_lists(static_list,dynamic_list)
		static_expand,static_expand_hc,dynamic_expand,dynamic_expand_hc = self.get_hc_local_lists(static_list,dynamic_list)
		# calculate non-hermitian elements
		diff = set( tuple(static_expand) ) - set( tuple(static_expand_hc) )
		
		if diff:
			unique_opstrs = list(set( zip(*tuple(diff))[0]) )
			warnings.warn("The following static operator strings contain non-hermitian couplings: {}".format(unique_opstrs),UserWarning,stacklevel=3)
			user_input = raw_input("Display all {} non-hermitian couplings? (y or n) ".format(len(diff)) )
			if user_input == 'y':
				print("   (opstr, indices, coupling)")
				for i in range(len(diff)):
					print("{}. {}".format(i+1, list(diff)[i]))
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
				print("   (opstr, indices, coupling(t))")
				for i in range(len(diff)):
					print("{}. {}".format(i+1, list(diff)[i]))
			raise TypeError("Hamiltonian not hermitian! To turn this check off set check_herm=False in hamiltonian.")

		print("Hermiticity check passed!")




	






	def check_pcon(self,static,dynamic):
		if not hasattr(self,"_check_pcon"):
			warnings.warn("Test for particle conservation not implemented for {0}, to turn off this warning set check_pcon=Flase in hamiltonian".format(type(self)),UserWarning,stacklevel=3)
			return

		if self._check_pcon:
			static_list,dynamic_list = self.get_local_lists(static,dynamic)
			static_list_exp = self.expand_local_list(static_list)
			dynamic_list_exp = self.expand_local_list(dynamic_list)
			static_list_exp,dynamic_list_exp = self.consolidate_local_lists(static_list_exp,dynamic_list_exp)
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
					print(" these operators do not conserve particle number{0}:".format(con))
					print("   (opstr, indices, coupling)")
					for i,op in enumerate(unique_odd_ops):
						print("{0}. {1}".format(i+1, op))
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
					print(" these operators do not conserve particle number{0}:".format(con))
					print("   (opstr, indices, coupling)")
					for i,op in enumerate(unique_odd_ops):
						print("{0}. {1}".format(i+1, op))
				raise TypeError("Hamiltonian does not conserve particle number{0} To turn off check, use check_pcon=False in hamiltonian.".format(con))

			print("Particle conservation check passed!")




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
						print(" these operators are needed for {}:".format(symm))
						print("   (opstr, indices, coupling)")
						for i,op in enumerate(unique_missing_ops):
							print("{0}. {1}".format(i+1, op))
						print(" ")
						print(" these do not obey the {}:".format(symm))
						print("   (opstr, indices, coupling)")
						for i,op in enumerate(unique_odd_ops):
							print("{0}. {1}".format(i+1, op))
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
						print(" these operators are needed for {}:".format(symm))
						print("   (opstr, indices, coupling)")
						for i,op in enumerate(unique_missing_ops):
							print("{0}. {1}".format(i+1, op))
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
						print(" these operators are missing for {}:".format(symm))
						print("   (opstr, indices, coupling)")
						for i,op in enumerate(unique_missing_ops):
							print("{0}. {1}".format(i+1, op))
						print(" ")
						print(" these do not obey {}:".format(symm))
						print("   (opstr, indices, coupling)")
						for i,op in enumerate(unique_odd_ops):
							print("{0}. {1}".format(i+1, op))
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
						print(" these operators are needed for {}:".format(symm))
						print("   (opstr, indices, coupling)")
						for i,op in enumerate(unique_missing_ops):
							print("{0}. {1}".format(i+1, op))
					raise TypeError("Hamiltonian does not obey {0}! To turn off check, use check_symm=False in hamiltonian.".format(symm))
			else:
				continue

		print("Symmetry checks passed!")







def isbasis(x):
	return isinstance(x,basis)



####################################################
# set of helper functions to implement the partial # 
# trace of lattice density matrices. They do not   #
# have any checks and states are assumed to be     #
# in the non-symmetry reduced basis.               #
####################################################

def _lattice_partial_trace_pure(psi,sub_sys_A,L,sps,return_rdm="A"):
	extra_dims = psi.shape[:-1]
	n_dims = len(extra_dims)

	sub_sys_B = set(range(L))-set(sub_sys_A)

	sub_sys_A = tuple(sub_sys_A)
	sub_sys_B = tuple(sorted(sub_sys_B))

	L_A = len(sub_sys_A)
	L_B = len(sub_sys_B)

	Ns_A = (sps**L_A)
	Ns_B = (sps**L_B)

	T_tup = tuple(sub_sys_A)+tuple(sub_sys_B) 
	T_tup = tuple(range(n_dims)) + tuple(n_dims + s for s in T_tup)
	R_tup = extra_dims + tuple(sps for i in range(L))

	psi_v = psi.reshape(R_tup) # DM where index is given per site as rho_v[i_1,...,i_L,j_1,...j_L]
	psi_v = psi_v.transpose(T_tup) # take transpose to reshuffle indices
	psi_v = psi_v.reshape(extra_dims+(Ns_A,Ns_B))

	if return_rdm == "A":
		return _np.squeeze(_np.einsum("...ij,...kj->...ik",psi_v,psi_v.conj()))
	elif return_rdm == "B":
		return _np.squeeze(_np.einsum("...ji,...jk->...ik",psi_v,psi_v.conj()))
	elif return_rdm == "both":
		return _np.squeeze(_np.einsum("...ij,...kj->...ik",psi_v,psi_v.conj())),_np.squeeze(_np.einsum("...ji,...jk->...ik",psi_v,psi_v.conj()))






def _lattice_partial_trace_mixed(rho,sub_sys_A,L,sps,return_rdm="A"):
	extra_dims = rho.shape[:-2]
	n_dims = len(extra_dims)

	sub_sys_B = set(range(L))-set(sub_sys_A)

	sub_sys_A = tuple(sub_sys_A)
	sub_sys_B = tuple(sorted(sub_sys_B))

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
	rho_v = rho_v.reshape(extra_dims+(Ns_A,Ns_B,Ns_A,Ns_B)) 

	if return_rdm == "A":
		return _np.squeeze(_np.einsum("...jlkl->...jk",rho_v))
	elif return_rdm == "B":
		return _np.squeeze(_np.einsum("...ljlk->...jk",rho_v))
	elif return_rdm == "both":
		return _np.squeeze(_np.einsum("...jlkl->...jk",rho_v)),_np.squeeze(_np.einsum("...ljlk->...jk",rho_v))





def _lattice_partial_trace_sparse_pure(psi,sub_sys_A,L,sps,return_rdm="A"):
	sub_sys_B = set(range(L))-set(sub_sys_A)

	sub_sys_A = tuple(sub_sys_A)
	sub_sys_B = tuple(sorted(sub_sys_B))

	L_A = len(sub_sys_A)
	L_B = len(sub_sys_B)

	Ns_A = (sps**L_A)
	Ns_B = (sps**L_B)

	psi = psi.tocoo()

	T_tup = sub_sys_A+sub_sys_B

	
	# reshuffle indices for the sub-systems.
	# j = sum( j[i]*(sps**i) for i in range(L))
	# this reshuffles the j[i] similar to the transpose operation
	# on the dense arrays psi_v.transpose(T_tup)
	if T_tup != tuple(range(L)):
		indx = _np.zeros(psi.col.shape,dtype=psi.col.dtype)
		for i_new,i_old in enumerate(T_tup):
			indx += ((psi.col//(sps**i_old)) % sps)*(sps**i_new)
	else:
		indx = psi.col


	# make shift way of reshaping array
	# j = j_A + Ns_A * j_B
	# j_A = j % Ns_A
	# j_B = j / Ns_A

	psi._shape = (Ns_A,Ns_B)
	psi.row[:] = indx % Ns_A
	psi.col[:] = indx / Ns_A

	psi = psi.tocsr()

	if return_rdm == "A":
		return psi.dot(psi.H)
	elif return_rdm == "B":
		return psi.H.dot(psi)
	elif return_rdm == "both":
		return psi.dot(psi.H),psi.H.dot(psi)

