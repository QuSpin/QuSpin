import warnings

class SymmException(Exception):
	pass


def sort_opstr(opstr,indx):
	opstr = list(opstr)
	indx = list(indx)
	zipstr = list(zip(opstr,indx))
	zipstr.sort(key = lambda x:(x[1],x[0]))
	new_opstr,new_indx = zip(*zipstr)
	return "".join(new_opstr),new_indx


def flip_sublat(opstr,indx,lat=0):
	sign = 1
	opstr = [str(s) for s in opstr]
	for s,i,j in zip(opstr,indx,range(len(indx))):
		if ((i % 2) == (lat % 2)):
			if (s in ['z','y']):
				sign *= -1
			elif (s == "+"):
				opstr[j] = '-'
			elif (s == "-"):
				opstr[j] = '+'

	return sign,"".join(opstr)





def check_T(operator_list,L,a):
	missing_ops=[]
	for i in xrange(L/a-1,0,-1):
		for opstr,indx,J in operator_list:
			indx = list(indx)
			for j,ind in enumerate(indx):
				indx[j] = (ind+i*a)%L
			

			new_opstr,new_indx = sort_opstr(opstr,indx)
			
			if not ([new_opstr,tuple(new_indx),J] in operator_list):
				missing_ops.append((new_opstr,new_indx,J))

	return missing_ops




def check_Z(operator_list):
	missing_ops=[]
	odd_ops=[]
	for opstr,indx,J in operator_list:
		z_count = opstr.count("z")
		y_count = opstr.count("y")
		if ((y_count + z_count) % 2) != 0:
			odd_ops.append((new_opstr,new_indx,J))

		new_opstr = opstr.replace("+","#").replace("-","+").replace("#","-")
		new_new_opstr,new_indx = sort_opstr(new_opstr,indx)
		if not ([new_new_opstr,new_indx,J] in operator_list):
			missing_ops.append((new_opstr,new_indx,J))

	return odd_ops,missing_ops



def check_P(operator_list,L):
	missing_ops = []
	for opstr,indx,J in operator_list:
		indx = list(indx)
		for j,ind in enumerate(indx):
			indx[j] = (L-1-ind) % L

		new_opstr,new_indx = sort_opstr(opstr,indx)
		if not ([new_opstr,new_indx,J] in operator_list):
			missing_ops.append((new_opstr,new_indx,J))

	return missing_ops




def check_PZ(operator_list,L):
	missing_ops = []
	for opstr,indx,J in operator_list:
		indx = list(indx)
		for j,ind in enumerate(indx):
			indx[j] = (L-1-ind) % L
		
		sign = (-1)**(opstr.count('z')+opstr.count('y'))
		
		new_opstr = opstr.replace("+","#").replace("-","+").replace("#","-")
		new_new_opstr,new_indx = sort_opstr(new_opstr,indx)
		if not ([new_new_opstr,new_indx,sign*J] in operator_list):
			missing_ops.append((new_opstr,new_indx,J))

	return missing_ops






def check_ZA(operator_list):
	missing_ops=[]
	odd_ops=[]

	for opstr,indx,J in operator_list:

		sign,new_opstr = flip_sublat(opstr,indx,lat=0)

		if sign == -1:
			odd_ops.append((new_opstr,new_indx,J))

		new_new_opstr,new_indx = sort_opstr(new_opstr,indx)

		if not ([new_new_opstr,new_indx,sign*J] in operator_list):
			missing_ops.append((new_opstr,new_indx,J))
	


def check_ZB(operator_list):
	missing_ops=[]
	odd_ops=[]

	for opstr,indx,J in operator_list:

		sign,new_opstr = flip_sublat(opstr,indx,lat=1)

		if sign == -1:
			odd_ops.append((new_opstr,new_indx,J))

		new_new_opstr,new_indx = sort_opstr(new_opstr,indx)

		if not ([new_new_opstr,new_indx,sign*J] in operator_list):
			missing_ops.append((new_opstr,new_indx,J))





	





def check_symm(static,dynamic,L,**blocks):
	from numpy import pi,euler_gamma,cos,exp
	kblock = blocks.get("kblock")
	pblock = blocks.get("pblock")
	zblock = blocks.get("zblock")
	pzblock = blocks.get("pzblock")
	zAblock = blocks.get("zAblock")
	zBblock = blocks.get("zBblock")
	a = blocks.get("a")

	t = cos( (pi/exp(0))**( 1.0/euler_gamma ) )

	static_list = []
	for opstr,bonds in static:
		for indx in bonds:
			J = complex(indx[0])
			indx = list(indx[1:])
			new_opstr,new_indx = sort_opstr(opstr,indx)
			static_list.append([new_opstr,tuple(new_indx),J])


	dynamic_list = []
	for opstr,bonds,f,f_args in dynamic:
		for indx in bonds:
			J = complex(indx[0]*f(t,*f_args))
			indx = list(indx[1:])
			new_opstr,new_indx = sort_opstr(opstr,indx)
			dynamic_list.append([new_opstr,tuple(new_indx),J])

	static_blocks = {}
	if kblock is not None:
		missing_ops = check_T(static_list,L,a)
		if missing_ops:	static_blocks["T symm"] = (missing_ops,)
	if pblock is not None:
		missing_ops = check_P(static_list,L)
		if missing_ops:	static_blocks["P symm"] = (missing_ops,)
	if zblock is not None:
		odd_ops,missing_ops = check_Z(static_list)
		if missing_ops or odd_ops:
			static_blocks["Z symm"] = (odd_ops,missing_ops)
	if zAblock is not None:
		odd_ops,missing_ops = check_ZA(static_list)
		if missing_ops or odd_ops:
			static_blocks["ZA symm"] = (odd_ops,missing_ops)
	if zBblock is not None:
		odd_ops,missing_ops = check_ZB(static_list)
		if missing_ops or odd_ops:
			static_blocks["ZB symm"] = (odd_ops,missing_ops)
	if pzblock is not None:
		missing_ops = check_PZ(static_list,L)
		if missing_ops:	static_blocks["PZ symm"] = (missing_ops,)


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
				user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(ops)) )
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
				raise TypeError("Hamiltonian does not obey {0}!".format(symm))


		elif len(static_blocks[symm]) == 1:
			missing_ops, = static_blocks[symm]
			unique_opstrs = list(set( zip(*tuple(missing_ops))[0]) )
			if unique_opstrs:
				unique_missing_ops = []
				[ unique_missing_ops.append(ele) for ele in missing_ops if ele not in unique_missing_ops]
				warnings.warn("The following static operator strings do not obey {0}: {1}".format(symm,unique_opstrs),UserWarning,stacklevel=4)
				user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(missing_ops)) )
				if user_input == 'y':
					print " these operators are needed for {}:".format(symm)
					print "   (opstr, indices, coupling)"
					for i,op in enumerate(unique_missing_ops):
						print "{0}. {1}".format(i+1, op)
				raise TypeError("Hamiltonian does not obey {0}!".format(symm))
		else:
			continue


	dynamic_blocks = {}
	if kblock is not None:
		missing_ops = check_T(dynamic_list,L,a)
		if missing_ops:	dynamic_blocks["T symm"] = (missing_ops,)
	if pblock is not None:
		missing_ops = check_P(dynamic_list,L)
		if missing_ops:	dynamic_blocks["P symm"] = (missing_ops,)
	if zblock is not None:
		odd_ops,missing_ops = check_Z(dynamic_list)
		if missing_ops or odd_ops:
			dynamic_blocks["Z symm"] = (odd_ops,missing_ops)
	if zAblock is not None:
		odd_ops,missing_ops = check_ZA(dynamic_list)
		if missing_ops or odd_ops:
			dynamic_blocks["ZA symm"] = (odd_ops,missing_ops)
	if zBblock is not None:
		odd_ops,missing_ops = check_ZB(dynamic_list)
		if missing_ops or odd_ops:
			dynamic_blocks["ZB symm"] = (odd_ops,missing_ops)
	if pzblock is not None:
		missing_ops = check_PZ(dynamic_list,L)
		if missing_ops:	dynamic_blocks["PZ symm"] = (missing_ops,)
	

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
				user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(ops)) )
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
				raise TypeError("Hamiltonian does not obey {0}!".format(symm))


		elif len(dynamic_blocks[symm]) == 1:
			missing_ops, = dynamic_blocks[symm]
			unique_opstrs = list(set( zip(*tuple(missing_ops))[0]) )
			if unique_opstrs:
				unique_missing_ops = []
				[ unique_missing_ops.append(ele) for ele in missing_ops if ele not in unique_missing_ops]
				warnings.warn("The following dynamic operator strings do not obey {0}: {1}".format(symm,unique_opstrs),UserWarning,stacklevel=4)
				user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(missing_ops)) )
				if user_input == 'y':
					print " these operators are needed for {}:".format(symm)
					print "   (opstr, indices, coupling)"
					for i,op in enumerate(unique_missing_ops):
						print "{0}. {1}".format(i+1, op)
				raise TypeError("Hamiltonian does not obey {0}!".format(symm))
		else:
			continue

	print "Symmetry checks passed!"







