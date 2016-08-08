import warnings

class SymmException(Exception):
	pass







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





def check_T(basis,operator_list,L,a):
	missing_ops=[]
	for i in xrange(0,L/a,1):
		for op in operator_list:
			opstr = str(op[0])
			indx = list(op[1])
			for j,ind in enumerate(indx):
				indx[j] = (ind+i*a)%L
			
			new_op = list(op)
			new_op[1] = indx

			new_op = basis.sort_opstr(new_op)
			
			if not (new_op in operator_list):
				missing_ops.append(new_op)

	return missing_ops




def check_Z(basis,operator_list):
	missing_ops=[]
	odd_ops=[]
	for op in operator_list:
		opstr = str(op[0])
		indx = list(op[1])

		if opstr.count("|") == 1:
			i = opstr.index("|")
		else:
			i = len(opstr)

		z_count = opstr[:i].count("z")
		y_count = opstr[:i].count("y")
		if ((y_count + z_count) % 2) != 0:
			odd_ops.append(op)

		new_op = list(op)
		new_op[0] = new_op[0][:i].replace("+","#").replace("-","+").replace("#","-") + op[0][i:]
		new_op = basis.sort_opstr(new_op)
		if not (new_op in operator_list):
			missing_ops.append(new_op)

	return odd_ops,missing_ops



def check_P(basis,operator_list,L):
	missing_ops = []
	for op in operator_list:
		indx = list(op[1])
		for j,ind in enumerate(indx):
			indx[j] = (L-1-ind) % L

		new_op = list(op)
		new_op[1] = indx
		new_op = basis.sort_opstr(new_op)
		if not (new_op in operator_list):
			missing_ops.append(new_op)

	return missing_ops




def check_PZ(basis,operator_list,L):
	missing_ops = []
	for op in operator_list:
		opstr = str(op[0])
		indx = list(op[1])

		if opstr.count("|") == 1:
			i = opstr.index("|")
		else:
			i = len(opstr)

		for j,ind in enumerate(indx):
			indx[j] = (L-1-ind) % L
		
		sign = (-1)**(opstr[:i].count('z')+opstr.count('y'))
	
		new_op = list(op)
		new_op[0] = new_op[0][:i].replace("+","#").replace("-","+").replace("#","-") + op[0][i:]
		new_op[1] = indx
		new_op[2] *= sign
		new_op = basis.sort_opstr(new_op)
		if not (new_op in operator_list):
			missing_ops.append(new_op)

	return missing_ops






def check_ZA(basis,operator_list):
	missing_ops=[]
	odd_ops=[]

	for op in operator_list:

		opstr = str(op[0])
		indx = list(op[1])

		if opstr.count("|") == 1:
			i = opstr.index("|")
		else:
			i = len(opstr)

		sign,new_opstr = flip_sublat(opstr[:i],indx[:i],lat=0)

		if sign == -1:
			odd_ops.append(op)

		new_op = list(op)
		new_op[0] = new_opstr + opstr[i:]
		new_op = basis.sort_opstr(new_op)
		

		if not (new_op in operator_list):
			missing_ops.append(new_op)

	return odd_ops,missing_ops



def check_ZB(basis,operator_list):
	missing_ops=[]
	odd_ops=[]

	for op in operator_list:

		opstr = str(op[0])
		indx = list(op[1])

		if opstr.count("|") == 1:
			i = opstr.index("|")
		else:
			i = len(opstr)

		sign,new_opstr = flip_sublat(opstr[:i],indx[:i],lat=1)

		if sign == -1:
			odd_ops.append(op)

		new_op = list(op)
		new_op[0] = new_opstr + opstr[i:]
		new_op = basis.sort_opstr(new_op)

		if not (new_op in operator_list):
			missing_ops.append(new_op)

	return odd_ops,missing_ops







def _remove_xy_opstr(op,num):
	opstr = str(op[0])
	indx = list(op[1])
	J = op[2]

	if len(opstr.replace("|","")) <= 1:
		if opstr == "x":
			op1 = list(op)
			op1[0] = op1[0].replace("x","+")
			op1[2] *= 0.5
			op1.append(num)

			op2 = list(op)
			op2[0] = op2[0].replace("x","-")
			op2[2] *= 0.5
			op2.append(num)

			return (tuple(op1),tuple(op2))
		elif opstr == "y":
			op1 = list(op)
			op1[0] = op1[0].replace("y","+")
			op1[2] *= -0.5j
			op1.append(num)

			op2 = list(op)
			op2[0] = op2[0].replace("y","-")
			op2[2] *= 0.5j
			op2.append(num)

			return (tuple(op1),tuple(op2))
		else:
			op = list(op)
			op.append(num)
			return [tuple(op)]	
	else:
 
		i = len(opstr)/2
		i_op = i + opstr[:i].count("|")

		op1 = list(op)
		op1[0] = opstr[:i_op]
		op1[1] = indx[:i]
		op1[2] = complex(J)
		op1 = tuple(op1)

		op2 = list(op)
		op2[0] = opstr[i_op:]
		op2[1] = indx[i:]
		op2[2] = complex(1)
		op2 = tuple(op2)

		l1 = remove_xy(op1,num)
		l2 = remove_xy(op2,num)
		l = []
		for op1 in l1:
			for op2 in l2:
				op = list(op1)
				op[0] += op2[0]
				op[1] += op2[1]
				op[2] *= op2[2]
				l.append(tuple(op))
				
	
		return tuple(l)


def remove_xy_list(op_list):
	op_list_exp = []
	for i,op in enumerate(op_list):
		new_ops = remove_xy(op,[i])
		op_list_exp.extend(new_ops)




def check_pcon(static,dynamic):
	static_list = []
	for opstr,bonds in static:
		for indx in bonds:
			J = complex(indx[0])
			indx = list(indx[1:])
			static_list.append([opstr,indx,J])

	static_list_exp = []
	for i,op in enumerate(static_list):
		op_list = remove_xy(op,[i])
		static_list_exp.extend(op_list)



	l = len(static_list_exp)
	i = 0

	while (i < l):
		j = 0
		while (j < l):
			if i != j:
				opstr1,indx1,J1,i1 = tuple(static_list_exp[i]) 
				opstr2,indx2,J2,i2 = tuple(static_list_exp[j])
				if opstr1 == opstr2 and indx1 == indx2:

					del static_list_exp[j]
					if J1 == -J2: 
						if j < i: i -= 1
						del static_list_exp[i]
					else:
						static_list_exp[i] = list(static_list_exp[i])
						static_list_exp[i][2] += J2
						static_list_exp[i][3].extend(i2)
						static_list_exp[i] = tuple(static_list_exp[i])
						
					l = len(static_list_exp)

					if i >= l: break
			j += 1
		i += 1


	k = 0
	while(k < len(static_list_exp)):
		opstr,indx,J,ii = tuple(static_list_exp[k])

		if opstr.count("|") == 1:
			ii = opstr.index("|")
		else:
			ii = len(opstr)

		i = len(opstr)
		indx = indx[:i]
				
		u_opstr = []
		u_indx = []
		i=0
		opstr = list(opstr)
		for i,j in enumerate(indx[:ii]):
			if j not in u_indx:
				u_indx.append(j)
				u_opstr.append(opstr[i])
			else:
				ll = u_indx.index(j)
				if opstr[i] == u_opstr[ll]:
					del dynamic_list_exp[k]
					break
		k += 1

			


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
		warnings.warn("The following static operator strings do not conserve Magnetization: {0}".format(unique_opstrs),UserWarning,stacklevel=4)
		user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(odd_ops)) )
		if user_input == 'y':
			print " these operators do not conserve Magnetization:"
			print "   (opstr, indices, coupling)"
			for i,op in enumerate(unique_odd_ops):
				print "{0}. {1}".format(i+1, op)
		raise TypeError("Hamiltonian does not conserve Magnetization! To turn off check, use check_pcon=False in hamiltonian.")


	dynamic_list = []
	for opstr,bonds,f,f_args in dynamic:
		for indx in bonds:
			J = complex(indx[0])
			indx = list(indx[1:])
			dynamic_list.append([opstr,indx,J,f,f_args])

	dynamic_list_exp = []
	for i,op in enumerate(dynamic_list):
		op_list = remove_xy(op,[i])
		print op_list
		dynamic_list_exp.extend(op_list)



	l = len(dynamic_list_exp)
	i = 0

	while (i < l):
		j = 0
		while (j < l):
			if i != j:
				opstr1,indx1,J1,f1,f1_args,i1 = tuple(dynamic_list_exp[i]) 
				opstr2,indx2,J2,f2_f2_args,i2 = tuple(dynamic_list_exp[j])
				if opstr1 == opstr2 and indx1 == indx2 and f1 == f2 and f1_args == f2_args:

					del dynamic_list_exp[j]
					if J1 == -J2: 
						if j < i: i -= 1
						del dynamic_list_exp[i]
					else:
						dynamic_list_exp[i] = list(dynamic_list_exp[i])
						dynamic_list_exp[i][2] += J2
						dynamic_list_exp[i][3].extend(i2)
						dynamic_list_exp[i] = tuple(dynamic_list_exp[i])
						
					l = len(dynamic_list_exp)

					if i >= l: break
			j += 1
		i += 1


	k = 0
	while(k < len(dynamic_list_exp)):
		opstr,indx,J,f,f_args,ii = tuple(dynamic_list_exp[k])

		if opstr.count("|") == 1:
			ii = opstr.index("|")
		else:
			ii = len(opstr)

		u_opstr = []
		u_indx = []
		i=0
		opstr = list(opstr)
		for j,i in enumerate(indx[:ii]):
			if j not in u_indx:
				u_indx.append(j)
				u_opstr.append(opstr[i])
			else:
				ll = u_indx.index(j)
				if opstr[i] == u_opstr[ll]:
					del dynamic_list_exp[k]
					break
		k += 1

			


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
		warnings.warn("The following dynamic operator strings do not conserve Magnetization: {0}".format(unique_opstrs),UserWarning,stacklevel=4)
		user_input = raw_input("Display all {0} couplings? (y or n) ".format(len(odd_ops)) )
		if user_input == 'y':
			print " these operators do not conserve Magnetization:"
			print "   (opstr, indices, coupling)"
			for i,op in enumerate(unique_odd_ops):
				print "{0}. {1}".format(i+1, op)
		raise TypeError("Hamiltonian does not conserve Magnetization! To turn off check, use check_pcon=False in hamiltonian.")

	print "Magnetization checks passed!"





def check_symm(basis,static,dynamic,L):
	from numpy import pi,euler_gamma,cos,exp
	kblock = basis._blocks.get("kblock")
	pblock = basis._blocks.get("pblock")
	zblock = basis._blocks.get("zblock")
	pzblock = basis._blocks.get("pzblock")
	zAblock = basis._blocks.get("zAblock")
	zBblock = basis._blocks.get("zBblock")
	a = basis._blocks.get("a")

	t = cos( (pi/exp(0))**( 1.0/euler_gamma ) )

	static_list,dynamic_list = basis.get_lists(static,dynamic)

	static_blocks = {}
	if kblock is not None:
		missing_ops = check_T(basis,static_list,L,a)
		if missing_ops:	static_blocks["T symm"] = (missing_ops,)
	if pblock is not None:
		missing_ops = check_P(basis,static_list,L)
		if missing_ops:	static_blocks["P symm"] = (missing_ops,)
	if zblock is not None:
		odd_ops,missing_ops = check_Z(basis,static_list)
		if missing_ops or odd_ops:
			static_blocks["Z symm"] = (odd_ops,missing_ops)
	if zAblock is not None:
		odd_ops,missing_ops = check_ZA(basis,static_list)
		if missing_ops or odd_ops:
			static_blocks["ZA symm"] = (odd_ops,missing_ops)
	if zBblock is not None:
		odd_ops,missing_ops = check_ZB(basis,static_list)
		if missing_ops or odd_ops:
			static_blocks["ZB symm"] = (odd_ops,missing_ops)
	if pzblock is not None:
		missing_ops = check_PZ(basis,static_list,L)
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


	dynamic_blocks = {}
	if kblock is not None:
		missing_ops = check_T(basis,dynamic_list,L,a)
		if missing_ops:	dynamic_blocks["T symm"] = (missing_ops,)
	if pblock is not None:
		missing_ops = check_P(basis,dynamic_list,L)
		if missing_ops:	dynamic_blocks["P symm"] = (missing_ops,)
	if zblock is not None:
		odd_ops,missing_ops = check_Z(basis,dynamic_list)
		if missing_ops or odd_ops:
			dynamic_blocks["Z symm"] = (odd_ops,missing_ops)
	if zAblock is not None:
		odd_ops,missing_ops = check_ZA(basis,dynamic_list)
		if missing_ops or odd_ops:
			dynamic_blocks["ZA symm"] = (odd_ops,missing_ops)
	if zBblock is not None:
		odd_ops,missing_ops = check_ZB(basis,dynamic_list)
		if missing_ops or odd_ops:
			dynamic_blocks["ZB symm"] = (odd_ops,missing_ops)
	if pzblock is not None:
		missing_ops = check_PZ(basis,dynamic_list,L)
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

































