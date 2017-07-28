from .base_hcb import hcb_basis_general
from .base_higher_spin import higher_spin_basis_general
import numpy as _np

try:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in xrange(1,10001)}
except NameError:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in range(1,10001)}


# general basis for hardcore bosons/spin-1/2
class spin_basis_general(hcb_basis_general,higher_spin_basis_general):
	def __init__(self,N,Nup=None,m=None,S="1/2",pauli=True,_Np=None,**kwargs):
		self._S = S
		self._pauli = pauli
		sps,S = S_dict[S]

		if Nup is not None and m is not None:
			raise ValueError("Cannot use Nup and m at the same time")
		if m is not None and Nup is None:
			if m < -S or m > S:
				raise ValueError("m must be between -S and S")

			Nup = int((m+S)*L)

		if sps==2:
			hcb_basis_general.__init__(self,N,Nb=Nup,_Np=_Np,**kwargs)
		else:
			higher_spin_basis_general.__init__(self,N,Nup=Nup,sps=sps,_Np=_Np,**kwargs)

	def Op(self,opstr,indx,J,dtype):
		
		if self._S == "1/2":
			ME,row,col = hcb_basis_general.Op(self,opstr,indx,J,dtype)
			if self._pauli:
				n = len(opstr.replace("I",""))
				ME *= (1<<n)
		else:
			return higher_spin_basis_general.Op(self,opstr,indx,J,dtype)

		return ME,row,col

	def __type__(self):
		return "<type 'qspin.basis.general_hcb'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.general_hcb' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.general_hcb'>"


	# functions called in base class:


	def _sort_opstr(self,op):
		if op[0].count("|") > 0:
			raise ValueError("'|' character found in op: {0},{1}".format(op[0],op[1]))
		if len(op[0]) != len(op[1]):
			raise ValueError("number of operators in opstr: {0} not equal to length of indx {1}".format(op[0],op[1]))

		op = list(op)
		zipstr = list(zip(op[0],op[1]))
		if zipstr:
			zipstr.sort(key = lambda x:x[1])
			op1,op2 = zip(*zipstr)
			op[0] = "".join(op1)
			op[1] = tuple(op2)
		return tuple(op)



	def _non_zero(self,op):
		opstr = _np.array(list(op[0]))
		indx = _np.array(op[1])
		if _np.any(indx):
			indx_p = indx[opstr == "+"].tolist()
			p = not any(indx_p.count(x) > 1 for x in indx_p)
			indx_p = indx[opstr == "-"].tolist()
			m = not any(indx_p.count(x) > 1 for x in indx_p)
			return (p and m)
		else:
			return True
		


	def _hc_opstr(self,op):
		op = list(op)
		# take h.c. + <--> - , reverse operator order , and conjugate coupling
		op[0] = list(op[0].replace("+","%").replace("-","+").replace("%","-"))
		op[0].reverse()
		op[0] = "".join(op[0])
		op[1] = list(op[1])
		op[1].reverse()
		op[1] = tuple(op[1])
		op[2] = op[2].conjugate()
		return self._sort_opstr(op) # return the sorted op.


	def _expand_opstr(self,op,num):
		opstr = str(op[0])
		indx = list(op[1])
		J = op[2]
 
		if len(opstr) <= 1:
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
	 
			i = len(opstr)//2
			op1 = list(op)
			op1[0] = opstr[:i]
			op1[1] = tuple(indx[:i])
			op1[2] = complex(J)
			op1 = tuple(op1)

			op2 = list(op)
			op2[0] = opstr[i:]
			op2[1] = tuple(indx[i:])
			op2[2] = complex(1)
			op2 = tuple(op2)

			l1 = self._expand_opstr(op1,num)
			l2 = self._expand_opstr(op2,num)

			l = []
			for op1 in l1:
				for op2 in l2:
					op = list(op1)
					op[0] += op2[0]
					op[1] += op2[1]
					op[2] *= op2[2]
					l.append(tuple(op))

			return tuple(l)




