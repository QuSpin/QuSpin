from ._constructors import hcp_basis_ops
from ._constructors import boson_basis_ops
from .base_1d import basis_1d
import numpy as _np



S_dict = {(str(i)+"/2" if i%2==1 else str(i/2)):i+1 for i in xrange(1,1001)}



class spin_basis_1d(basis_1d):
	def __init__(self,L,Nup=None,_Np=None,S="1/2",pauli=False,**blocks):
		input_keys = set(blocks.keys())

		expected_keys = set(["kblock","zblock","zAblock","zBblock","pblock","pzblock","a","count_particles","check_z_symm","L"])
		wrong_keys = input_keys - expected_keys 
		if wrong_keys:
			temp = ", ".join(["{}" for key in wrong_keys])
			raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))


		if blocks.get("a") is None: # by default a = 1
			blocks["a"] = 1

		self._blocks = blocks
		
		pblock = blocks.get("pblock")
		zblock = blocks.get("zblock")
		zAblock = blocks.get("zAblock")
		zBblock = blocks.get("zBblock")

		if (type(pblock) is int) and (type(zblock) is int):
			blocks["pzblock"] = pblock*zblock
			self._blocks["pzblock"] = pblock*zblock

		if (type(zAblock) is int) and (type(zBblock) is int):
			blocks["zblock"] = zAblock*zBblock
			self._blocks["zblock"] = zAblock*zBblock

		self._sps = S_dict[S]

		if self._sps <= 2:
			self._pauli = pauli
			pars = _np.array([0])
			self._operators = ("availible operators for spin_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tx: x pauli/spin operator"+
								"\n\ty: y pauli/spin operator"+
								"\n\tz: z pauli/spin operator")

			self._allowed_ops = set(["I","+","-","x","y","z"])
			basis_1d.__init__(self,hcp_basis_ops,L,Np=Nup,_Np=_Np,pars=pars,**blocks)
		else:
			self._pauli = False
			pars = [L]
			pars.extend([self._sps**i for i in range(L+1)])
			pars.append(1) # flag to turn on higher spin matrix elements for +/- operators
			pars = _np.asarray(pars)
			self._operators = ("availible operators for spin_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tz: z pauli/spin operator")

			self._allowed_ops = set(["I","+","-","z"])
			basis_1d.__init__(self,boson_basis_ops,L,Np=Nup,_Np=_Np,pars=pars,**blocks)


	def Op(self,opstr,indx,J,dtype):
		ME,row,col = basis_1d.Op(self,opstr,indx,J,dtype)

		if not self._pauli:
			n_ops = len(opstr.replace("I",""))
			ME *= (1<<n_ops)

		return ME,row,col


	@property
	def blocks(self):
		return dict(self._blocks)

	def __type__(self):
		return "<type 'qspin.basis.spin_basis_1d'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.spin_basis_1d' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.spin_basis_1d'>"


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



