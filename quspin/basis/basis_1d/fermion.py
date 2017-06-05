from ._constructors import hcp_basis,hcp_ops
from .base_1d import basis_1d
import numpy as _np



class fermion_basis_1d(basis_1d):
	def __init__(self,L,Nf=None,nf=None,_Np=None,**blocks):
		input_keys = set(blocks.keys())

		expected_keys = set(["kblock","cblock","cAblock","cBblock","pblock","pcblock","a","count_particles","check_z_symm","L"])
		wrong_keys = input_keys - expected_keys 
		if wrong_keys:
			temp = ", ".join(["{}" for key in wrong_keys])
			raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))


		if blocks.get("a") is None: # by default a = 1
			blocks["a"] = 1

		self._blocks = blocks

		pblock = blocks.get("pblock")
		zblock = blocks.get("cblock")
		zAblock = blocks.get("cAblock")
		zBblock = blocks.get("cBblock")

		if type(zblock) is int:
			del blocks["cblock"]
			blocks["zblock"] = zblock

		if type(zAblock) is int:
			del blocks["cAblock"]
			blocks["zAblock"] = zAblock

		if type(zBblock) is int:
			del blocks["cBblock"]
			blocks["zBblock"] = zBblock

		if (type(pblock) is int) and (type(zblock) is int):
			blocks["pzblock"] = pblock*zblock
			self._blocks["pcblock"] = pblock*zblock

		if (type(zAblock) is int) and (type(zBblock) is int):
			blocks["zblock"] = zAblock*zBblock
			self._blocks["cblock"] = zAblock*zBblock

		if Nf is not None and nf is not None:
			raise ValueError("Cannot Nf and nf simultaineously.")
		elif Nf is None and nf is not None:
			if nf < 0 or nf > 1:
				raise ValueError("nf must be between 0 and 1")
			Nf = int(nf*L)


		self._sps = 2

		pars = _np.array([1,L]) # set sign to be calculated
		self._operators = ("availible operators for ferion_basis_1d:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tn: number operator"+
							"\n\tz: c-symm number operator")

		self._allowed_ops = set(["I","+","-","n","z"])
		basis_1d.__init__(self,hcp_basis,hcp_ops,L,Np=Nf,_Np=_Np,pars=pars,**blocks)



	@property
	def blocks(self):
		return dict(self._blocks)

	def __type__(self):
		return "<type 'qspin.basis.fermion_basis_1d'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.fermion_basis_1d' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.fermion_basis_1d'>"


	# functions called in base class:

	def _sort_opstr(self,op):
		if op[0].count("|") > 0:
			raise ValueError("'|' character found in op: {0},{1}".format(op[0],op[1]))
		if len(op[0]) != len(op[1]):
			raise ValueError("number of operators in opstr: {0} not equal to length of indx {1}".format(op[0],op[1]))

		op = list(op)
		zipstr = list(zip(op[0],op[1]))
		if zipstr:
			n = len(zipstr)
			swapped = True
			anticommutes = 0
			while swapped:
				swapped = False
				for i in range(n-1):
					if zipstr[i][1] > zipstr[i+1][1]:
						temp = zipstr[i]
						zipstr[i] = zipstr[i+1]
						zipstr[i+1] = temp
						swapped = True

						if zipstr[i][0] in ["+","-"] and zipstr[i+1][0] in ["+","-"]:
							anticommutes += 1

			op1,op2 = zip(*zipstr)
			op[0] = "".join(op1)
			op[1] = tuple(op2)
			op[2] *= (1 if anticommutes%2 == 0 else -1)
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
		op = list(op)
		op.append(num)
		return [tuple(op)]	

