from ._constructors import hcp_basis_ops
from ._constructors import boson_basis_ops
from .base_1d import basis_1d
import numpy as _np



S_dict = {(str(i)+"/2" if i%2==1 else str(i/2)):i+1 for i in xrange(1,1001)}



class boson_basis_1d(basis_1d):
	def __init__(self,L,Nb=None,Nmax_ps=None,_Np=None,**blocks):
		input_keys = set(blocks.keys())

		expected_keys = set(["kblock","cblock","cAblock","cBblock","pblock","pcblock","a","count_particles","check_z_symm","L"])
		wrong_keys = input_keys - expected_keys 
		if wrong_keys:
			temp = ", ".join(["{}" for key in wrong_keys])
			raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))

		if Nb is None and Nmax_ps is None:
			raise ValueError("Must specify either Nb or Nmax_ps")

		if Nmax_ps is not None:
			if type(Nmax_ps) is not int:
				raise TypeError("Nmax_ps must be integer >= 1.")
			if Nmax_ps < 1:
				raise ValueError("Nmax_ps must be >= 1 or None")


		if type(Nb) is int and Nmax_ps is None:
			Nmax_ps = Nb

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

		self._m = Nmax_ps + 1

		if self._m <= 2:
			pars = _np.array([0]) # set sign to not be calculated
			self._operators = ("availible operators for boson_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tn: number operator"+
								"\n\tz: c-symm number operator")

			self._allowed_ops = set(["I","+","-","n","z"])
			basis_1d.__init__(self,hcp_basis_ops,L,Np=Nb,_Np=_Np,pars=pars,**blocks)
		else:
			pars = [L]
			pars.extend([self._m**i for i in range(L+1)])
			pars = _np.asarray(pars)
			self._operators = ("availible operators for ferion_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tn: number operator"+
								"\n\tz: ph-symm number operator")

			self._allowed_ops = set(["I","+","-","z"])
			basis_1d.__init__(self,boson_basis_ops,L,Np=Nb,_Np=_Np,pars=pars,**blocks)


	@property
	def blocks(self):
		return dict(self._blocks)

	def __type__(self):
		return "<type 'qspin.basis.boson_basis_1d'>"

	def __repr__(self):
		return "< instance of 'qspin.basis.boson_basis_1d' with {0} states >".format(self._Ns)

	def __name__(self):
		return "<type 'qspin.basis.boson_basis_1d'>"


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
		op = list(op)
		op.append(num)
		return [tuple(op)]	


