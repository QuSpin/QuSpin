from ._constructors import hcp_basis,hcp_ops
from ._constructors import boson_basis,boson_ops
from .base_1d import basis_1d
import numpy as _np

try:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in xrange(1,10001)}
except NameError:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in range(1,10001)}

class spin_basis_1d(basis_1d):
	"""Basis for spin operators

	"""	
	def __init__(self,L,Nup=None,m=None,S="1/2",pauli=True,**blocks):
		""" Intializes the `fermion_basis_1d` object (basis for fermionic operators).

		Parameters
		----------

		L: int
			length of chain/number of sites

		Nup: {int,list}, optional
			total :math:`S^z` projection, can be integer or list to specify one or more particle sectors.

		m: float, optional
			density of fermions to put on chain.

		S: str, optional
			size of local spin degrees of freedom. could be one of the following:
			"1/2","1","3/2",...,"9999/2","5000"

		pauli: bool, optional
			for S=1/2, switch between standard spin-1/2 and pauli-matrices.

		**blocks: optional
			extra keyword arguements which include:

				**a** (*int*) - specify how many sites to step for translation.

				**kblock** (*int*) - specify momentum block

				**pblock** (*int*) - specify parity block

				**zblock** (*int*) - specify spin inversion symmetry block.

				**zAblock** (*int*) - specify spin inversion of sublattice A symmetry block

				**zAblock** (*int*) - specify spin inversion of sublattice B symmetry block

		"""
		input_keys = set(blocks.keys())
		expected_keys = set(["_Np","kblock","zblock","zAblock","zBblock","pblock","pzblock","a","count_particles","check_z_symm","L"])
		wrong_keys = input_keys - expected_keys 
		if wrong_keys:
			temp = ", ".join(["{}" for key in wrong_keys])
			raise ValueError(("unexpected optional argument(s): "+temp).format(*wrong_keys))


		if blocks.get("a") is None: # by default a = 1
			blocks["a"] = 1

		_Np = blocks.get("_Np")
		if _Np is not None:
			blocks.pop("_Np")


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

		self._sps,S = S_dict[S]

		if Nup is not None and m is not None:
			raise ValueError("Cannot use Nup and m at the same time")
		elif Nup is None and m is not None:
			if m < -S or m > S:
				raise ValueError("N must be between -S and S")

			Nup = int((m+S)*L)



		if self._sps <= 2:
			self._pauli = pauli
			Imax = (1<<L)-1
			stag_A = sum(1<<i for i in range(0,L,2))
			stag_B = sum(1<<i for i in range(1,L,2))
			pars = _np.array([0,L,Imax,stag_A,stag_B])
			self._operators = ("availible operators for spin_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tx: x pauli/spin operator"+
								"\n\ty: y pauli/spin operator"+
								"\n\tz: z pauli/spin operator")

			self._allowed_ops = set(["I","+","-","x","y","z"])
			basis_1d.__init__(self,hcp_basis,hcp_ops,L,Np=Nup,_Np=_Np,pars=pars,**blocks)
		else:
			self._pauli = False
			pars = (L,) + tuple(self._sps**i for i in range(L+1)) + (1,) # flag to turn off higher spin matrix elements for +/- operators
			self._operators = ("availible operators for spin_basis_1d:"+
								"\n\tI: identity "+
								"\n\t+: raising operator"+
								"\n\t-: lowering operator"+
								"\n\tz: z pauli/spin operator")

			self._allowed_ops = set(["I","+","-","z"])
			basis_1d.__init__(self,boson_basis,boson_ops,L,Np=Nup,_Np=_Np,pars=pars,**blocks)


	def Op(self,opstr,indx,J,dtype):
		ME,row,col = basis_1d.Op(self,opstr,indx,J,dtype)
		if self._pauli:
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



