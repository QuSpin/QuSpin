from ._basis_1d_core import hcp_basis,hcp_ops
from .base_1d import basis_1d
from ..base import basis
import numpy as _np



class spinless_fermion_basis_1d(basis_1d):
	"""Constructs basis for spinless fermionic operators in a specified 1-d symmetry sector.

	The supported operator strings for `spinless_fermion_basis_1d` are:

	.. math::
			\\begin{array}{cccc}
				\\texttt{basis}/\\texttt{opstr}   &   \\texttt{"I"}   &   \\texttt{"+"}   &   \\texttt{"-"}  &   \\texttt{"n"}   &   \\texttt{"z"}    \\newline	
				\\texttt{spinless_fermion_basis_1d}& \\hat{1}        &   \\hat c^\\dagger      &       \\hat c          & \\hat c^\\dagger c     &  \\hat c^\\dagger\\hat c - \\frac{1}{2}      \\newline
			\\end{array}

	Examples
	--------

	The code snippet below shows how to use the `spinless_fermion_basis_1d` class to construct the basis in the zero momentum sector of positive parity for the fermion Hamiltonian 

	.. math::
		H(t)=-J\\sum_j c^\\dagger_{j+1}c_j + \\mathrm{h.c.} - \\mu\\sum_j n_j + U\\cos\\Omega t\\sum_j n_{j+1} n_j

	.. literalinclude:: ../../doc_examples/spinless_fermion_basis_1d-example.py
		:linenos:
		:language: python
		:lines: 7-

	"""	
	def __init__(self,L,Nf=None,nf=None,**blocks):
		"""Intializes the `fermion_basis_1d` object (basis for fermionic operators).

		Parameters
		-----------
		L: int
			Length of chain/number of sites.
		Nf: {int,list}, optional
			Number of fermions in chain. Can be integer or list to specify one or more particle sectors.
		nf: float, optional
			Density of fermions in chain (fermions per site).
		**blocks: optional
			extra keyword arguments which include:

				**a** (*int*) - specifies unit cell size for translation.

				**kblock** (*int*) - specifies momentum block.

				**pblock** (*int*) - specifies parity block.

				**cblock** (*int*) - specifies particle-hole symmetry block.

				**pcblock** (*int*) - specifies parity followed by particle-hole symmetry block.

				**cAblock** (*int*) - specifies particle-hole symmetry block for sublattice A.

				**cBblock** (*int*) - specifies particle-hole symmetry block for sublattice B.

		"""

		input_keys = set(blocks.keys())

		expected_keys = set(["_Np","kblock","pblock","a","count_particles","check_z_symm","L"])
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

		if Nf is not None and nf is not None:
			raise ValueError("Cannot Nf and nf simultaineously.")
		elif Nf is None and nf is not None:
			if nf < 0 or nf > 1:
				raise ValueError("nf must be between 0 and 1")
			Nf = int(nf*L)


		self._sps = 2
		Imax = (1<<L)-1
		stag_A = sum(1<<i for i in range(0,L,2))
		stag_B = sum(1<<i for i in range(1,L,2))
		pars = _np.array([1,L,Imax,stag_A,stag_B]) # sign to be calculated
		self._operators = ("availible operators for ferion_basis_1d:"+
							"\n\tI: identity "+
							"\n\t+: raising operator"+
							"\n\t-: lowering operator"+
							"\n\tn: number operator"+
							"\n\tz: c-symm number operator")

		self._allowed_ops = set(["I","+","-","n","z"])
		basis_1d.__init__(self,hcp_basis,hcp_ops,L,Np=Nf,_Np=_Np,pars=pars,**blocks)
		# self._check_symm=None

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
