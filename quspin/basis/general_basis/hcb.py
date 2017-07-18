from ._general_basis_core import hcb_basis_core_wrap_32,hcb_basis_core_wrap_64
from .general_base import general_basis
import numpy as _np
from scipy.misc import comb

# general basis for hardcore bosons/spin-1/2
class general_hcb(general_basis):
	def __init__(self,N,Np=None,_Np=None,**kwargs):
		general_basis.__init__(self,N,**kwargs)
		self._check_pcon = False
		count_particles = False
		if _Np is not None and Np is None:
			count_particles = True
			if type(_Np) is not int:
				raise ValueError("_Np must be integer")
			if _Np >= -1:
				if _Np+1 > N: 
					Np = list(range(N+1))
				elif _Np==-1:
					Np = None
				else:
					Np = list(range(_Np+1))
			else:
				raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")

		if Np is None:
			Ns = (1<<N)	
		elif type(Np) is int:
			self._check_pcon = True
			Ns = comb(N,Np,exact=True)
		else:
			try:
				Np_iter = iter(Np)
			except TypeError:
				raise TypeError("Np must be integer or iteratable object.")
			Ns = 0
			for np in Np_iter:
				if np > N:
					raise ValueError("particle number Np must satisfy: 0 <= Np <= N")
				Ns += comb(N,np,exact=True)

		Ns = max(int(float(Ns)/_np.multiply.reduce(self._pers))*2,1000)


		if N<=32:
			self._basis = _np.zeros(Ns,dtype=_np.uint32)
			self._n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = hcb_basis_core_wrap_32(N,self._maps,self._rev_maps,self._pers,self._qs)
		elif N<=64:
			self._basis = _np.zeros(Ns,dtype=_np.uint64)
			self._n     = _np.zeros(Ns,dtype=_np.uint16)
			self._core = hcb_basis_core_wrap_64(N,self._maps,self._rev_maps,self._pers,self._qs)
		else:
			raise ValueError("system size N must be <=64.")

		self._sps=2
		if count_particles and (Np is not None):
			self._Np_list = _np.zeros_like(self._basis,dtype=_np.uint8)
			self._Ns = self._core.make_basis(self._basis,self._n,Np=Np,count=self._Np_list)
			self._basis = self._basis[:self._Ns]
			arg = self._basis.argsort()[::-1]
			self._basis = self._basis[arg].copy()
			self._n = self._n[arg].copy()
			self._Np_list = self._Np_list[arg].copy()
		else:
			self._Ns = self._core.make_basis(self._basis,self._n,Np=Np)
			self._basis = self._basis[:self._Ns]
			arg = self._basis.argsort()[::-1]
			self._basis = self._basis[arg].copy()
			self._n = self._n[arg].copy()

		self._N = N
		self._index_type = _np.min_scalar_type(-self._Ns)
		self._allowed_ops=set(["I","x","y","z","+","-"])

		self._check_symm = None

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







