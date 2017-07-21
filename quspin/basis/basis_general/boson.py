from ._basis_general_core import boson_basis_core_wrap_32,boson_basis_core_wrap_64
from .base_hcb import hcb_basis_general
from .base_general import basis_general
import numpy as _np
from scipy.misc import comb


def H_dim(N,length,m_max):
    """
    Returns the total number of states in the bosonic Hilbert space

    --- arguments:

    N: total number of bosons in lattice
    length: total number of sites
    m_max+1: max number of states per site 
    """
    Ns = 0
    for r in range(N//(m_max+1)+1):
        r_2 = N - r*(m_max+1)
        if r % 2 == 0:
            Ns +=  comb(length,r,exact=True) * comb(length + r_2 - 1,r_2,exact=True)
        else:
            Ns += -comb(length,r,exact=True) * comb(length + r_2 - 1,r_2,exact=True)

    return Ns




def get_basis_type(L, Np, sps, **blocks):
    # calculates the datatype which will fit the largest representative state in basis
    if Np is None:
        # if no particle conservation the largest representative is sps**L
        dtype = _np.min_scalar_type(int(sps**L-1))
        return _np.result_type(dtype,_np.uint32)
    else:
        # if particles are conservated the largest representative is placing all particles as far left
        # as possible. 
        l=Np//(sps-1)
        s_max = sum((sps-1)*sps**(L-1-i)  for i in range(l))
        s_max += (Np%(sps-1))*sps**(L-l-1)
        dtype = _np.min_scalar_type(int(s_max))
        return _np.result_type(dtype,_np.uint32)


# general basis for hardcore bosons/spin-1/2
class boson_basis_general(hcb_basis_general,basis_general):
	def __init__(self,N,Nb=None,sps=None,_Np=None,**kwargs):
		if sps == 2:
			general_hcb_basis.__init__(self,N,Nb=Nb,_Np=_Np,**kwargs)
			self._allowed_ops=set(["I","n","+","-"])
		else:
			basis_general.__init__(self,N,**kwargs)
			self._check_pcon = False
			count_particles = False
			if _Np is not None and Nb is None:
				count_particles = True
				if type(_Np) is not int:
					raise ValueError("_Np must be integer")
				if _Np >= -1:
					if _Np+1 > N: 
						Nb = list(range(N+1))
					elif _Np==-1:
						Nb = None
					else:
						Nb = list(range(_Np+1))
				else:
					raise ValueError("_Np == -1 for no particle conservation, _Np >= 0 for particle conservation")

			if Nb is None and sps is None:
				raise ValueError("must specify number of boons or sps")

			if Nb is not None and sps is None:
				sps = Nb+1

			if Nb is None:
				Ns = sps**N
				basis_type = get_basis_type(N,Nb,sps)
			elif type(Nb) is int:
				self._check_pcon = True
				Ns = H_dim(Nb,N,sps-1)
				basis_type = get_basis_type(N,Nb,sps)
			else:
				try:
					Np_iter = iter(Nb)
				except TypeError:
					raise TypeError("Nb must be integer or iteratable object.")
				Ns = 0
				for Nb in Np_iter:
					Ns += H_dim(Nb,N,sps-1)

				basis_type = get_basis_type(N,max(iter(Nb)),sps)

			if len(self._pers)>0:
				Ns = max(int(float(Ns)/_np.multiply.reduce(self._pers))*2,1000)

			if basis_type==_np.uint32:
				self._basis = _np.zeros(Ns,dtype=_np.uint32)
				self._n     = _np.zeros(Ns,dtype=_np.uint16)
				self._core = boson_basis_core_wrap_32(N,sps,self._maps,self._pers,self._qs)
			elif basis_type==_np.uint64:
				self._basis = _np.zeros(Ns,dtype=_np.uint64)
				self._n     = _np.zeros(Ns,dtype=_np.uint16)
				self._core = boson_basis_core_wrap_64(N,sps,self._maps,self._pers,self._qs)
			else:
				raise ValueError("states can't be represented as 64-bit unsigned integer")

			self._sps=sps
			if count_particles and (Nb is not None):
				self._Np_list = _np.zeros_like(self._basis,dtype=_np.uint8)
				self._Ns = self._core.make_basis(self._basis,self._n,Np=Nb,count=self._Np_list)
				self._basis = self._basis[:self._Ns]
				arg = self._basis.argsort()[::-1]
				self._basis = self._basis[arg].copy()
				self._n = self._n[arg].copy()
				self._Np_list = self._Np_list[arg].copy()
			else:
				self._Ns = self._core.make_basis(self._basis,self._n,Np=Nb)
				self._basis = self._basis[:self._Ns]
				arg = self._basis.argsort()[::-1]
				self._basis = self._basis[arg].copy()
				self._n = self._n[arg].copy()

			self._N = N
			self._index_type = _np.min_scalar_type(-self._Ns)

			self._check_symm = None

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




