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




def get_basis_type(N, Np, sps, **blocks):
    # calculates the datatype which will fit the largest representative state in basis
    if Np is None:
        # if no particle conservation the largest representative is sps**N
        dtype = _np.min_scalar_type(int(sps**N-1))
        return _np.result_type(dtype,_np.uint32)
    else:
        # if particles are conservated the largest representative is placing all particles as far left
        # as possible. 
        l=Np//(sps-1)
        s_max = sum((sps-1)*sps**(N-1-i)  for i in range(l))
        s_max += (Np%(sps-1))*sps**(N-l-1)
        dtype = _np.min_scalar_type(int(s_max))
        return _np.result_type(dtype,_np.uint32)


# general basis for hardcore bosons/spin-1/2
class boson_basis_general(hcb_basis_general,basis_general):
	def __init__(self,N,Nb=None,nb=None,sps=None,_Np=None,**kwargs):
		if sps is None:

			if Nb is not None:
				if nb is not None:
					raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")

			elif nb is not None:
				if Nb is not None:
					raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")

				Nb = int(nb*N)
			else:
				raise ValueError("expecting value for 'Nb','nb' or 'sps'")

			self._sps = Nb+1
		else:
			if Nb is not None:
				if nb is not None:
					raise ValueError("cannot use 'nb' and 'Nb' simultaineously.")

			elif nb is not None:
				Nb = int(nb*N)

			self._sps = sps

		if self._sps == 2:
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

			if Nb is None:
				Ns = sps**N
				basis_type = get_basis_type(N,Nb,sps)
			elif type(Nb) is int:
				self._check_pcon = True
				Ns = H_dim(Nb,N,self._sps-1)
				basis_type = get_basis_type(N,Nb,self._sps)
			else:
				try:
					Np_iter = iter(Nb)
				except TypeError:
					raise TypeError("Nb must be integer or iteratable object.")
				Ns = 0
				for b in Nb:
					Ns += H_dim(b,N,self._sps-1)

				basis_type = get_basis_type(N,max(Nb),self._sps)

			if len(self._pers)>0:
				Ns = max(int(float(Ns)/_np.multiply.reduce(self._pers))*2,1000)

			if basis_type==_np.uint32:
				basis = _np.zeros(Ns,dtype=_np.uint32)
				n     = _np.zeros(Ns,dtype=_np.uint16)
				self._core = boson_basis_core_wrap_32(N,self._sps,self._maps,self._pers,self._qs)
			elif basis_type==_np.uint64:
				basis = _np.zeros(Ns,dtype=_np.uint64)
				n     = _np.zeros(Ns,dtype=_np.uint16)
				self._core = boson_basis_core_wrap_64(N,self._sps,self._maps,self._pers,self._qs)
			else:
				raise ValueError("states can't be represented as 64-bit unsigned integer")

			if count_particles and (Nb_list is not None):
				Np_list = _np.zeros_like(basis,dtype=_np.uint8)
				self._Ns = self._core.make_basis(basis,n,Np=Nb,count=Np_list)
				if self._Ns < 0:
					raise ValueError("symmetries failed to produce proper reduction in H-space size, please check that mappings do not overlap.")

				basis,ind = _np.unique(basis,return_index=True)
				if self.Ns != basis.shape[0]:
					basis = basis[1:]
					ind = ind[1:]

				self._basis = basis[::-1].copy()
				self._n = n[ind[::-1]].copy()
				self._Np_list = Np_list[ind[::-1]].copy()
			else:
				self._Ns = self._core.make_basis(basis,n,Np=Nb)
				if self._Ns < 0:
					raise ValueError("symmetries failed to produce proper reduction in H-space size, please check that mappings do not overlap.")

				basis,ind = _np.unique(basis,return_index=True)
				if self.Ns != basis.shape[0]:
					basis = basis[1:]
					ind = ind[1:]
				self._basis = basis[::-1].copy()
				self._n = n[ind[::-1]].copy()

			self._N = N
			self._index_type = _np.min_scalar_type(-self._Ns)

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




