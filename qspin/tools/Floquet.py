from __future__ import print_function, division

from ..operators import hamiltonian,ishamiltonian


# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp

import numpy as _np

from scipy.integrate import complex_ode
from joblib import delayed,Parallel
from numpy import vstack 

import warnings

__all__ = ['Floquet_t_vec','Floquet']

#warnings.warn("Floquet Package has not been fully tested yet, please report bugs to: https://github.com/weinbe58/qspin/issues.",UserWarning,stacklevel=3)


# xrange is replaced with range in python 3.
# if python 2 is being used, range will cause memory overflow.
# This function is a work around to get the functionality of 
# xrange for both python 2 and 3 simultaineously. 
def range_iter(start,stop,step):
	from itertools import count
	counter = count(start,step)
	while True:
		i = counter.next()
		if i < stop:
			yield i
		else:
			break


def _evolve_cont(i,H,T,atol=1E-9,rtol=1E-9):
	"""
	This function evolves the ith local basis state under the Hamiltonian H up to period T. 
	This is used to construct the stroboscpoic evolution operator
	"""
	
	nsteps=_np.iinfo(_np.int32).max # huge number to make sure solver is successful.
	psi0=_np.zeros((H.Ns,),dtype=_np.complex128) 
	psi0[i]=1.0

	solver=complex_ode(H._hamiltonian__SO)
	solver.set_integrator('dop853', atol=atol,rtol=rtol,nsteps=nsteps) 
	solver.set_initial_value(psi0,t=0.0)
	t_list = [0,T]
	nsteps = 1
	while True:
		for t in t_list[1:]:
			solver.integrate(t)
			if solver.successful():
				if t == T:
					return solver.y
				continue
			else:
				break

		nsteps *= 10
		t_list = _np.linspace(0,T,num=nsteps+1,endpoint=True)




def _evolve_step_1(i,H_list,dt_list):
	"""
	This function calculates the evolved state 
	"""
	
	psi0=_np.zeros((H_list[0].Ns,),dtype=_np.complex128) 
	psi0[i]=1.0

	for dt,H in zip(dt_list,H_list):
		psi0 = _sla.expm_multiply(-1j*dt*H.tocsr(),psi0)

	return psi0



def _evolve_step_2(i,H,t_list,dt_list):
	
	psi0=_np.zeros((H.Ns,),dtype=_np.complex128) 
	psi0[i]=1.0

	for t,dt in zip(t_list,dt_list):
		psi0 = _sla.expm_multiply(-1j*dt*H.tocsr(t),psi0)

	return psi0
	
	

### USING JOBLIB ###
def _get_U_cont(H,T,n_jobs,atol=1E-9,rtol=1E-9): 
	
	sols=Parallel(n_jobs=n_jobs)(delayed(_evolve_cont)(i,H,T,atol,rtol) for i in range_iter(0,H.Ns,1))

	return vstack(sols)


def _get_U_step_1(H_list,dt_list,n_jobs): 
	
	sols=Parallel(n_jobs=n_jobs)(delayed(_evolve_step_1)(i,H_list,dt_list) for i in range_iter(0,H_list[0].Ns,1))

	return vstack(sols)

def _get_U_step_2(H,t_list,dt_list,n_jobs): 
	
	sols=Parallel(n_jobs=n_jobs)(delayed(_evolve_step_2)(i,H,t_list,dt_list) for i in range_iter(0,H.Ns,1))

	return vstack(sols)


class Floquet(object):
	def __init__(self,evo_dict,HF=False,UF=False,thetaF=False,VF=False,n_jobs=1):
		"""
		Calculates the Floquet spectrum for a given protocol, and optionally the Floquet hamiltonian matrix,
		and Floquet eigen-vectors.

		--- arguments ---
		* evo_dict: (compulsory) dictionary which passes the different types of protocols to calculate evolution operator:

			1. Continuous protocol.

				* 'H': (compulsory) hamiltonian object to generate the time evolution. 

				* 'T': (compulsory) period of the protocol. 

				* 'rtol': (optional) relative tolerance for the ode solver. (default = 1E-9)

				* 'atol': (optional) absolute tolerance for the ode solver. (default = 1E-9)

			2. Step protocol from a hamiltonian object. 

				* 'H': (compulsory) hamiltonian object to generate the hamiltonians at each step.
				
				* 't_list': (compulsory) list of times to evaluate the hamiltonian at when doing each step.

				* 'dt_list': (compulsory) list of time steps for each step of the evolution. 

			3. Step protocol from a list of hamiltonians. 

				* 'H_list': (compulsory) list of matrices which to evolve with.

				* 'dt_list': (compulsory) list of time steps to evolve with. 


		* HF: (optional) if set to 'True' calculate Floquet hamiltonian. 

		* UF: (optional) if set to 'True' save evolution operator. 

		* ThetaF: (optional) if set to 'True' save the eigenvalues of the evolution operator. 

		* VF: (optional) if set to 'True' save the eigenvectors of the evolution operator. 

		* n_jobs: (optional) set the number of processors which are used when looping over the basis states. 

		--- Floquet attributes ---: '_. ' below stands for 'object. '

		Always given:

		_.EF: ordered Floquet qausi-energies in interval [-Omega,Omega]

		Calculate via flags:

		_.HF: Floquet Hamiltonian dense array

		_.UF: Evolution operator

		_.VF: Floquet eigenstates

		_.thetaF: eigenvalues of evolution operator


		"""

		variables = []
		if HF: variables.append('HF')
		if UF: variables.append('UF')
		if VF: variables.append('VF')
		if thetaF: variables.append('thetaF')

		
		if isinstance(evo_dict,dict):

			keys = evo_dict.keys()
			if set(keys) == set(["H","T"]) or set(keys) == set(["H","T","arol"]) or set(keys) == set(["H","T","atol","rtol"]):

				H = evo_dict["H"]
				T = evo_dict["T"]
				self._atol = evo_dict.get("atol")
				self._rtol = evo_dict.get("rtol")

				if self._atol is None:
					self._atol=1E-12
				elif type(self._atol) is not float:
					raise ValueError("expecting float for 'atol'.")

				if self._rtol is None:
					self._rtol=1E-12
				elif type(self._rtol) is not float:
					raise ValueError("expecting float for 'rtol'.")
				

				if not ishamiltonian(H):
					raise ValueError("expecting hamiltonian object for 'H'.")

				if not _np.isscalar(T):
					raise ValueError("expecting scalar object for 'T'.")

				if _np.iscomplex(T):
					raise ValueError("expecting real value for 'T'.")


				### check if H is periodic with period T
				# define arbitrarily complicated weird-ass number

				t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )

				for _, f, f_args in H.dynamic:
					if abs(f(t,*f_args) - f(t+T,*f_args) ) > 1E3*_np.finfo(_np.complex128).eps:
						raise TypeError("Hamiltonian 'H' must be periodic with period 'T'!")

				if not (type(n_jobs) is int):
					raise TypeError("expecting integer value for optional variable 'n_jobs'!")



				self._T = T

				# calculate evolution operator
				UF = _get_U_cont(H,self.T,n_jobs,atol=self._atol,rtol=self._rtol)

			elif set(keys) == set(["H","t_list","dt_list"]):
				H = evo_dict["H"]
				t_list = _np.asarray(evo_dict["t_list"],dtype=_np.float64)
				dt_list = _np.asarray(evo_dict["dt_list"],dtype=_np.float64)

				if t_list.ndim != 1:
					raise ValueError("t_list must be 1d array.")

				if dt_list.ndim != 1:
					raise ValueError("dt_list must be 1d array.")

				self._T = dt_list.sum()

				if not ishamiltonian(H):
					raise ValueError("expecting hamiltonian object for 'H'.")

				# calculate evolution operator
				UF = _get_U_step_2(H,t_list,dt_list,n_jobs)



			elif set(keys) == set(["H_list","dt_list"]):
				H_list = evo_dict["H_list"]
				dt_list = _np.asarray(evo_dict["dt_list"],dtype=_np.float64)


				if dt_list.ndim != 1:
					raise ValueError("dt_list must be 1d array.")

				self._T = dt_list.sum()
				
				if type(H_list) not in (list,tuple):
					raise ValueError("expecting list/tuple for H_list.")

				if len(dt_list) != len(H_list):
					raise ValueError("Expecting arguments 'H_list' and 'dt_list' to have the same length!")


				# calculate evolution operator
				UF = _get_U_step_1(H_list,dt_list,n_jobs)
				
			else:
				raise ValueError("evo_dict={0} is not correct format.".format(evo_dict))	
		else:
			raise ValueError("evo_dict={0} is not correct format.".format(evo_dict))

		if 'UF' in variables:
			self._UF = _np.copy(UF)

		if 'HF' in variables:
			self._HF = 1j/self._T*_la.logm(UF)

		# find Floquet states and phases
		if "VF" in variables:
			thetaF, VF = _la.eig(UF,overwrite_a=True)
			# check and orthogonalise VF in degenerate subspaces
			if _np.any( _np.diff(_np.sort(thetaF)) < 1E3*_np.finfo(thetaF.dtype).eps):
				VF,_ = _la.qr(VF, overwrite_a=True) 
			# calculate and order q'energies
			EF = _np.real( 1j/self.T*_np.log(thetaF) )
			# sort and order
			ind_EF = _np.argsort(EF)
			self._EF = _np.array(EF[ind_EF])
			self._VF = _np.array(VF[:,ind_EF])
			# clear up junk
			del VF
		else:
			thetaF = _la.eigvals(UF,overwrite_a=True)
			# calculate and order q'energies
			EF = _np.real( 1j/self.T*_np.log(thetaF) )
			ind_EF = _np.argsort(EF)
			self._EF = _np.array(EF[ind_EF])

		if 'thetaF' in variables:
			# sort phases
			thetaF = _np.array(thetaF[ind_EF])
			self._thetaF = thetaF


	@property
	def T(self):
		return self._T

	@property
	def EF(self):
		return self._EF

	@property
	def HF(self):
		if hasattr(self,"_HF"):
			return self._HF
		else:
			raise AttributeError("missing atrribute 'HF'.")

	@property
	def UF(self):
		if hasattr(self,"_UF"):
			return self._UF
		else:
			raise AttributeError("missing atrribute 'UF'.")

	@property
	def thetaF(self):
		if hasattr(self,"_thetaF"):
			return self._thetaF
		else:
			raise AttributeError("missing atrribute 'thetaF'.")


	@property
	def VF(self):
		if hasattr(self,"_VF"):
			return self._VF
		else:
			raise AttributeError("missing atrribute 'VF'.")
			



class Floquet_t_vec(object):
	def __init__(self,Omega,N_const,len_T=100,N_up=0,N_down=0):
		"""
		Returns a time vector (np.array) which hits the stroboscopic times, and has as attributes
		their indices. The time vector can be divided in three regimes: ramp-up, constant and ramp-down.

		--- arguments ---

		Omega: (compulsory) drive frequency

		N_const: (compulsory) # of time periods in the constant period

		N_up: (optional) # of time periods in the ramp-up period

		N_up: (optional) # of time periods in the ramp-down period

		len_T: (optional) # of time points within a period. N.B. the last period interval is assumed 
				open on the right, i.e. [0,T) and the point T does not go into the definition of 'len_T'. 


		--- time vector attributes ---: '_. ' below stands for 'object. '


		_.vals: time vector values

		_.i: initial time value

		_.f: final time value

		_.tot: total length of time: t.i - t.f 

		_.T: period of drive

		_.dt: time vector spacing

		_.len: length of total time vector

		_.len_T: # of points in a single period interval, assumed half-open: [0,T)

		_.N: total # of periods


		--- strobo attribues ---


		_.strobo.vals: strobosopic time values

		_.strobo.inds: strobosopic time indices


		--- regime attributes --- (available if N_up or N_down are parsed)


		_.up : referes to time vector of up-regime; inherits the above attributes (e.g. _up.strobo.inds) except _.T, _.dt, and ._lenT

		_.const : referes to time vector of const-regime; inherits the above attributes except _.T, _.dt, and ._lenT

		_.down : referes to time vector of down-regime; inherits the above attributes except _.T, _.dt, and ._lenT

		"""

		# total number of periods
		self._N = N_up+N_const+N_down
		# total length of a period 
		self._len_T = len_T
		# driving period T
		self._T = 2.0*_np.pi/Omega 


		# define time vector
		n = _np.linspace(-N_up, N_const+N_down, self.N*len_T+1)
		self._vals = self.T*n
		# total length of time vector
		self._len = self.vals.size
		# time step
		self._dt = self.T/self.len_T
		# define index of period -N_up
		ind0 = 0 #int( _np.squeeze( (n==-N_up).nonzero() ) )

		# calculate stroboscopic times
		self._strobo = _strobo_times(self.vals,self.len_T,ind0)

		# define initial and final times and total duration
		self._i = self.vals[0]
		self._f = self.vals[-1]
		self._tot = self._i - self._f

		# if ramp is on, define more attributes
		if N_up > 0 and N_down > 0:
			t_up = self.vals[:self.strobo.inds[N_up]]
			self._up = _periodic_ramp(N_up,t_up,self.T,self.len_T,ind0)

			t_const = self.vals[self.strobo.inds[N_up]:self.strobo.inds[N_up+N_const]+1]
			ind0 = self.up.strobo.inds[-1]+self.len_T
			self._const = _periodic_ramp(N_const,t_const,self.T,self.len_T,ind0)

			t_down = self.vals[self.strobo.inds[N_up+N_const]+1:self.strobo.inds[-1]+1]
			ind0 = self.const.strobo.inds[-1]+self.len_T
			self._down = _periodic_ramp(N_down,t_down,self.T,self.len_T,ind0)

		elif N_up > 0:
			t_up = self.vals[:self.strobo.inds[N_up]]
			self._up = _periodic_ramp(N_up,t_up,self.T,self.len_T,ind0)

			t_const = self.vals[self.strobo.inds[N_up]:self.strobo.inds[N_up+N_const]+1]
			ind0 = self.up.strobo.inds[-1]+self.len_T
			self._const = _periodic_ramp(N_const,t_const,self.T,self.len_T,ind0)

		elif N_down > 0:
			t_const = self.vals[self.strobo.inds[N_up]:self.strobo.inds[N_up+N_const]+1]
			self._const = _periodic_ramp(N_const,t_const,self.T,self.len_T,ind0)

			t_down = self.vals[self.strobo.inds[N_up+N_const]+1:self.strobo.inds[-1]+1]
			ind0 = self.const.strobo.inds[-1]+self.len_T
			self._down = _periodic_ramp(N_down,t_down,self.T,self.len_T,ind0)


	def __iter__(self):
		return self.vals.__iter__()

	def __getitem__(self,s):
		return self._vals.__getitem__(s)

	def __len__(self):
		return self._vals.__len__()

	@property
	def N(self):
		return self._N

	@property
	def len_T(self):
		return self._len_T

	@property
	def T(self):
		return self._T

	@property
	def vals(self):
		return self._vals

	@property
	def len(self):
		return self._len

	@property
	def dt(self):
		return self._dt

	@property
	def strobo(self):
		return self._strobo

	@property
	def i(self):
		return self._i

	@property
	def f(self):
		return self._f


	@property
	def tot(self):
		return self._tot



	@property
	def up(self):
		if hasattr(self,"_up"):
			return self._up
		else:
			raise AttributeError("missing attribute 'up'")

	@property
	def const(self):
		if hasattr(self,"_const"):
			return self._up
		else:
			raise AttributeError("missing attribute 'const'")

	@property
	def down(self):
		if hasattr(self,"_down"):
			return self._up
		else:
			raise AttributeError("missing attribute 'down'")


	





class _strobo_times():
	def __init__(self,t,len_T,ind0):
		"""
		Calculates stroboscopic times in time vector t with period length len_T and assigns them as
		attributes.
		"""
		# indices of strobo times
		self._inds = _np.arange(0,t.size,len_T).astype(int)
		#discrete stroboscopic t_vecs
		self._vals = t.take(self._inds)
		# update strobo indices to match shifted (ramped) ones
		self._inds += ind0

	@property
	def inds(self):
		return self._inds

	@property
	def vals(self):
		return self._vals
	
		 


class _periodic_ramp():
	def __init__(self,N,t,T,len_T,ind0):
		"""
		Defines time vector attributes of each regime.
		"""
		self._N=N # total # periods
		self._vals = t # time values
		self._i = self._vals[0] # initial value
		self._f = self._vals[-1] # final value
		self._tot = self._N*T # total duration
		self._len = self._vals.size # total length
		self._strobo = _strobo_times(self._vals,len_T,ind0) # strobo attributes

	def __iter__(self):
		return self.vals.__iter__()

	def __getitem__(self,s):
		return self._vals.__getitem__(s)

	def __len__(self):
		return self._vals.__len__()

	@property
	def N(self):
		return self._N
	
	@property
	def vals(self):
		return self._vals

	@property
	def i(self):
		return self._i

	@property
	def f(self):
		return self._f

	@property
	def tot(self):
		return self._tot

	@property
	def len(self):
		return self._len

	@property
	def strobo(self):
		return self._strobo
	
	
	
	
	
	





