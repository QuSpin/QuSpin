from ..hamiltonian import hamiltonian,ishamiltonian

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import scipy.sparse.linalg as _sla
import numpy as _np

from scipy.integrate import complex_ode
from joblib import delayed,Parallel
from numpy import vstack 

import warnings

warnings.warn("Floquet Package has not been fully tested yet, please report bugs to: https://github.com/weinbe58/qspin/issues.",UserWarning,stacklevel=3)


# this function evolves the ith local basis state with Hamiltonian H
# this is used to construct the stroboscpoic evolution operator
def evolve_cont(i,H,T,atol=1E-9,rtol=1E-9):
	
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




def evolve_step_1(i,H_list,dt_list):
	
	psi0=_np.zeros((H.Ns,),dtype=_np.complex128) 
	psi0[i]=1.0

	for dt,H in zip(dt_list,H_list):
		psi0 = _sla.expm_multiply(-1j*dt*H,psi0)

	return psi0



def evolve_step_2(i,H,t_list,dt_list):
	
	psi0=_np.zeros((H.Ns,),dtype=_np.complex128) 
	psi0[i]=1.0

	for t,dt in zip(t_list,dt_list):
		psi0 = _sla.expm_multiply(-1j*dt*H.tocsr(t),psi0)

	return psi0
	
	

### USING JOBLIB ###
def get_U_cont(H,T,n_jobs,atol=1E-9,rtol=1E-9): 
	
	sols=Parallel(n_jobs=n_jobs)(delayed(evolve_cont)(i,H,T,atol,rtol) for i in xrange(H.Ns))

	return vstack(sols)


def get_U_step_1(H_list,dt_list,n_jobs): 
	
	sols=Parallel(n_jobs=n_jobs)(delayed(evolve_step_1)(i,H_list,dt_list) for i in xrange(H.Ns))

	return vstack(sols)

def get_U_step_2(H,t_list,dt_list,n_jobs): 
	
	sols=Parallel(n_jobs=n_jobs)(delayed(evolve_step_2)(i,H,t_list,dt_list) for i in xrange(H.Ns))

	return vstack(sols)


class Floquet(object):
	def __init__(self,evo_dict,HF=False,UF=False,thetaF=False,VF=False,n_jobs=1):

		variables = []
		if HF: variables.append('HF')
		if UF: variables.append('UF')
		if VF: variables.append('VF')
		if thetaF: variables.append('thetaF')

		
		if isinstance(evo_dict,dict):

			keys = evo_dict.keys()
			if set(keys) == set(["H","T"]):

				H = evo_dict["H"]
				T = evo_dict["T"]
				self._atol = evo_dict.get("atol")
				self._rtol = evo_dict.get("rtol")

				if self._atol is None:
					self._atol=1E-12
				elif type(self.atol) is not float:
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
				UF = get_U_cont(H,self.T,n_jobs,atol=self._atol,rtol=self._rtol)

			elif set(keys) == set(["H","t_list","dt_list"]):
				H = evo_dict["H"]
				t_list = _np.asarray(evo_dict["t_list"],dtype=_np.float64)
				dt_list = _np.asarray(evo_dict["dt_list"],dtype=_np.float64)
				print t_list
				print dt_list

				if t_list.ndim != 1:
					raise ValueError("t_list must be 1d array.")

				if dt_list.ndim != 1:
					raise ValueError("dt_list must be 1d array.")

				self._T = dt_list.sum()

				if not ishamiltonian(H):
					raise ValueError("expecting hamiltonian object for 'H'.")

				# calculate evolution operator
				UF = get_U_step_2(H,t_list,dt_list,n_jobs)



			elif set(keys) == set(["H_list","dt_list"]):
				H_list = evo_dict["H_list"]
				dt_list = _np.asarray(evo_dict["dt_list"],dtype=_np.float64)

				if t_list.ndim != 1:
					raise ValueError("t_list must be 1d array.")

				if dt_list.ndim != 1:
					raise ValueError("dt_list must be 1d array.")

				self._T = dt_list.sum()
				
				if type(H_list) not in (list,tuple):
					raise ValueError("expecting list/tuple for H_list.")

				# calculate evolution operator
				UF = get_U_cont(H_list,dt_list,n_jobs)
				
			else:
				raise ValueError("evo_dict={0} is not correct format.".format(evo_dict))	
		else:
			raise ValueError("evo_dict={0} is not correct format.".format(evo_dict))


		# find Floquet states and phases
		thetaF, VF = _la.eig(UF)


		# calculate and order q'energies
		EF = _np.real( 1j/self.T*_np.log(thetaF) )
		ind_EF = _np.argsort(EF)
		VF = _np.array(VF[:,ind_EF])
		self.EF = EF[ind_EF]
		# clear up junk
		del ind_EF






		if 'HF' in variables:
			self._HF = 1j/self.T*_np.logm(UF)
#			self._Hf = np.einsum("ij,j,jk->ik",VF.T.conj(),EF,VF)
		if 'UF' in variables:
			self._UF = UF
		if 'thetaF' in variables:
			self._thetaF = thetaF
		if 'VF' in variables:
			self._VF = VF



	def __getattr__(self, attr):
		if hasattr(self,"_"+attr):
			return eval("self."+"_"+attr)
		else:
			raise AttributeError
			






