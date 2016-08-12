from ..hamiltonian import hamiltonian

# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

from scipy.integrate import complex_ode
from joblib import delayed,Parallel
from numpy import vstack 

import warnings

warnings.warn("Floquet Package has not been fully tested yet, please report bugs to: https://github.com/weinbe58/qspin/issues.",UserWarning)


# this function evolves the ith local basis state with Hamiltonian H
# this is used to construct the stroboscpoic evolution operator
def evolve(i,H,T,atol=1E-9,rtol=1E-9):
	
	nsteps=sum([2**_i for _i in xrange(16,31)]) # huge number to make sure solver is successful.
	psi0=_np.zeros((H.Ns,),dtype=_np.complex128) 
	psi0[i]=1.0

	solver=complex_ode(H._hamiltonian__SO)
	solver.set_integrator('dop853', atol=atol,rtol=rtol,nsteps=nsteps) 
	solver.set_initial_value(psi0,t=0.0)
	solver.integrate(T)

	if solver.successful():
		return solver.y
	else:
		raise Exception('failed to integrate')

### USING JOBLIB ###
def get_U(H,n_jobs,T,atol=1E-9,rtol=1E-9): 
	
	sols=Parallel(n_jobs=n_jobs)(delayed(evolve)(i,H,T,atol,rtol) for i in xrange(H.Ns))

	return vstack(sols)

class Floquet(object):
	def __init__(self,H,T,HF=False,UF=False,thetaF=False,VF=False,n_jobs=1,atol=1E-12,rtol=1E-12):


		if not isinstance(H,hamiltonian):
			raise TypeError("Variable 'H' must be an instance of 'hamiltonian' class!")

		### check if H is periodic with period T
		# define arbitrarily complicated weird-ass number
		t = _np.cos( (_np.pi/_np.exp(0))**( 1.0/_np.euler_gamma ) )

		for _, f, f_args in H.dynamic:
			if abs(f(t,*f_args) - f(t+T,*f_args) ) > 1E3*_np.finfo(_np.complex128).eps:
				raise TypeError("Hamiltonian 'H' must be periodic with period 'T'!")

		if not (type(n_jobs) is int):
			raise TypeError("Expecting integer value for optional variable 'n_jobs'!")


		self._atol = atol
		self._rtol = rtol

		self.T = T

		variables = []
		if HF: variables.append('HF')
		if UF: variables.append('UF')
		if VF: variables.append('VF')
		if thetaF: variables.append('thetaF')


		# calculate evolution operator
		UF = get_U(H,n_jobs,self.T,atol=self._atol,rtol=self._rtol)
		# find Floquet states and phases
		thetaF, VF = _la.eig(UF)


		# calculate and order q'energies
		EF = _np.real( 1j/self.T*_np.log(thetaF) )
		ind_EF = _np.argsort(EF)
		VF = VF[:,ind_EF]
		self.EF = EF[ind_EF]
		# clear up junk
		del ind_EF

		if 'HF' in variables:
			self.HF = 1j/self.T*_np.logm(UF)
		if 'UF' in variables:
			self.UF = UF
		if 'thetaF' in variables:
			self.thetaF = thetaF
		if 'VF' in variables:
			self.VF = VF




