from ..spins import hamiltonian as _hamiltonian

from krylov import expm_krylov as _expm_krylov

from scipy.sparse import issparse as _issparse
from scipy.sparse.linalg import expm_multiply as _expm_multiply
from scipy.integrate import complex_ode as _complex_ode
from scipy.integrate import ode as _ode

import numpy as _np


def evolve(H,v0,t0,time,real_time=True,verbose=False,**integrator_params):
	"""
	args:
		H, hamiltonian to evolve with
		v0, intial wavefunction to evolve.
		t0, intial time 
		time, iterable or scalar, or time to evolve v0 to
		real_time, evolve real or imaginary time
		verbose, print times out as you evolve
		**integrator_params, the parameters used for the dop853 explicit rung-kutta solver.
		see documentation http://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.integrate.ode.html

	description:
		This function uses complex_ode to evolve an input wavefunction.
	"""
	if isinstance(H,_hamiltonian):
		H=H.tocsr(time=time)
	else:
		raise TypeError("H must be hamiltonian object")

	if H.Ns <= 0:
		return _np.asarray([])

	v0=_np.asarray(v0)

	if real_time:
		solver=_complex_ode(H._hamiltonian__SO)
	else:
		if H.dtype in [_np.float32,_np.float64]:
			solver=_ode(H._hamiltonian__ISO)
		else:
			solver=_complex_ode(H._hamiltonian__ISO)

	solver.set_integrator("dop853",**integrator_params)
	solver.set_initial_value(v0,t=t0)
		
	if _np.isscalar(time):
		if time==t0: return v0
		solver.integrate(time)
		if solver.successful():
			return solver.y
		else:
			raise RuntimeError('failed to integrate')		
	else:
		sol=[]
		for t in time:
			if verbose: print t
			if t==t0: 
				sol.append(v0)
				continue
			solver.integrate(t)
			if solver.successful():
				sol.append(solver.y)
			else:
				raise RuntimeError('failed to integrate')
		return sol









def step_drive(H_list,t_list,v0,Nsteps=1,krylov=False,tol=10**(-15),hermitian=True):
	if krylov:
		for i in xrange(Nsteps):
			for t,H in zip(H_list,t_list):
				v0 = _expm_krylov(H,v0,z=-1j*t,tol=10**(-15),hermitian=True)
	else:
		for i in xrange(Nsteps):
			for t,H in zip(H_list,t_list):
				v0 = _expm_multiply(-1j*t*H,v0)
		

	return v0





















