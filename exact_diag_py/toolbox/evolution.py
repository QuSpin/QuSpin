from ..spins import hamiltonian as _hamiltonian
from scipy.sparse import issparse as _issparse

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









def exp(H,V,z,time=0,n=1,atol=10**(-15)):
	"""
	args:
		H, either hamiltonian or sparse matrix
		V, vector to apply the matrix exponential on.
		a, the parameter in the exponential exp(aH)V
		time, time to evaluate drive at.
		n, the number of steps to break the expoential into exp(aH/n)^n V
		error, if the norm the vector of the taylor series is less than this number
		then the taylor series is truncated.

	description:
		this function computes exp(zH)V as a taylor series in zH. not useful for long time evolution.

	"""
	if self.Ns <= 0:
		return _np.asarray([])
	if not _np.isscalar(time):
		raise NotImplementedError
	if n <= 0: raise ValueError('n must be > 0')


	if isinstance(H,_hamiltonian):
		H=H.tocsr(time=time)
	elif issparse(H):
		H=H.tocsr()
	else:
		raise TypeError("H must be either scipy.sparse or hamiltonian objects")


	V=_np.asarray(V)
	for j in xrange(n):
		V1=_np.array(V)
		e=1.0; i=1		
		while e > atol:
			V1=(z/(n*i))*H.dot(V1)
			V+=V1
			if i%2 == 0:
				e=_norm(V1)
			i+=1
	return V


