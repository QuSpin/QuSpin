# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

from scipy.integrate import complex_ode
from joblib import delayed,Parallel
from numpy import vstack 

import warnings


# this function evolves the ith local basis state with Hamiltonian H
# this is used to construct the stroboscpoic evolution operator
def evolve(i,H,T):
	
	nsteps=sum([2**_i for _i in xrange(32,63)]) # huge number to make sure solver is successful.
	psi0=_np.zeros((H.Ns,),dtype=_np.complex128) 
	psi0[i]=1.0

	solver=complex_ode(H._hamiltonian__SO)
	solver.set_integrator('dop853', atol=1E-12,rtol=1E-12,nsteps=nsteps) 
	solver.set_initial_value(psi0,t=0.0)
	solver.integrate(T)

	if solver.successful():
		return solver.y
	else:
		raise Exception('failed to integrate')

### USING JOBLIB ###
def get_U(H,n_jobs,T): 
	
	sols=Parallel(n_jobs=n_jobs)(delayed(evolve)(i,H,T) for i in xrange(H.Ns))

	return vstack(sols)

