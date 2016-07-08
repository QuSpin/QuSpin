from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis
import numpy as np
import scipy as sp
import scipy.sparse as sm
from numpy.linalg import norm
from numpy.random import random,seed

seed()

n_jobs=2

L = 14
Ntot = L


Omega = 10.0
T = 2*np.pi/Omega
A = 0.5

J = 1.0
hz = 0.9045
hx = 0.809

def f(t,Omega):
	return np.exp(1j*Omega*t)
def f_conj(t,Omega):
	return np.exp(-1j*Omega*t)
drive_args = [Omega]


x_field=[[hx,i] for i in xrange(L)]
z_field=[[hz,i] for i in xrange(L)]
J_nn=[[J,i,(i+1)%L] for i in xrange(L)]

absorb=[[A,i,0] for i in xrange(L)]
emit=[[np.conj(A),i,0] for i in xrange(L)]

ph_energy = [[Omega,0]]


static_sp_ph = [["zz|",J_nn], ["z|",z_field], ["x|",x_field], ["+|-",absorb], ["-|+",emit], ["|n",ph_energy]]
dynamic_sp_ph = []

static_sp = [["zz",J_nn], ["z",z_field], ["x",x_field]]
dynamic_sp = [["+",absorb,f,drive_args], ["-",emit,f_conj,drive_args]]

basis_sp_ph = photon_basis(spin_basis_1d,L=L,Ntot=Ntot,kblock=0,pblock=1)
basis_sp = spin_basis_1d(L=L,kblock=0,pblock=1)

H_sp_ph=hamiltonian(static_sp_ph,dynamic_sp_ph,L=L,dtype=np.float64,basis=basis_sp_ph)
H_sp=hamiltonian(static_sp,dynamic_sp_ph,L=L,dtype=np.float64,basis=basis_sp)


Ns = H_sp.Ns

print "spin-photon H-space size is {}".format(H_sp_ph.Ns)
print "spin H-space size is {}".format(Ns)

E_sp_ph = H_sp_ph.eigvalsh()
E_sp = H_sp.eigvalsh()

# this function evolves the ith local basis state with Hamiltonian H
# this is used to construct the stroboscpoic evolution operator
def evolve(i,H,T):
	from numpy import zeros
	from scipy.integrate import complex_ode

	nsteps=sum([2**_i for _i in xrange(32,63)]) # huge number to make sure solver is successful.
	psi0=zeros((Ns,),dtype=np.complex128); psi0[i]=1.0
	solver=complex_ode(H._hamiltonian__SO)
	solver.set_integrator('dop853', atol=10**-15,rtol=10**-15,nsteps=nsteps) 
	solver.set_initial_value(psi0,t=0.0)
	solver.integrate(T)
	if solver.successful():
		return solver.y
	else:
		raise Exception('failed to integrate')


### USING JOBLIB ###
def get_U(H,n_jobs,T): 
	from joblib import delayed,Parallel
	from numpy import vstack # or hstack, I can't remember

	sols=Parallel(n_jobs=n_jobs)(delayed(evolve)(i,H,T) for i in xrange(Ns))

	return vstack(sols)


# calculate Floquet Hamiltonian
U_F = get_U(H_sp,n_jobs,T)
#print evolve(0,H_sp,T)



print E_sp 
print E_sp_ph

			