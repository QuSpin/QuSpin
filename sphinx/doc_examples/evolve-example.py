from __future__ import print_function, division
#
import sys,os
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import evolve # ODE evolve tool
from quspin.tools.Floquet import Floquet_t_vec # stroboscopic time vector
import numpy as np # generic math functions
from six import iteritems
#
L=50 # number of lattice sites
i_CM = L//2-0.5 # centre of chain
#
### static model parameters
J=1.0 # hopping
mu_trap=0.002 # harmonic trap strength
U=1.0 # mean-field (GPE) interaction
#
### periodic driving
A=1.0 # drive amplitude
Omega=10.0 # drive frequency
def drive(t,Omega):
	return np.exp(-1j*A*np.sin(Omega*t) )
def drive_conj(t,Omega):
	return np.exp(+1j*A*np.sin(Omega*t) )
drive_args=[Omega] # drive arguments
t=Floquet_t_vec(Omega,30,len_T=1) # time vector, 30 stroboscopic periods
#
### site-couping lists
hopping=[[-J,i,(i+1)%L] for i in range(L)]
trap=[[mu_trap*(i-i_CM)**2,i] for i in range(L)]
#
### operator strings for single-particle Hamiltonian
static=[['n',trap]]
dynamic=[["+-",hopping,drive,drive_args],["-+",hopping,drive_conj,drive_args]]
# define single-particle basis
basis = boson_basis_1d(L,Nb=1,sps=2) # Nb=1 boson and sps=2 states per site [empty and filled]
#
### build Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.complex128)
# calculate eigenvalues and eigenvectors of free particle
E,V=H.eigh(time=0)
#######
def GPE(time,phi):
	'''This function solves the complex-valued time-dependent Gross-Pitaevskii equation:
	#
	$-i\dot\phi(t) = H(t)\phi(t) + U |\phi(t)|^2 \phi(t)$
	#

	'''
	# solve static part of GPE
	phi_dot = -1j*( H.static.dot(phi) + U*np.abs(phi)**2*phi )
	# solve dynamic part of GPE
	for fun,Hdyn in iteritems(H.dynamic):
		phi_dot += -1j*fun(time)*Hdyn.dot(phi)
	return phi_dot
	#
# initial state
phi0=V[:,0]*np.sqrt(L)
# solve cpx-valued GPE
phi_t = evolve(phi0,t.i,t.vals,GPE)
########
def GPE_real(time,psi,H,U):
	'''This function defines the Gross-Pitaevskii equation, cast into real-valued form so it can be solved with a 
	real-valued ODE solver.
	#
	The goal is to solve: 
	#
	$-i\dot\phi(t) = H(t)\phi(t) + U |\phi(t)|^2 \phi(t)$
	#
	for the complex-valued $\phi(t)$ by casting it as a real-valued vector $\psi=[u,v]$ where 
	$\phi(t) = u(t) + iv(t)$. The realand imaginary parts, $u(t)$ and $v(t)$, have the same dimension as 
	$\phi(t)$.
	#
	In the most general form, the single-particle Hamiltonian can be decomposed as 
	$H(t)= H_{stat} + f(t)H_{dyn}$, with a complex-valued driving function $f(t)$. Then, the GPE can be cast in 
	the following real-valued form:
	#
	$\dot u(t) = +\left[H_{stat} + U(|u(t)|^2 + |v(t)|^2) \right]v(t) + Re[f(t)]H_{dyn}v(t) + Im[f(t)]H_{dyn}u(t)$
	$\dot v(t) = -\left[H_{stat} + U(|u(t)|^2 + |v(t)|^2) \right]u(t) - Re[f(t)]H_{dyn}u(t) + Im[f(t)]H_{dyn}v(t)$
	#

	'''
	# preallocate psi_dot
	psi_dot = np.zeros_like(psi)
	# read off number of lattice sites (number of complex elements in psi)
	Ns=H.Ns
	# static single-particle
	psi_dot[:Ns] =  H.static.dot(psi[Ns:]).real
	psi_dot[Ns:] = -H.static.dot(psi[:Ns]).real
	# static GPE interaction
	psi_dot_2 = np.abs(psi[:Ns])**2 + np.abs(psi[Ns:])**2
	psi_dot[:Ns] += U*psi_dot_2*psi[Ns:]
	psi_dot[Ns:] -= U*psi_dot_2*psi[:Ns]
	# dynamic single-particle term
	for func, Hdyn in iteritems(H.dynamic):
		fun=func(time) # evaluate drive 
		psi_dot[:Ns] +=  ( +(fun.real)*Hdyn.dot(psi[Ns:]) + (fun.imag)*Hdyn.dot(psi[:Ns])  ).real
		psi_dot[Ns:] +=  ( -(fun.real)*Hdyn.dot(psi[:Ns]) + (fun.imag)*Hdyn.dot(psi[Ns:])  ).real
	#
	return psi_dot
#
# define initial condition
phi0=V[:,0]*np.sqrt(L)
# define ODE solver parameters
GPE_params = (H,U)
# solve real-valued GPE
phi_t = evolve(phi0,t.i,t.vals,GPE_real,stack_state=True,f_params=GPE_params)