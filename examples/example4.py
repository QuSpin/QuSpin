from __future__ import print_function, division

import sys,os
import argparse

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import evolve


import numpy as np # generic math functions
import scipy.sparse as sp

import matplotlib.pyplot as plt


##### define model parameters #####
L=300 # system size
# define middle of the chain
if L%2==0:
	i_CM = L//2-0.5 # centre of chain
else:
	i_CM = L//2


q_vec=2*np.pi*np.fft.fftfreq(L)

J=1.0 # hopping
U=1.0 # Bose-Hubbard interaction strength

mu_i=0.02 # initial chemical potential
mu_f=0.0002 # final chemical potential
t_f=30.0/J # set total ramp time

print('ramp speed:', (mu_f - mu_i)/t_f )

# define ramp protocol
def ramp(t,mu_i,mu_f,t_f):
	return  (mu_f - mu_i)*t/t_f
# ramp protocol params
ramp_args=[mu_i,mu_f,t_f]
#define site-coupling lists
hopping=[[-J,i,(i+1)%L] for i in range(L)]
trap_static=[[mu_i*(i-i_CM)**2,i] for i in range(L)]
trap_dynamic=[[(i-i_CM)**2,i] for i in range(L)]
# define basis
basis = boson_basis_1d(L,Nb=1,sps=2)
# define static and dynamic lists
static=[["+-",hopping],["-+",hopping],["n",trap_static]]
dynamic=[['n',trap_dynamic, ramp, ramp_args]]

#### calculate Hamiltonian
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
E,V=H.eigh(time=0.0)


##### imaginary-time evolution

def GPE_imag_time(time,V,H,U):
	return -( H.static.dot(V) + U*np.abs(V)**2*V )


y_trial=V[:,0]*np.sqrt(L)
t_imag=np.linspace(0.0,20.0,21)

GPE_params = (H,U) #
y0_t = evolve(y_trial,0.0,t_imag,GPE_imag_time,real=True,iterate=True,imag_time=True,f_params=GPE_params)

for y0 in y0_t:
	E_new=(H.matrix_ele(y0,y0) + 0.5*U*np.sum(np.abs(y0)**4) ).real

	E_old=E_new
print('finished calculating GS w/ conv error', E_old-E_new)


print("trial state energy:",     (H.matrix_ele(y_trial,y_trial) + np.sum(0.5*U*np.abs(y_trial)**4) ) )
print("GPE GS energy:",    (H.matrix_ele(y0,y0) + np.sum(0.5*U*np.abs(y0)**4) ) )



"""
plt.scatter(np.arange(L)-i_CM, abs(y_trial)**2, color='green' )
plt.scatter(np.arange(L)-i_CM, abs(y0)**2, color='red' )
plt.show()

plt.scatter(q_vec, abs(np.fft.fft(y_trial)/L)**2, color='green' )
plt.scatter(q_vec, abs(np.fft.fft(y0)/L)**2, color='red' )
plt.show()
#exit()
"""


##### real-time evolution
t=np.linspace(0.0,t_f,21)


def GPE_cpx(time,psi,H,U):
	"""
	This function defines the Gross-Pitaevskii equation, cast into real-valued form so it can be solved
	with a real-valued ODE solver.

	The goal is to solve: 

	-i\dot\phi(t) = H(t)\phi(t) + U |\phi(t)|^2 \phi(t)

	for the complex-valued $\phi(t)$ by casting it as a real-valued vector $\psi=[u,v]$ where
	$\phi(t) = u(t) + iv(t)$. The realand imaginary parts, $u(t)$ and $v(t)$, have the same dimension 
	as $\phi(t)$.

	In the most general form, the single-particle Hamiltonian can be decompsoed as $H(t)= H_{stat} + f(t)H_{dyn}$,
	with a complex-valued driving function $f(t)$. Then, the GPE can be cast in the following real-valued form:

	\dot u(t) = +\left[H_{stat} + U(|u(t)|^2 + |v(t)|^2) \right]v(t) + Re[f(t)]H_{dyn}v(t) + Im[f(t)]H_{dyn}u(t)
	\dot v(t) = -\left[H_{stat} + U(|u(t)|^2 + |v(t)|^2) \right]u(t) - Re[f(t)]H_{dyn}u(t) + Im[f(t)]H_{dyn}v(t)

	"""
	# preallocate psi_dot
	psi_dot = np.zeros_like(psi)
	# read off number of lattice sites (number of complex elements in psi)
	Ns=H.Ns
	# static single-particle
	psi_dot[:Ns] =  H.static.dot(psi[Ns:]).real
	psi_dot[Ns:] = -H.static.dot(psi[:Ns]).real
	# static GPE interaction
	psi_dot_2 = np.abs(psi[:Ns])**2 + np.abs(psi[Ns:])**2
	psi_dot[:Ns] += U*psi_dot_2*V[Ns:]
	psi_dot[Ns:] -= U*psi_dot_2*V[:Ns]
	# dynamic single-particle term
	for Hdyn,f,f_args in H.dynamic:
		psi_dot[:Ns] +=  ( +(f(time,*f_args).real)*Hdyn.dot(psi[Ns:]) + (f(time,*f_args).imag)*Hdyn.dot(psi[:Ns]) ).real
		psy_dot[Ns:] +=  ( -(f(time,*f_args).real)*Hdyn.dot(psi[:Ns]) + (f(time,*f_args).imag)*Hdyn.dot(psi[Ns:]) ).real

	return psi_dot


def GPE(time,phi):
	"""
	This function solves the complex-valued time-dependent Gross-Pitaevskii equation:

	-i\dot\phi(t) = H(t)\phi(t) + U |\phi(t)|^2 \phi(t)
	
	"""
	# solve static part of GPE
	phi_dot = -1j*( H.static.dot(phi) + U*np.abs(phi)**2*phi )
	# solve dynamic part of GPE
	for Hd,f,f_args in H.dynamic:
		phi_dot += -1j*f(time,*f_args)*Hd.dot(phi)
	return phi_dot


y0=V[:,0]*np.sqrt(L)
GPE_params = (H,U) #
y_t = evolve(y0,t[0],t,GPE_cpx,stack_state=True,iterate=True,f_params=GPE_params)
#y_t = evolve(y0,t[0],t,GPE,iterate=True)


print('starting real-time evolution...')
E=[]
for i,y in enumerate(y_t):
	E.append( (H.matrix_ele(y,y) + 0.5*U*np.sum(np.abs(y)**4) ).real )
	print("(t,mu(t))=:", (t[i],ramp(t[i],mu_i,mu_f,t_f) + mu_i) )

	plt.plot(np.arange(L)-i_CM, abs(y)**2, color='blue',marker='o' )
	plt.plot(np.arange(L)-i_CM, (abs(y)**2)[::-1], color='green',marker='s' )
	plt.plot(np.arange(L)-i_CM, (ramp(t[i],mu_i,mu_f,t_f) + mu_i)*(np.arange(L)-i_CM)**2,'--',color='red')
	#plt.scatter(q_vec, abs(np.fft.fft(y))**2/L**2, color='blue',marker='o' )
	plt.ylim([-0.01,max(abs(y0)**2)+0.01])
	
	plt.title('$Jt=%0.2f$'%(t[i]))
	
	plt.draw()
	plt.pause(0.005)
	plt.clf()
plt.close()


#plt.plot(t,(E-E[0])/L)
#plt.show()

