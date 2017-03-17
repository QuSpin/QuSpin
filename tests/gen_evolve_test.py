from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import evolve
from quspin.tools.Floquet import Floquet_t_vec
import numpy as np # generic math functions

import matplotlib.pyplot as plt

##### define model parameters #####
L=201 # system size

J=1.0
mu=0.00
g=10.0

A=1.0
Omega=2.0

hopping=[[-J,i,i+1] for i in range(L-1)]
trap=[[mu*(i-L/2.0)**2,i] for i in range(L)]
shaking=[[A*Omega*(i-L/2.0),i] for i in range(L)]

def drive(t,Omega):
	return np.cos(Omega*t)

drive_args=[Omega]

static=[["+-",hopping],["-+",hopping],['n',trap]]
dynamic=[["n",shaking,drive,drive_args]]

#### define boson model

basis = boson_basis_1d(L,Nb=1,sps=2)

H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,check_herm=False,check_symm=False,check_pcon=False)
E,V=H.eigh(time=np.pi/(2*Omega) )


"""
plt.scatter(range(H.Ns),E)
plt.show()
exit()
"""

def GPE(time,V,H,g=1.0):

	V_dot = np.zeros_like(V)

	Ns=H.Ns

	# static single-particle
	V_dot[Ns:] =  H.static.dot(V[:Ns])
	V_dot[:Ns] = -H.static.dot(V[Ns:])


	# static GPE interaction
	V_dot_2 = (np.abs(V[:Ns])**2 + np.abs(V[Ns:])**2)
	V_dot[Ns:] += g*V_dot_2*V[:Ns]
	V_dot[:Ns] -= g*V_dot_2*V[Ns:]

	# dynamic single-particle
	for Hd,f,f_args in H.dynamic:
		V_dot[Ns:] += f(time,*f_args)*Hd.dot(V[:Ns])
		V_dot[:Ns] -= f(time,*f_args)*Hd.dot(V[Ns:])

	return V_dot


GPE_params = (H,g)

N = 20
t=Floquet_t_vec(Omega,N,len_T=1)


evolve_args={'ode_args':GPE_params,'solver_args':{} }
y_t = evolve(V[:,0],t.i,t.vals,GPE,real=True,iterate=True,**evolve_args)


for y in y_t:

	#plt.scatter(range(H.Ns),abs(y)**2)
	
	plt.scatter(range(H.Ns), np.fft.fft(y.conj())*np.fft.fft(y), color='red' )

	#plt.ylim([0,0.02])

	plt.show()
	#exit()





#### 

#np.testing.assert_allclose(E-E_ho,0.0,atol=1E-5,err_msg='Failed boson and ho energies comparison!')

