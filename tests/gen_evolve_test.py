from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis
from quspin.tools.evolution import evolve
from quspin.tools.Floquet import Floquet_t_vec
import numpy as np # generic math functions
import scipy.sparse as sp

import matplotlib.pyplot as plt

##### define model parameters #####
L=20 # system size
if L%2==0:
	i_CM = L//2-0.5 # centre of chain
else:
	i_CM = L//2

# model params
J=1.0
mu=0.002

A=1.0
Omega=2.5 


hopping=[[-J,i,(i+1)%L] for i in range(L-1)]
trap=[[mu*(i-i_CM)**2,i] for i in range(L)]
shaking=[[A*Omega*(i-i_CM),i] for i in range(L)]

# define basis
basis = boson_basis_1d(L,Nb=1,sps=2)

### Hamiltonian
def drive(t,Omega):
	return np.cos(Omega*t)

drive_args=[Omega]

static=[["+-",hopping],["-+",hopping],['n',trap]]
dynamic=[["n",shaking,drive,drive_args]]

#### calculate Hamiltonian

H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)

E,V=H.eigh()



def SO_real(time,V,H):

	V_dot = np.zeros_like(V)

	Ns=H.Ns

	# static single-particle
	V_dot[:Ns] =  H.static.dot(V[Ns:])
	V_dot[Ns:] = -H.static.dot(V[:Ns])


	# dynamic single-particle
	for func,Hd in H.dynamic.items():
		V_dot[:Ns] += func(time)*Hd.dot(V[Ns:])
		V_dot[Ns:] -= func(time)*Hd.dot(V[:Ns])

	return V_dot


##### time evolutino
t=Floquet_t_vec(Omega,20,len_T=1)


for H_real in [0,1]:
	
	y_genevolve = evolve(V[:,0],t.i,t.f,SO_real,real=True,stack_state=True,f_params=(H,))
	y = H.evolve(V[:,0],t.i,t.f,H_real=H_real)
	
	np.testing.assert_allclose(y-y_genevolve,0.0,atol=1E-6,err_msg='Failed evolve_scalar comparison!')



	y_genevolve = evolve(V[:,0],t.i,t.vals,SO_real,real=True,stack_state=True,f_params=(H,))
	y = H.evolve(V[:,0],t.i,t.vals,H_real=H_real)

	np.testing.assert_allclose(y-y_genevolve,0.0,atol=1E-6,err_msg='Failed evolve_list comparison!')



	y_genevolve = evolve(V[:,0],t.i,t.vals,SO_real,real=True,stack_state=True,iterate=True,f_params=(H,))
	y = H.evolve(V[:,0],t.i,t.vals,iterate=True,H_real=H_real)

	for y_genevolve_t,y_t in zip(y_genevolve,y):
		np.testing.assert_allclose(y_t-y_genevolve_t,0.0,atol=1E-6,err_msg='Failed evolve_iter comparison!')

