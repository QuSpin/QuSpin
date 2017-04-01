from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space spin basis
from quspin.tools.measurements import evolve
from quspin.tools.Floquet import Floquet_t_vec
import numpy as np # generic math functions
import scipy.sparse as sp

import matplotlib.pyplot as plt

##### define model parameters #####
L=100 # system size
if L%2==0:
	i_CM = L//2-0.5 # centre of chain
else:
	i_CM = L//2

q_vec=2*np.pi*np.fft.fftfreq(L)

J=1.0
mu=0.002

g = 17.43
rho = 96*(425E-3)**2 # 27*(425E-3)^2
U = g/rho # Bose-Hubbard interaction strength


A=1.0
Omega=2.5  #3.13 # 4*Jeff = 3.06079


hopping=[[-J,i,(i+1)%L] for i in range(L-1)]
trap=[[mu*(i-i_CM)**2,i] for i in range(L)]
shaking=[[A*Omega*(i-i_CM),i] for i in range(L)]

# define basis
basis = boson_basis_1d(L,Nb=1,sps=2)

### lab-frame Hamiltonian
def drive(t,Omega):
	return np.cos(Omega*t)

drive_args=[Omega]

static=[["+-",hopping],["-+",hopping],['n',trap]]
dynamic=[["n",shaking,drive,drive_args]]

### rot-frame Hamiltonian
def drive_rot(t,Omega):
	return np.exp(-1j*A*np.sin(Omega*t) )

def drive_rot_cc(t,Omega):
	return np.exp(+1j*A*np.sin(Omega*t) )

drive_args=[Omega]

static_rot=[['n',trap]]
dynamic_rot=[["+-",hopping,drive_rot,drive_args],["-+",hopping,drive_rot_cc,drive_args]]


#### calculate Hamiltonian

#,check_herm=False,check_symm=False,check_pcon=False
H_static=hamiltonian(static,[],basis=basis,dtype=np.float64)
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
H_rot=hamiltonian(static_rot,dynamic_rot,basis=basis,dtype=np.complex128)

E,V=H_static.eigh()
E_rot,V_rot=H_rot.eigh()



def GPE(time,V,H,U):

	V_dot = np.zeros_like(V)

	Ns=H.Ns

	# static single-particle
	V_dot[Ns:] =  H.static.dot(V[:Ns])
	V_dot[:Ns] = -H.static.dot(V[Ns:])


	# static GPE interaction
	V_dot_2 = (np.abs(V[:Ns])**2 + np.abs(V[Ns:])**2)
	V_dot[Ns:] += U*V_dot_2*V[:Ns]
	V_dot[:Ns] -= U*V_dot_2*V[Ns:]

	# dynamic single-particle
	for Hd,f,f_args in H.dynamic:
		V_dot[Ns:] += f(time,*f_args)*Hd.dot(V[:Ns])
		V_dot[:Ns] -= f(time,*f_args)*Hd.dot(V[Ns:])

	return V_dot


def GPE_cpx(time,V,H,U):

	V_dot = np.zeros_like(V)

	Ns=H.Ns

	# static single-particle
	V_dot[Ns:] =  H.static.dot(V[:Ns]).real
	V_dot[:Ns] = -H.static.dot(V[Ns:]).real


	# static GPE interaction
	V_dot_2 = (np.abs(V[:Ns])**2 + np.abs(V[Ns:])**2)
	V_dot[Ns:] += U*V_dot_2*V[:Ns]
	V_dot[:Ns] -= U*V_dot_2*V[Ns:]

	# dynamic single-particle
	for Hd,f,f_args in H.dynamic:
		V_dot[Ns:] +=  ( +(f(time,*f_args).real)*Hd.dot(V[:Ns]) + (f(time,*f_args).imag)*Hd.dot(V[Ns:]) ).real
		V_dot[:Ns] +=  ( -(f(time,*f_args).real)*Hd.dot(V[Ns:]) + (f(time,*f_args).imag)*Hd.dot(V[:Ns]) ).real

	return V_dot


def GPE_imag_time(time,V,H,U):

	"""
	\dot y = - ([f+ih]Hy + g abs(y)^2 y)

	y = u + iv

	\dot u + i\dot v = - { [f+ih]H(u + iv) + U( abs(u)^2 + abs(v)^2 )(u + iv) }
					 = - { fHu - hHv + i(fHv + hHu) + U( abs(u)^2 + abs(v)^2 )(u + iv)  }

	\dot u = - { fHu - hHv + U( abs(u)^2 + abs(v)^2 )u }
	\dot v = - { fHv + hHu + U( abs(u)^2 + abs(v)^2 )v }
	"""

	V_dot = np.zeros_like(V)

	Ns=H.Ns

	# static single-particle
	V_dot[Ns:] = H.static.dot(V[Ns:]).real
	V_dot[:Ns] = H.static.dot(V[:Ns]).real
	
	# static GPE interaction 
	V_dot_2 = (np.abs(V[:Ns])**2 + np.abs(V[Ns:])**2)
	V_dot[Ns:] += U*V_dot_2*V[Ns:]
	V_dot[:Ns] += U*V_dot_2*V[:Ns]

	# dynamic single-particle
	for Hd,f,f_args in H.dynamic:
		V_dot[Ns:] += ( (f(time,*f_args).real)*Hd.dot(V[Ns:]) - (f(time,*f_args).imag)*Hd.dot(V[:Ns]) ).real
		V_dot[:Ns] += ( (f(time,*f_args).real)*Hd.dot(V[:Ns]) + (f(time,*f_args).imag)*Hd.dot(V[Ns:]) ).real

	return -V_dot


def GPE_imag_time2(time,V,H,U):
	return -( H.static.dot(V) + U*np.abs(V)**2*V )






##### imaginary-time evolutino
t_imag=Floquet_t_vec(Omega,20,len_T=1)

GPE_params = (H_static,U) #
evolve_args={'ode_args':GPE_params,'solver_args':{} }
y0_t = evolve(V[:,0],t_imag.i,t_imag.vals,GPE_imag_time2,real=False,iterate=True,imag_time=True,**evolve_args)

E_old=0.
for y0 in y0_t:
	E_new=1.0/L*(H_static.matrix_ele(y0,y0) + 0.5*U*np.sum(np.abs(0)**4) ).real
	#print("energy:", E_new )
	
	E_old=E_new
print('finished calculating GS w/ conv error', E_old-E_new)



print("GS energy:",     1.0/L*(H_static.matrix_ele(V[:,0],V[:,0]) + np.sum(0.5*U*np.abs(V[:,0])**4) ) )
#print("GS kin energy:", 1.0/L*np.sum( H_static.matrix_ele(V[:,0],V[:,0]) )    )
#print("GS int energy:", 1.0/L*np.sum( 0.5*g*np.abs(V[:,0])**4)     )


print("GPE energy:",     1.0/L*(H_static.matrix_ele(y0,y0) + np.sum(0.5*U*np.abs(y0)**4) ) )
#print("GPE kin energy:", 1.0/L*np.sum( H_static.matrix_ele(y,y) ) )
#print("GPE int energy:", 1.0/L*np.sum( 0.5*g*np.abs(y)**4) )

"""
plt.scatter(np.arange(L)-i_CM, abs(V[:,0])**2, color='green' )
plt.scatter(np.arange(L)-i_CM, abs(y0)**2, color='red' )
plt.show()

plt.scatter(q_vec, abs(np.fft.fft(V[:,0]))**2/L, color='green' )
plt.scatter(q_vec, abs(np.fft.fft(y0))**2/L, color='red' )
plt.show()
"""


### TOF
def ToF(L,d,alpha,beta,corr):
	"""
	calculates density after time of flight. The parameters are
	L: number of lattice sites
	d: lattice spacing
	x = r/d: dim'less position
	alpha = 1.0/( (sigma/d)**2 + (hbar*t/(m*sigma*d))**2 )
	beta = alpha*hbar*t/(m*sigma**2)
	corr = \langle a^\dagger_j a_l \rangle: two-point function
	"""
	from numpy import exp

	n_ToF= np.zeros((L,))

	prefactor=1.0/d*np.sqrt(alpha/np.pi)

	for i,xi in enumerate( range(-L//2,L//2,1) ):
		xi+=i_CM
		S=0.0j

		for j,xj in enumerate(range(-L//2,L//2,1)):
			xj+=i_CM
			for l,xl in enumerate(range(-L//2,L//2,1)):
				xl+=i_CM

				#print(alpha*xi**2 + alpha*xi*(xl+xj) - 0.5*alpha*(xl**2+xj**2) )

				S+=corr[j,l]*exp( #  alpha*xi**2 \
								  #+ alpha*xi*(xl+xj) \
								  #- 0.5*alpha*(xl**2+xj**2) \
								  - 1j*xi*(xl-xj) \
								  + 0.5j*beta*(xl**2-xj**2) 
								)

		#print(S)
		n_ToF[i]=prefactor*S.real
		#exit()

	return n_ToF


sigma=50e-9
d=425e-9
hbar=1.0545718e-34
m=6.476106e-26
t_toF=2000.0e-6

alpha=1.0/( (sigma/d)**2 + (hbar*t_toF/(m*sigma*d))**2 )
beta = alpha*hbar*t_toF/(m*sigma**2)

corr=np.outer(y0.conj(),y0)/L**2



n_ToF=ToF(L,d,alpha,beta,corr)


plt.scatter(np.arange(L)-i_CM, n_ToF, color='red' )
plt.show()

exit()


##### real-time evolution

N=1600
t=Floquet_t_vec(Omega,N,len_T=1)


GPE_params = (H_rot,U) #
evolve_args={'ode_args':GPE_params,'solver_args':{} }
y_t = evolve(y0,t.i,t.vals,GPE_cpx,real=True,iterate=True,**evolve_args)

print('starting real-time evolution...')
E=[]
for i,y in enumerate(y_t):
	E.append( 1.0/L*(H_static.matrix_ele(y,y) + 0.5*U*np.sum(np.abs(y)**4) ).real )
	print("(N_T,E)=:", (t.vals[i]/t.T,E[-1]) )

	#plt.scatter(np.arange(L)-i_CM, abs(y)**2, color='blue' )
	#plt.show()

	plt.plot(q_vec, abs(np.fft.fft(y))**2/L, color='blue',marker='o' )
	plt.ylim([0.0,0.1])
	plt.title('$\\mathrm{period}\\ l=%i$'%(i))
	
	plt.draw()
	plt.pause(0.01)
	plt.clf()
plt.close()

plt.plot(t.vals/t.T,E-E[0])
plt.show()

plt.scatter(np.arange(L)-i_CM, abs(V[:,0])**2, color='green' )
plt.scatter(np.arange(L)-i_CM, abs(y0)**2, color='red' )
plt.scatter(np.arange(L)-i_CM, abs(y)**2, color='blue' )
plt.show()

plt.scatter(q_vec, abs(np.fft.fft(V[:,0]))**2/L, color='green' )
plt.scatter(q_vec, abs(np.fft.fft(y0))**2/L, color='red' )
plt.scatter(q_vec, abs(np.fft.fft(y))**2/L, color='blue' )
plt.show()




#### 

#np.testing.assert_allclose(E-E_ho,0.0,atol=1E-5,err_msg='Failed boson and ho energies comparison!')

