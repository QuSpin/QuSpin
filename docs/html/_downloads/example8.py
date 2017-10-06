from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # Hilbert space boson basis
from quspin.tools.measurements import evolve # nonlinear evolution 
import numpy as np # generic math functions
import matplotlib.pyplot as plt # plotting library
#
##### define model parameters #####
L=300 # system size
# calculate centre of chain
if L%2==0:
	j0 = L//2-0.5 # centre of chain
else:
	j0 = L//2 # centre of chain
sites=np.arange(L)-j0
# static parameters
J=1.0 # hopping
U=1.0 # Bose-Hubbard interaction strength
# dynamic parameters
kappa_trap_i=0.001 # initial chemical potential
kappa_trap_f=0.0001 # final chemical potential
t_ramp=40.0/J # set total ramp time
# ramp protocol
def ramp(t,kappa_trap_i,kappa_trap_f,t_ramp):
	return  (kappa_trap_f - kappa_trap_i)*t/t_ramp + kappa_trap_i
# ramp protocol parameters
ramp_args=[kappa_trap_i,kappa_trap_f,t_ramp]
#
##### construct single-particle Hamiltonian #####
# define site-coupling lists
hopping=[[-J,i,(i+1)%L] for i in range(L-1)]
trap=[[0.5*(i-j0)**2,i] for i in range(L)]
# define static and dynamic lists
static=[["+-",hopping],["-+",hopping]]
dynamic=[['n',trap,ramp,ramp_args]]
# define basis
basis = boson_basis_1d(L,Nb=1,sps=2)
# build Hamiltonian
Hsp=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
E,V=Hsp.eigsh(time=0.0,k=1,which='SA')
#
##### imaginary-time evolution to compute GS of GPE #####
def GPE_imag_time(tau,phi,Hsp,U):
	"""
	This function solves the real-valued GPE in imaginary time:
	$$ -\dot\phi(\tau) = Hsp(t=0)\phi(\tau) + U |\phi(\tau)|^2 \phi(\tau) $$
	"""
	return -( Hsp.dot(phi,time=0) + U*np.abs(phi)**2*phi )
# define ODE parameters
GPE_params = (Hsp,U)
# define initial state to flow to GS from
phi0=V[:,0]*np.sqrt(L) # initial state normalised to 1 particle per site
# define imaginary time vector
tau=np.linspace(0.0,35.0,71)
# evolve state in imaginary time
psi_tau = evolve(phi0,tau[0],tau,GPE_imag_time,f_params=GPE_params,
							imag_time=True,real=True,iterate=True)
#
# display state evolution
for i,psi0 in enumerate(psi_tau):
	# compute energy
	E_GS=(Hsp.matrix_ele(psi0,psi0,time=0) + 0.5*U*np.sum(np.abs(psi0)**4) ).real
	# plot wave function
	plt.plot(sites, abs(phi0)**2, color='r',marker='s',alpha=0.2,
										label='$|\\phi_j(0)|^2$')
	plt.plot(sites, abs(psi0)**2, color='b',marker='o',
								label='$|\\phi_j(\\tau)|^2$' )
	plt.xlabel('$\\mathrm{lattice\\ sites}$',fontsize=14)
	plt.title('$J\\tau=%0.2f,\\ E_\\mathrm{GS}(\\tau)=%0.4fJ$'%(tau[i],E_GS)
																,fontsize=14)
	plt.ylim([-0.01,max(abs(phi0)**2)+0.01])
	plt.legend(fontsize=14)
	plt.draw() # draw frame
	plt.pause(0.005) # pause frame
	plt.clf() # clear figure
plt.close()
#
##### real-time evolution of GPE #####
def GPE(time,psi):
	"""
	This function solves the complex-valued time-dependent GPE:
	$$ i\dot\psi(t) = Hsp(t)\psi(t) + U |\psi(t)|^2 \psi(t) $$
	"""
	# solve static part of GPE
	psi_dot = Hsp.static.dot(psi) + U*np.abs(psi)**2*psi
	# solve dynamic part of GPE
	for Hd,f,f_args in Hsp.dynamic:
		psi_dot += f(time,*f_args)*Hd.dot(psi)
	return -1j*psi_dot
# define real time vector
t=np.linspace(0.0,t_ramp,101)
# time-evolve state according to GPE
psi_t = evolve(psi0,t[0],t,GPE,iterate=True,atol=1E-12,rtol=1E-12)
#
# display state evolution
for i,psi in enumerate(psi_t):
	# compute energy
	E=(Hsp.matrix_ele(psi,psi,time=t[i]) + 0.5*U*np.sum(np.abs(psi)**4) ).real
	# compute trap
	kappa_trap=ramp(t[i],kappa_trap_i,kappa_trap_f,t_ramp)*(sites)**2
	# plot wave function
	plt.plot(sites, abs(psi0)**2, color='r',marker='s',alpha=0.2
								,label='$|\\psi_{\\mathrm{GS},j}|^2$')
	plt.plot(sites, abs(psi)**2, color='b',marker='o',label='$|\\psi_j(t)|^2$')
	plt.plot(sites, kappa_trap,'--',color='g',label='$\\mathrm{trap}$')
	plt.ylim([-0.01,max(abs(psi0)**2)+0.01])
	plt.xlabel('$\\mathrm{lattice\\ sites}$',fontsize=14)
	plt.title('$Jt=%0.2f,\\ E(t)-E_\\mathrm{GS}=%0.4fJ$'%(t[i],E-E_GS),fontsize=14)
	plt.legend(loc='upper right',fontsize=14)
	plt.draw() # draw frame
	plt.pause(0.00005) # pause frame
	plt.clf() # clear figure
plt.close()