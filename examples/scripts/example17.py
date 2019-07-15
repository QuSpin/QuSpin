from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#
from quspin.operators import hamiltonian, commutator, anti_commutator
from quspin.basis import spin_basis_1d # Hilbert space spin basis_1d
from quspin.tools.evolution import evolve
import numpy as np
from six import iteritems # loop over elements of dictionary
import matplotlib.pyplot as plt # plotting library
#
###### model parameters
#
L=1 # one qubit
delta=1.0 # detuning
Omega_0=np.sqrt(2) # bare Rabi frequency
Omega_Rabi=np.sqrt(Omega_0**2+delta**2) # Rabi frequency
gamma = np.sqrt(3)/2.0 # decay rate
#
##### create Hamiltonian to evolve unitarily
# basis
basis=spin_basis_1d(L,pauli=-1) # uses convention "+" = [[0,1],[0,0]]
# site-coupling lists
hx_list=[[Omega_0,i] for i in range(L)]
hz_list=[[delta,i] for i in range(L)]
# static opstr list 
static_H=[['x',hx_list],['z',hz_list]]
# hamiltonian
H=hamiltonian(static_H,[],basis=basis,dtype=np.float64)
print('Hamiltonian:\n',H.toarray())
#
##### create Lindbladian
# site-coupling lists
L_list=[[1.0,i] for i in range(L)]
# static opstr list 
static_L=[['+',L_list]]
# Lindblad operator
L=hamiltonian(static_L,[],basis=basis,dtype=np.complex128,check_herm=False)
print('Lindblad operator:\n',L.toarray())
# pre-compute operators for efficiency
L_dagger=L.getH()
L_daggerL=L_dagger*L
#
##### define Lindblad equation in diagonal form
# slow, straightforward function
def Lindblad_EOM_v1(time,rho):
	"""
	This function solves the complex-valued time-dependent GPE:
	$$ \dot\rho(t) = -i[H,\rho(t)] + 2\gamma\left( L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho \} \right) $$
	"""
	# solve static part of Lindblad equation
	rho = rho.reshape((H.Ns,H.Ns))
	rho_dot = (-1j*commutator(H,rho) + 2.0*gamma*L*rho*L_dagger - gamma*anti_commutator(L_daggerL,rho) ).static
	# solve dynamic part of Lindblad equation (no time-dependence in Lindbladian for this example)
	for f,Hd in iteritems(H.dynamic):
		rho_dot += -1j*f(time)*commutator(Hd,rho)
	return rho_dot.ravel()
# fast function (not as memory efficient)
def Lindblad_EOM_v2(time,rho,rho_out,rho_aux):
	"""
	This function solves the complex-valued time-dependent GPE:
	$$ \dot\rho(t) = -i[H,\rho(t)] + 2\gamma\left( L\rho L^\dagger - \frac{1}{2}\{L^\dagger L, \rho \} \right) $$
	"""
	rho = rho.reshape((H.Ns,H.Ns))
	### Hamiltonian part
	# commutator term (unitary
	# rho_out = H._static.dot(rho))
	H._static_matvec(H.static  ,rho  ,out=rho_out  ,a=+1.0,overwrite_out=True)
	# rho_out -= (H._static.T.dot(rho.T)).T // RHS~rho.dot(H) 
	H._static_matvec(H.static.T,rho.T,out=rho_out.T,a=-1.0,overwrite_out=False)
	# 
	for func,Hd in iteritems(H._dynamic):
		ft = func(time)
		H._dynamic_matvec[func](Hd  ,rho  ,out=rho_out  ,a=+ft,overwrite_out=False) # rho_out += ft*Hd.dot(rho)
		H._dynamic_matvec[func](Hd.T,rho.T,out=rho_out.T,a=-ft,overwrite_out=False) # rho_out -= ft*(Hd.T.dot(rho.T)).T
	# multiply by -1j
	rho_out *= -1.0j
	#
	### Lindbladian part (static only)
	# 1st Lindblad term (nonunitary)
	# rho_aux = 2\gamma*L.dot(rho)
	L._static_matvec(L.static  ,rho             ,out=rho_aux  ,a=+2.0*gamma,overwrite_out=True)
	# rho_out += (L.static.T.conj().dot(rho_aux.T)).T // RHS~rho_aux.dot(L_dagger) 
	L._static_matvec(L.static.T.conj(),rho_aux.T,out=rho_out.T,a=+1.0,overwrite_out=False) 
	# anticommutator (2nd Lindblad) term (nonunitary)
	# rho_out += gamma*L_daggerL._static.dot(rho)
	L._static_matvec(L_daggerL.static  ,rho  ,out=rho_out  ,a=-gamma,overwrite_out=False)
	# rho_out += gamma*(L_daggerL._static.T.dot(rho.T)).T // RHS~rho.dot(L_daggerL) 
	L._static_matvec(L_daggerL.static.T,rho.T,out=rho_out.T,a=-gamma,overwrite_out=False) 
	
	# # anticommutator (2nd Lindblad) term (nonunitary)
	# # rho_aux = L._static.dot(rho)
	# L._static_matvec(L.static         ,rho      ,out=rho_aux  ,a=1.0,overwrite_out=True)
	# # rho_out += gamma*L_dagger(rho_aux)
	# L._static_matvec(L_dagger.static  ,rho_aux  ,out=rho_out  ,a=-gamma,overwrite_out=False)
	# # rho_aux = (L_dagger._static.T.dot(rho.T)).T // RHS~rho.dot(L_dagger) 
	# L._static_matvec(L.static.T.conj()       ,rho.T    ,out=rho_aux.T,a=1.0,overwrite_out=True)
	# # rho_out += gamma*(L._static.T.dot(rho_aux.T)).T // RHS~rho_aux.dot(L)
	# L._static_matvec(L_dagger.static.T.conj(),rho_aux.T,out=rho_out.T,a=-gamma,overwrite_out=False)

	#
	return rho_out.ravel()
#
# define auxiliary arguments
EOM_args=(np.zeros((H.Ns,H.Ns),dtype=np.complex128,order="C"),    # auxiliary variable rho_out
		  np.zeros((H.Ns,H.Ns),dtype=np.complex128,order="C"),  ) # auxiliary variable rho_aux
#
##### time-evolve state according to Lindlad equation
# define real time vector
t_max=6.0
time=np.linspace(0.0,t_max,101)
# define initial state
rho0=np.array([[0.5,0.5j],[-0.5j,0.5]],dtype=np.complex128)
# slow solution
#rho_t = evolve(rho0,time[0],time,Lindblad_EOM_v1,iterate=True,atol=1E-12,rtol=1E-12)
# fast solution (but 3 times as memory intensive)
rho_t = evolve(rho0,time[0],time,Lindblad_EOM_v2,f_params=EOM_args,iterate=True,atol=1E-12,rtol=1E-12) 
#
##### compute state evolution
#
population_down=np.zeros(time.shape,dtype=np.float64)
for i,rho_flattened in enumerate(rho_t):
	rho=rho_flattened.reshape(H.Ns,H.Ns)
	population_down[i]=rho_flattened[1,1]
	print("time={0:.2f}, population of down state ={1:0.8f}".format(time[i],population_down[i]) )
#
##### plot population dynamics of down state
# plot data
plt.plot(Omega_Rabi*time, population_down)
plt.xlabel('$\\Omega_R t$')
plt.ylabel('$\\rho_{\\downarrow\\downarrow}$')
plt.xlim(time[0],time[-1])
plt.ylim(0.0,1.0)
plt.tight_layout()
plt.show()
