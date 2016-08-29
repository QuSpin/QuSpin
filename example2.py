from qspin.hamiltonian import hamiltonian,expO
from qspin.basis import spin_basis_1d
from qspin.tools import observables
from qspin.tools.Floquet import Floquet, Floquet_t_vec
import numpy as np

import scipy.sparse.linalg as _sla  

#
##### define model parameters #####
L=4 # system size
J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field
# define periodic step drive
def drive(t,Omega):
	return np.sign(np.cos(Omega*t))
Omega=10.0 # drive frequency
drive_args=[Omega]
#
##### set up alternating Hamiltonians #####
# compute basis in the 0-total momentum and +1-parity sector
basis=spin_basis_1d(L=L,kblock=0,pblock=1)
# define operators with PBC
x_field_pos=[[+g,i]		for i in range(L)]
x_field_neg=[[-g,i]		for i in range(L)]
z_field=[[h,i]		for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)]
# static and dynamic parts
static=[["zz",J_nn],["z",z_field],["x",x_field_pos]]
dynamic=[["zz",J_nn,drive,drive_args],["z",z_field,drive,drive_args],["x",x_field_neg,drive,drive_args]]
# compute Hamiltonians
H=0.5*hamiltonian(static,dynamic,dtype=np.float64,basis=basis)
#
##### set up second-order van Vleck Floquet Hamiltonian #####
# zeroth-order term
Heff_0=H(time=0)
# second-order term
Heff2_term_1=[[J**2*g			,i,(i+1)%L,(i+2)%L] for i in range(L)]
Heff2_term_2=[[J*g*h 			,i,(i+1)%L] 		for i in range(L)]
Heff2_term_3=[[-J*g**2 			,i,(i+1)%L] 		for i in range(L)]
Heff2_term_4=[[J**2*g+0.5*h**2*g,i] 				for i in range(L)]
Heff2_term_5=[[0.5*h*g**2		,i] 				for i in range(L)]
# define static part
Heff_static=[["zxz",Heff2_term_1],
			 ["xz",Heff2_term_2],["zx",Heff2_term_2],
			 ["yy",Heff2_term_3],["zz",Heff2_term_2],
			 ["x",Heff2_term_4],
			 ["z",Heff2_term_5]							] 
# compute Hamiltonian
Heff_2=hamiltonian(Heff_static,[],dtype=np.float64,basis=basis)
Heff_2*=-np.pi**2/(12.0*Omega**2)
# zeroth + second order
Heff_02=Heff_0+Heff_2
#
##### set up second-order van Vleck Kick operator #####
Keff2_term_1=[[J*g,i,(i+1)%L] for i in range(L)]
Keff2_term_2=[[h*g,i] for i in range(L)]
# define static part
Keff_static=[["zy",Keff2_term_1],["yz",Keff2_term_1],["y",Keff2_term_2]]
Keff_02=hamiltonian(Keff_static,[],dtype=np.complex128,basis=basis)
Keff_02*=-np.pi**2/(8*Omega**2)


##### rotate Heff to stroboscopic basis #####
V_K = _sla.expm(-1j*Keff_02.tocsr()).astype(np.float64)
HF_02 = hamiltonian( [(V_K.dot(Heff_02)).dot(V_K.T.conj())],[] )


##### calculate exact Floquet eigensystem #####
T = 2.0*np.pi/Omega # calculate period
# call Floquet class
t_list = np.array([0.0,T/4.0,3.0*T/4.0]) + np.finfo(float).eps # times to evaluate H at
dt_list = np.array([T/4.0,T/2.0,T/4.0]) # time duration of each step to apply H(t) for
Floq = Floquet({'H':H,'t_list':t_list,'dt_list':dt_list},VF=True)
VF = Floq.VF # Floquet states
EF = Floq.EF # quasienergies
del Floq


##### calculate only initial state and its energy
E, psi_i = HF_02.eigsh(k=1,sigma=-100.0)
psi_i = psi_i.squeeze()

##### time-dependent measurements
# define time vector of stroboscopic times
t=Floquet_t_vec(Omega,10,1) # t.vals = times, t.i = initial time, t.T = driving period
# calculate measurements
Sent_args = {'basis':basis}
meas = observables.obs_vs_time((psi_i,EF.copy(),VF.copy(),t.vals),[1.0/L*HF_02],return_state=True,Sent_args=Sent_args)
# read off measurements
Energy_t = meas['Expt_time']
psi_t = meas['psi_t']
Entropy_t = meas['Sent_time']['Sent']
# double check evolution by solving Schroedingers eqn
psi_t2 = H.evolve(psi_i,t.i,t.vals,iterate=True,rtol=1E-9,atol=1E-9)



Energy_t2, Entropy_t2 = [], []
for psi in psi_t2:
	Energy_t2.append( 1.0/L*HF_02.matrix_ele(psi,psi).real )
	Entropy_t2.append( observables.ent_entropy(psi,**Sent_args) )

print np.round( Energy_t.T, 3)
print np.round( Energy_t2, 3)

#####

