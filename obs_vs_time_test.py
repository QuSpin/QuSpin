from qspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from qspin.operators import hamiltonian # Hamiltonian and observables
from qspin.tools.measurements import obs_vs_time
import numpy as np
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers
seed()


"""
This test only makes sure the function 'obs_vs_time' runs properly.
"""

dtypes={"float32":np.float32,"float64":np.float64,"float128":np.float128,
		"complex64":np.complex64,"complex128":np.complex128,"complex256":np.complex256}

atols={"float32":1E-4,"float64":1E-13,"float128":1E-13,
		"complex64":1E-4,"complex128":1E-13,"complex256":1E-13}


def drive(t):
	return np.exp(-0.25*t)*np.cos(t)
drive_args=[]

def time_dep(t):
	return np.cosh(-0.25*t)*np.cos(2.0*t)

L=6
basis = spin_basis_1d(L,kblock=0,pblock=1,zblock=1)
J_zxz = [[1.0,i,(i+1)%L,(i+2)%L] for i in range(0,L)]
J_zz = [[1.0,i,(i+1)%L] for i in range(0,L)] 
J_xy = [[1.0,i,(i+1)%L] for i in range(0,L)]
J_yy = [[1.0,i,(i+1)%L] for i in range(0,L)]
# static and dynamic lists
static_pm = [["+-",J_xy],["-+",J_xy]]
static_yy = [["yy",J_yy]]
dynamic_zz = [["zz",J_zz,time_dep,drive_args]]
dynamic_zxz = [["zxz",J_zxz,drive,drive_args]]






dtype=dtypes["float64"]
atol=atols["float64"]

H=hamiltonian(static_pm,dynamic_zxz,basis=basis,dtype=dtype,check_herm=False,check_symm=False)
Ozz=hamiltonian([],dynamic_zz,basis=basis,dtype=dtype,check_herm=False,check_symm=False)
H2=hamiltonian(static_yy,[],basis=basis,dtype=dtype,check_herm=False,check_symm=False) 

_,psi0 = H.eigsh(time=0,k=1,sigma=-100.)
psi0=psi0.squeeze()

t=np.linspace(0,2,20)
psi_t=H.evolve(psi0,0.0,t,iterate=True,rtol=atol,atol=atol)
psi_t2=H.evolve(psi0,0.0,t,rtol=atol,atol=atol)

Obs_list = [Ozz,Ozz(time=np.sqrt(np.exp(0.0)) )] 
Sent_args={'basis':basis,'chain_subsys':range(L/2)}

Obs = obs_vs_time(psi_t,t,Obs_list,return_state=True,Sent_args=Sent_args)
Obs2 = obs_vs_time(psi_t2.T,t,Obs_list,return_state=True,Sent_args=Sent_args)


Expn = Obs['Expt_time']
psi_t = Obs['psi_t']
Sent = Obs['Sent_time']['Sent']

Expn2 = Obs2['Expt_time']
psi_t2 = Obs2['psi_t']
Sent2 = Obs2['Sent_time']['Sent']


np.testing.assert_allclose(Expn,Expn2,atol=atol,err_msg='Failed observable comparison!')
np.testing.assert_allclose(psi_t,psi_t2,atol=atol,err_msg='Failed state comparison!')
np.testing.assert_allclose(Sent,Sent2,atol=atol,err_msg='Failed ent entropy comparison!')

### check obs_vs_time vs ED

E,V = H2.eigh()

psi_t=H2.evolve(psi0,0.0,t,iterate=True,rtol=atol,atol=atol)

Obs = obs_vs_time(psi_t,t,Obs_list,return_state=True,Sent_args=Sent_args)
Obs2 = obs_vs_time((psi0,E,V),t,Obs_list,return_state=True,Sent_args=Sent_args)

Expn = Obs['Expt_time']
psi_t = Obs['psi_t']
Sent = Obs['Sent_time']['Sent']

Expn2 = Obs2['Expt_time']
psi_t2 = Obs2['psi_t']
Sent2 = Obs2['Sent_time']['Sent']


np.testing.assert_allclose(Expn,Expn2,atol=atol,err_msg='Failed observable comparison!')
np.testing.assert_allclose(psi_t,psi_t2,atol=atol,err_msg='Failed state comparison!')
np.testing.assert_allclose(Sent,Sent2,atol=atol,err_msg='Failed ent entropy comparison!')





