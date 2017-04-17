import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian
from quspin.tools.measurements import ent_entropy, _reshape_as_subsys
import numpy as np
import scipy.sparse as sp

L=5

Jxz=[[1.0,i,i+1] for i in range(L-1)]
hx=[[(-1)**i*np.sqrt(2),i] for i in range(L-1)]

static = [["zx",Jxz],["xz",Jxz],["y",hx]]
H = hamiltonian(static,[],N=L,dtype=np.complex128)
basis =  H.basis

E,V = H.eigh()



########## DENSE STATE ###########

state=V[:,0]
sub_sys_A=[i for i in range(basis.L//2)]

#####

p=basis._p_pure(state,sub_sys_A)
lmbda=ent_entropy(state,basis,sub_sys_A,svd_return_vec=[0,1,0])['lmbda']

np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')

#####

p,p_rdm_A=basis._p_pure(state,sub_sys_A,return_rdm='A')
Sent=ent_entropy(state,basis,sub_sys_A,DM='chain_subsys',svd_return_vec=[0,1,0])
lmbda=Sent['lmbda']
rdm_A=Sent['DM_chain_subsys']

np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')
np.testing.assert_allclose(p_rdm_A-rdm_A,0.0,atol=1E-5,err_msg='Failed subsys_A comparison!')

#####

p,p_rdm_B=basis._p_pure(state,sub_sys_A,return_rdm='B')
Sent=ent_entropy(state,basis,sub_sys_A,DM='other_subsys',svd_return_vec=[0,1,0])
lmbda=Sent['lmbda']
rdm_B=Sent['DM_other_subsys']

np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')
np.testing.assert_allclose(p_rdm_B-rdm_B,0.0,atol=1E-5,err_msg='Failed subsys_B comparison!')

#####

p,p_rdm_A,rdm_B=basis._p_pure(state,sub_sys_A,return_rdm='both')
Sent=ent_entropy(state,basis,sub_sys_A,DM='both',svd_return_vec=[0,1,0])
lmbda=Sent['lmbda']
rdm_A=Sent['DM_chain_subsys']
rdm_B=Sent['DM_other_subsys']

np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')
np.testing.assert_allclose(p_rdm_A-rdm_A,0.0,atol=1E-5,err_msg='Failed subsys_A comparison!')
np.testing.assert_allclose(p_rdm_B-rdm_B,0.0,atol=1E-5,err_msg='Failed subsys_B comparison!')




########## MANY DENSE STATES ###########

state=V
sub_sys_A=[i for i in range(basis.L//2)]

#####

p=basis._p_pure(state,sub_sys_A)
lmbda=ent_entropy({'V_states':state},basis,sub_sys_A,svd_return_vec=[0,1,0])['lmbda']

np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')

#####

p,p_rdm_A=basis._p_pure(state,sub_sys_A,return_rdm='A')
Sent=ent_entropy({'V_states':state},basis,sub_sys_A,DM='chain_subsys',svd_return_vec=[0,1,0])
lmbda=Sent['lmbda']
rdm_A=Sent['DM_chain_subsys']


np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')
np.testing.assert_allclose(p_rdm_A-rdm_A,0.0,atol=1E-5,err_msg='Failed subsys_A comparison!')

#####

p,p_rdm_B=basis._p_pure(state,sub_sys_A,return_rdm='B')
Sent=ent_entropy({'V_states':state},basis,sub_sys_A,DM='other_subsys',svd_return_vec=[0,1,0])
lmbda=Sent['lmbda']
rdm_B=Sent['DM_other_subsys']

np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')
np.testing.assert_allclose(p_rdm_B-rdm_B,0.0,atol=1E-5,err_msg='Failed subsys_B comparison!')

#####

p,p_rdm_A,rdm_B=basis._p_pure(state,sub_sys_A,return_rdm='both')
Sent=ent_entropy({'V_states':state},basis,sub_sys_A,DM='both',svd_return_vec=[0,1,0])
lmbda=Sent['lmbda']
rdm_A=Sent['DM_chain_subsys']
rdm_B=Sent['DM_other_subsys']

np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')
np.testing.assert_allclose(p_rdm_A-rdm_A,0.0,atol=1E-5,err_msg='Failed subsys_A comparison!')
np.testing.assert_allclose(p_rdm_B-rdm_B,0.0,atol=1E-5,err_msg='Failed subsys_B comparison!')

########## SPARSE STATE ###########
system_state=V[:,0]
state=sp.csr_matrix( system_state )#.T

sub_sys_A=[i for i in range(basis.L//2)]

for svds in [1,0]:


	#####
	p=basis._p_pure_sparse(state,sub_sys_A,svds=svds)
	lmbda=ent_entropy(system_state,basis,sub_sys_A,svd_return_vec=[0,1,0])['lmbda']

	# print(p)
	# print(lmbda**2)
	# exit()

	np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')
	
	#####


	p,p_rdm_A=basis._p_pure_sparse(state,sub_sys_A,return_rdm='A',svds=svds)
	Sent=ent_entropy(system_state,basis,sub_sys_A,DM='chain_subsys',svd_return_vec=[0,1,0])
	lmbda=Sent['lmbda']
	rdm_A=Sent['DM_chain_subsys']


	np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')	
	np.testing.assert_allclose(p_rdm_A-rdm_A,0.0,atol=1E-5,err_msg='Failed subsys_A comparison!')


	#####

	ppure,p_rdm_pure_B=basis._p_pure(system_state,sub_sys_A,return_rdm='B')
	p,p_rdm_B=basis._p_pure_sparse(state,sub_sys_A,return_rdm='B',svds=svds)
	Sent=ent_entropy(system_state,basis,sub_sys_A,DM='other_subsys',svd_return_vec=[0,1,0])
	lmbda=Sent['lmbda']
	rdm_B=Sent['DM_other_subsys']

	np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')	
	np.testing.assert_allclose(p_rdm_B-rdm_B,0.0,atol=1E-5,err_msg='Failed subsys_B comparison!')

	
	#####
	


	p,p_rdm_A,p_rdm_B=basis._p_pure_sparse(state,sub_sys_A,return_rdm='both',svds=svds)
	Sent=ent_entropy(system_state,basis,sub_sys_A,DM='both',svd_return_vec=[0,1,0])
	lmbda=Sent['lmbda']
	rdm_A=Sent['DM_chain_subsys']
	rdm_B=Sent['DM_other_subsys']


	np.testing.assert_allclose(p-lmbda**2,0.0,atol=1E-5,err_msg='Failed lmbda^2 comparison!')
	np.testing.assert_allclose(p_rdm_A-rdm_A,0.0,atol=1E-5,err_msg='Failed subsys_A comparison!')
	np.testing.assert_allclose(p_rdm_B-rdm_B,0.0,atol=1E-5,err_msg='Failed subsys_B comparison!')



