import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian
from quspin.tools.measurements import ent_entropy, _reshape_as_subsys
import numpy as np
import scipy.sparse as sp

np.random.seed(12)

L=3

Jxz=[[1.0,i,i+1] for i in range(L-1)]
hx=[[(-1)**i*np.sqrt(2),i] for i in range(L)]

static = [["zx",Jxz],["xz",Jxz],["y",hx]]
H = hamiltonian(static,[],N=L,dtype=np.complex128)
basis =  H.basis

E,V = H.eigh()


"""

########## PURE STATE ###########

state=V[:,0]
sub_sys_A=[i for i in range(basis.L//2)]

#####

sent_b=basis.ent_entropy(state,sub_sys_A)
sent=ent_entropy(state,basis,chain_subsys=sub_sys_A,densities=False)

np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')

#####

sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='A')
sent=ent_entropy(state,basis,chain_subsys=sub_sys_A,DM='chain_subsys',densities=False)

np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')

#####

sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='B')
sent=ent_entropy(state,basis,chain_subsys=sub_sys_A,DM='other_subsys',densities=False)

np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')

#####

sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='both')
sent=ent_entropy(state,basis,chain_subsys=sub_sys_A,DM='both',densities=False)

np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')


########## COLLECTIN OF PURE STATES ###########

sub_sys_A=[i for i in range(basis.L//2)]

for state in [V, V[:,1:6]]:

	#####

	sent_b=basis.ent_entropy(state.T,sub_sys_A,enforce_pure=True)
	sent=ent_entropy({'V_states':state},basis,chain_subsys=sub_sys_A,densities=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')

	#####

	sent_b=basis.ent_entropy(state.T,sub_sys_A,return_rdm='A',enforce_pure=True)
	sent=ent_entropy({'V_states':state},basis,sub_sys_A,DM='chain_subsys',densities=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')

	#####

	sent_b=basis.ent_entropy(state.T,sub_sys_A,return_rdm='B',enforce_pure=True)
	sent=ent_entropy({'V_states':state},basis,chain_subsys=sub_sys_A,DM='other_subsys',densities=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')

	#####

	sent_b=basis.ent_entropy(state.T,sub_sys_A,return_rdm='both',enforce_pure=True)
	sent=ent_entropy({'V_states':state},basis,chain_subsys=sub_sys_A,DM='both',densities=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
	np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')


"""
########## COLLECTIN OF PURE STATES ###########

sub_sys_A=[i for i in range(basis.L//2)]
#sub_sys_A=[1,2]

p_DM=np.random.uniform(size=len(E))
DM = reduce(np.dot,[V,np.diag(p_DM/np.sum(p_DM)),V.T.conj()])	

#####

sent_b=basis.ent_entropy(DM,sub_sys_A)
sent=ent_entropy(DM,basis,sub_sys_A,densities=False)

print(sent_b['Sent'])
print(sent['Sent'])

np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')

#####

sent_b=basis.ent_entropy(DM,sub_sys_A,return_rdm='A')
sent=ent_entropy(DM,basis,chain_subsys=sub_sys_A,DM='chain_subsys',densities=False)

print(sent_b['Sent'])
print(sent['Sent'])

np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')



#####

sent_b=basis.ent_entropy(DM,sub_sys_A,return_rdm='B')
sent=ent_entropy(DM,basis,chain_subsys=sub_sys_A,DM='other_subsys',densities=False)

print(sent_b['Sent'])
print(sent['Sent'])

exit()


np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')

#####

sent_b=basis.ent_entropy(DM,sub_sys_A,return_rdm='both')
sent=ent_entropy(DM,basis,chain_subsys=sub_sys_A,DM='both',densities=False)

print(sent_b['Sent'])
print(sent['Sent'])

exit()
np.testing.assert_allclose(sent['Sent']-sent_b['Sent'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
exit()
np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')



