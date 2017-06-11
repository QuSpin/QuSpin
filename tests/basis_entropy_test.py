from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian
from quspin.tools.measurements import _ent_entropy, _reshape_as_subsys
import numpy as np
import scipy.sparse as sp

from functools import reduce


np.random.seed(0)

L=4

Jxz=[[1.0,i,i+1] for i in range(L-1)]
hx=[[(-1)**i*np.sqrt(2),i] for i in range(L)]

static = [["zx",Jxz],["xz",Jxz],["y",hx]]
H = hamiltonian(static,[],N=L,dtype=np.complex128)
basis =  H.basis

E,V = H.eigh()




########## PURE STATE ###########

state=V[:,0]
sub_sys_A=[i for i in range(basis.L//2)]

#####

sent_b=basis.ent_entropy(state,sub_sys_A,density=False)
sent=_ent_entropy(state,basis,chain_subsys=sub_sys_A,density=False)
np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')

#####

sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='A',density=False)
sent=_ent_entropy(state,basis,chain_subsys=sub_sys_A,DM='chain_subsys',density=False)

np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')

#####

sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='B',density=False)
sent=_ent_entropy(state,basis,chain_subsys=sub_sys_A,DM='other_subsys',density=False)

np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')



#####

sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='both',density=False)
sent=_ent_entropy(state,basis,chain_subsys=sub_sys_A,DM='both',density=False)

np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')


########## COLLECTION OF PURE STATES ###########

sub_sys_A=[i for i in range(basis.L//2)]

for state,enforce_pure in zip([V, V[:,1:6]],[True,False]):

	#####
	sent_b=basis.ent_entropy(state,sub_sys_A,enforce_pure=enforce_pure,density=False)
 
	sent=_ent_entropy({'V_states':state},basis,chain_subsys=sub_sys_A,density=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')

	#####

	sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='A',enforce_pure=enforce_pure,density=False)
	sent=_ent_entropy({'V_states':state},basis,sub_sys_A,DM='chain_subsys',density=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')

	#####

	sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='B',enforce_pure=enforce_pure,density=False)
	sent=_ent_entropy({'V_states':state},basis,chain_subsys=sub_sys_A,DM='other_subsys',density=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')

	#####

	sent_b=basis.ent_entropy(state,sub_sys_A,return_rdm='both',enforce_pure=enforce_pure,density=False)
	sent=_ent_entropy({'V_states':state},basis,chain_subsys=sub_sys_A,DM='both',density=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
	np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')




########## MiXED PURE STATE ###########

sub_sys_A=[i for i in range(basis.L//2)]
#sub_sys_A=[1,2]

p_DM=np.random.uniform(size=len(E)) #np.ones(len(E))#
DM1 = reduce(np.dot,[V,np.diag(p_DM/sum(p_DM)),V.T.conj()])
p_DM2=np.random.uniform(size=len(E)) #np.ones(len(E))#
DM2 = reduce(np.dot,[V,np.diag(p_DM2/sum(p_DM2)),V.T.conj()])	
DM3=np.outer(V[:,0],V[:,0].conj())
#DM=V


#####

for DM in [DM1,DM2,DM3]:

	sent_b=basis.ent_entropy(DM,sub_sys_A,density=False)
	sent=_ent_entropy(DM.squeeze(),basis,sub_sys_A,density=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')

	#####

	sent_b=basis.ent_entropy(DM,sub_sys_A,return_rdm='both',density=False)
	sent=_ent_entropy(DM.squeeze(),basis,chain_subsys=sub_sys_A,DM='both',density=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
	np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')

	#####

	sent_b=basis.ent_entropy(DM,sub_sys_A,return_rdm='A',density=False)
	sent=_ent_entropy(DM.squeeze(),basis,chain_subsys=sub_sys_A,DM='chain_subsys',density=False)

	np.testing.assert_allclose(sent['Sent']-sent_b['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['DM_chain_subsys']-sent_b['rdm_A'],0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')

	#####

	sent_b=basis.ent_entropy(DM,sub_sys_A,return_rdm='B',density=False)
	sent=_ent_entropy(DM.squeeze(),basis,chain_subsys=sub_sys_A,DM='other_subsys',density=False)
	
	np.testing.assert_allclose(sent['Sent']-sent_b['Sent_B'],0.0,atol=1E-5,err_msg='Failed Sent comparison!') 
	np.testing.assert_allclose(sent['DM_other_subsys']-sent_b['rdm_B'],0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')


########## MANY MiXED PURE STATES ###########

p_DM=np.random.uniform(size=len(E)) #np.ones(len(E))#
DM = reduce(np.dot,[V,np.diag(p_DM/sum(p_DM)),V.T.conj()])
DMs = np.dstack([DM,DM,DM])

sent_b=basis.ent_entropy(DMs,sub_sys_A)

np.testing.assert_allclose(np.diff(sent_b['Sent_A']),0.0,atol=1E-5,err_msg='Failed Sent comparison!')
#####

sent_b=basis.ent_entropy(DMs,sub_sys_A,return_rdm='both',density=False)


np.testing.assert_allclose( np.diff( sent_b['Sent_A'] ),0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose( np.diff( sent_b['rdm_A'], axis=0 ),0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
np.testing.assert_allclose( np.diff( sent_b['rdm_B'], axis=0 ),0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')

#####

sent_b=basis.ent_entropy(DMs,sub_sys_A,return_rdm='A',density=False)

np.testing.assert_allclose( np.diff( sent_b['Sent_A'] ),0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose( np.diff( sent_b['rdm_A'], axis=0 ),0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')

#####

sent_b=basis.ent_entropy(DMs,sub_sys_A,return_rdm='B',density=False)

np.testing.assert_allclose( np.diff( sent_b['Sent_B'] ),0.0,atol=1E-5,err_msg='Failed Sent comparison!')
np.testing.assert_allclose( np.diff( sent_b['rdm_B'], axis=0 ),0.0,atol=1E-5,err_msg='Failed rdm_B comparison!')







