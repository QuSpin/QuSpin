import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian
from quspin.tools.measurements import _ent_entropy, _reshape_as_subsys
import numpy as np
import scipy.sparse as sp

np.random.seed(12)
np.set_printoptions(linewidth=1000000,precision=3)
L=4

Jxz=[[1.0,i,i+1] for i in range(L-1)]
hx=[[(-1)**i*np.sqrt(2),i] for i in range(L)]

static = [["zx",Jxz],["xz",Jxz],["y",hx]]
H = hamiltonian(static,[],N=L,dtype=np.complex128)
basis =  H.basis

E,V = H.eigh()

sub_sys_A=[i for i in range(basis.L//2)]

########## PURE STATE ###########

for state, state_sp in zip([V[:,0], V], [sp.csr_matrix(V[:,0]).T, sp.csr_matrix(V)]):
	#####

	sent=basis.ent_entropy(state,sub_sys_A,enforce_pure=True)
	sent_sp=basis.ent_entropy(state_sp,sub_sys_A,enforce_pure=True)

	np.testing.assert_allclose(sent['Sent_A']-sent_sp['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')

	#####

	sent=basis.ent_entropy(state,sub_sys_A,return_rdm='A',enforce_pure=True)
	sent_sp=basis.ent_entropy(state_sp,sub_sys_A,return_rdm='A',enforce_pure=True)

	seq=list(rdm.toarray() for rdm in np.atleast_1d(sent_sp["rdm_A"]))
	rdm_A_sp = np.dstack(seq).transpose((2,0,1)).squeeze()

	np.testing.assert_allclose(sent['Sent_A']-sent_sp['Sent_A'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['rdm_A']-rdm_A_sp,0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')

	#####

	sent=basis.ent_entropy(state,sub_sys_A,return_rdm='B',enforce_pure=True)
	sent_sp=basis.ent_entropy(state_sp,sub_sys_A,return_rdm='B',enforce_pure=True)

	seq=list(rdm.toarray() for rdm in np.atleast_1d(sent_sp["rdm_B"]))
	rdm_B_sp = np.dstack(seq).transpose((2,0,1)).squeeze()

	np.testing.assert_allclose(sent['Sent_B']-sent_sp['Sent_B'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['rdm_B']-rdm_B_sp,0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
	
	#####

	sent=basis.ent_entropy(state,sub_sys_A,return_rdm='both',enforce_pure=True)
	sent_sp=basis.ent_entropy(state_sp,sub_sys_A,return_rdm='both',enforce_pure=True)

	seq=list(rdm.toarray() for rdm in np.atleast_1d(sent_sp["rdm_A"]))
	rdm_A_sp = np.dstack(seq).transpose((2,0,1)).squeeze()

	seq=list(rdm.toarray() for rdm in np.atleast_1d(sent_sp["rdm_B"]))
	rdm_B_sp = np.dstack(seq).transpose((2,0,1)).squeeze()

	np.testing.assert_allclose(sent['Sent_B']-sent_sp['Sent_B'],0.0,atol=1E-5,err_msg='Failed Sent comparison!')
	np.testing.assert_allclose(sent['rdm_B']-rdm_B_sp,0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
	np.testing.assert_allclose(sent['rdm_A']-rdm_A_sp,0.0,atol=1E-5,err_msg='Failed rdm_A comparison!')
	
