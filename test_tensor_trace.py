from quspin.basis import spin_basis_1d,tensor_basis
import numpy as np
import scipy.sparse as sp



b1 = spin_basis_1d(1)
b2 = spin_basis_1d(3)
bt = tensor_basis(b1,b2)



state = np.array([0,1,1,0,0,0,0,0],dtype=np.float64)
state /= np.linalg.norm(state)
state = sp.csr_matrix(state.reshape((-1,1)))

Sent,rdm = b2.ent_entropy(state,sub_sys_A=[0,2],return_rdm="A")
print(Sent)
print()
print(rdm)


"""
state = np.array([1,0,0,1,0,1,0,1],dtype=np.float64)
state /= np.linalg.norm(state)
state = sp.csr_matrix(state.reshape((-1,1)))



rdm = bt.partial_trace(state,sub_sys_A="right")
Sent,dm1,dm2 = bt.ent_entropy(state,alpha=1.0,return_rdm="both")


print(Sent)
print()
print(dm1)
print()
print(dm2)
"""