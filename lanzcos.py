import numpy as np
from quspin.operators import hamiltonian,quantum_LinearOperator
from quspin.basis import spin_basis_1d

def diagonalize(L,J,h):
	basis = spin_basis_1d(L,zblock=1,pauli=True)

	J_list = [[J[i,j],i,j] for i in range(L) for j in range(L) if j>i ]
	h_list = [[h,i] for i in range(L)]

	H = quantum_LinearOperator([["x",h_list],["zz",J_list]],basis=basis,dtype=np.float32)

	return H.eigsh(k=1,which="SA",maxiter=1000)


np.random.seed(0)

it = 1
L=20
J = np.random.choice([-1,1],size=(L,L))
J = np.triu(J,k=1)

E,V = diagonalize(L,J,h=0.4)






