from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
from quspin.tools.evolution import expm_multiply_parallel
import numpy as np
import matplotlib.pyplot as plt



def corr_nosymm(L,times,S="1/2"):
	J_list = [[1.0,i,(i+1)%L] for i in range(L)]
	static = [[op,J_list] for op in ["-+","+-","zz"]]
	basis = spin_basis_general(L,S=S,m=0,pauli=False)
	kwargs = dict(basis=basis,dtype=np.float64,
		check_symm=False,check_herm=False,check_pcon=False)
	H = hamiltonian(static,[],**kwargs)

	E,V = H.eigsh(k=1,which="SA")
	psi0 = V[:,0]
	psi0_t = H.evolve(psi0,0,times)
	sqs = []
	

	op_list = [["z",[0],2.0]]

	psi1 = basis.inplace_Op(psi0,op_list,np.float64)
	psi1_t = H.evolve(psi1,0,times)

	psi2_t = basis.inplace_Op(psi0_t,op_list,np.float64)
	sqs.append(np.einsum("ij,ij->j",psi2_t.conj(),psi1_t))

	return sum(sqs)

def corr_symm(L,times,S="1/2"):
	J_list = [[1.0,i,(i+1)%L] for i in range(L)]
	static = [[op,J_list] for op in ["-+","+-","zz"]]

	if (L//2)%2:
		q0 = L//2
		dtype = np.complex128
	else:
		q0 = 0
		dtype = np.float64

	t = (np.arange(L)+1)%L
	basis = spin_basis_general(L,S=S,m=0,kblock=(t,q0),pauli=False)
	kwargs = dict(basis=basis,dtype=dtype,
		check_symm=False,check_herm=False,check_pcon=False)
	H = hamiltonian(static,[],**kwargs)
	E,V = H.eigsh(k=1,which="SA")
	psi0 = V[:,0]
	sqs = []

	psi0_t = H.evolve(psi0.ravel(),0,times)

	for q in range(L):

		op_pq = [["z",[i],(2.0/L)*np.exp(-2j*np.pi*q*i/L)] for i in range(L)]

		basis_q = spin_basis_general(L,S=S,m=0,kblock=(t,q0+q),pauli=False)

		kwargs = dict(basis=basis_q,dtype=np.complex128,
			check_symm=False,check_herm=False,check_pcon=False)

		Hq = hamiltonian(static,[],**kwargs)

		psi1 = basis_q.Op_shift_sector(basis,op_pq,psi0)

		psi1_t = Hq.evolve(psi1,0,times)
		psi2_t = basis_q.Op_shift_sector(basis,op_pq,psi0_t)
		sqs.append(np.einsum("ij,ij->j",psi2_t.conj(),psi1_t))

	return sum(sqs)

times = np.linspace(0,5,101)
for L in [6,8,10,12]:
	print("testing Op_shift_sector for L={}".format(L))
	c_full = corr_nosymm(L,times)
	c_symm = corr_symm(L,times)
	np.testing.assert_allclose(c_full,c_symm,atol=1e-10,rtol=0)




