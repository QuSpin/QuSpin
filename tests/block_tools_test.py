from __future__ import print_function,division
import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.tools.block_tools import block_ops,block_diag_hamiltonian
from quspin.operators import hamiltonian,exp_op
from quspin.basis import spin_basis_1d
import numpy as np

try:
	from itertools import izip
except ImportError:
	izip = zip

np.set_printoptions(linewidth=100000,precision=2)

def f(t):
	return np.sin(t)*t

def test():
	L=5

	start = 0
	stop  = 10
	num   = 10


	J = [[1.0,i,(i+1)%L] for i in range(L)]
	h = [[1.0,i] for i in range(L)]


	static = [["xx",J],["yy",J],["zz",J]]
	dynamic = [["z",h,np.sin,()]]


	blocks = []

	for Nup in range(L+1):
		blocks.append({"Nup":Nup})


	H_block = block_diag_hamiltonian(blocks,static,dynamic,spin_basis_1d,(L,),np.float64)
	H = hamiltonian(static,dynamic,N=L)

	for t in np.linspace(0,2*np.pi):
		E = H.eigvalsh(time=t)
		E_blocks = H.eigvalsh(time=t)
		np.testing.assert_allclose(E,E_blocks)

	static = [["zz",J]]
	dynamic = [["x",h,f,[]]]

	blocks = []
	for kblock in range(L):
		blocks.append({"kblock":kblock})

	H = hamiltonian(static,dynamic,N=L)

	[E0] = H.eigsh(time=0.3,k=1,which='SA',return_eigenvectors=False)

	block_op = block_ops(blocks,static,dynamic,spin_basis_1d,(L,),np.complex128,compute_all_blocks=True)



	# real time.
	expH = exp_op(H,a=-1j,start=start,stop=stop,iterate=True,num=num,endpoint=True)

	times = np.linspace(start,stop,num=num,endpoint=True)

	psi0 = np.random.ranf(H.Ns)
	psi0 /= np.linalg.norm(psi0)


	psi_exact_1 = H.evolve(psi0,0,times,iterate=True,atol=1e-15,rtol=1e-15)
	psi_block_1 = block_op.evolve(psi0,0,times,iterate=True,atol=1e-15,rtol=1e-15,n_jobs=4)

	psi_exact_2 = expH.dot(psi0,time=0.3)
	psi_block_2 = block_op.expm(psi0,H_time_eval=0.3,start=start,stop=stop,iterate=True,num=num,endpoint=True,n_jobs=2,block_diag=True)




	for psi_e_1,psi_e_2,psi_b_1,psi_b_2 in izip(psi_exact_1,psi_exact_2,psi_block_1,psi_block_2):
		np.testing.assert_allclose(psi_b_1,psi_e_1,atol=1e-7)
		np.testing.assert_allclose(psi_b_2,psi_e_2,atol=1e-7)



	expH.set_iterate(False)

	psi_exact_1 = H.evolve(psi0,0,times,iterate=False,atol=1e-15,rtol=1e-15)
	psi_block_1 = block_op.evolve(psi0,0,times,iterate=False,atol=1e-15,rtol=1e-15,n_jobs=4)

	psi_exact_2 = expH.dot(psi0,time=0.3)
	psi_block_2 = block_op.expm(psi0,H_time_eval=0.3,start=start,stop=stop,iterate=False,num=num,endpoint=True,block_diag=True)




	for psi_e_1,psi_b_1 in izip(psi_exact_1,psi_block_1):
		np.testing.assert_allclose(psi_b_1,psi_e_1,atol=1e-7)

	for psi_e_2,psi_b_2 in izip(psi_exact_2,psi_block_2):
		np.testing.assert_allclose(psi_b_2,psi_e_2,atol=1e-7)





	# imaginary time
	expH = exp_op(H,a=-1,start=start,stop=stop,iterate=True,num=num,endpoint=True)

	times = np.linspace(start,stop,num=num,endpoint=True)

	psi0 = np.random.ranf(H.Ns)
	psi0 /= np.linalg.norm(psi0)


	psi_exact_2 = expH.dot(psi0,time=0.3,shift=-E0)
	psi_block_2 = block_op.expm(psi0,a=-1,shift=-E0,H_time_eval=0.3,start=start,stop=stop,iterate=True,num=num,endpoint=True,block_diag=True)



	for psi_e_2,psi_b_2 in izip(psi_exact_2,psi_block_2):
		np.testing.assert_allclose(psi_b_2,psi_e_2,atol=1e-7)

	# same for iterate=False
	expH.set_iterate(False)

	psi_exact_2 = expH.dot(psi0,time=0.3,shift=-E0)
	psi_block_2 = block_op.expm(psi0,a=-1,shift=-E0,H_time_eval=0.3,start=start,stop=stop,iterate=False,num=num,endpoint=True,block_diag=True)



	for psi_e_2,psi_b_2 in izip(psi_exact_2,psi_block_2):
		np.testing.assert_allclose(psi_b_2,psi_e_2,atol=1e-7)

if __name__ == '__main__':
	test()

