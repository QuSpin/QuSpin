from __future__ import print_function,division
import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.tools.block_tools import block_ops,block_diag_hamiltonian
from quspin.operators import hamiltonian,exp_op
from quspin.basis import spin_basis_1d
from itertools import izip
import numpy as np

np.set_printoptions(linewidth=100000,precision=2)
L=4

basis = spin_basis_1d(L,kblock=1)

P = basis.get_proj(np.complex128)


J = [[1.0,i,(i+1)%L] for i in range(L)]
h = [[1.0,i] for i in range(L)]


static = [["xx",J],["yy",J],["zz",J]]
dynamic = [["z",h,lambda t:np.sin(t),()]]


blocks = []

for Nup in range(L+1):
	blocks.append({"Nup":Nup})


H_block = block_diag_hamiltonian(blocks,static,[],spin_basis_1d,(L,),np.float64)
H = hamiltonian(static,[],N=L)

for t in np.linspace(0,2*np.pi):
	E = H.eigvalsh(time=t)
	E_blocks = H.eigvalsh(time=t)
	np.testing.assert_allclose(E,E_blocks)


static = [["zz",J]]
dynamic = [["x",h,lambda t:t,[]]]

blocks = []
for kblock in range(L):
	blocks.append({"kblock":kblock})

H = hamiltonian(static,dynamic,N=L)
block_op = block_ops(blocks,static,dynamic,spin_basis_1d,(L,),np.complex128,compute_all_blocks=True)

expH = exp_op(H,a=-1j,start=0,stop=10,iterate=True,num=50,endpoint=True)

times = np.linspace(0,10,num=50,endpoint=True)

psi0 = np.random.ranf(H.Ns)
psi0 /= np.linalg.norm(psi0)

psi_exact_1 = H.evolve(psi0,0,times,iterate=True,atol=1e-15,rtol=1e-15)
psi_block_1 = block_op.evolve(psi0,0,times,iterate=True,atol=1e-15,rtol=1e-15)

psi_exact_2 = expH.dot(psi0,time=0.3)
psi_block_2 = block_op.expm(psi0,H_time_eval=0.3,start=0,stop=10,iterate=True,num=50,endpoint=True)

for psi_e_1,psi_e_2,psi_b_1,psi_b_2 in izip(psi_exact_1,psi_exact_2,psi_block_1,psi_block_2):
	np.testing.assert_allclose(psi_b_1,psi_e_1,atol=1e-7)
	np.testing.assert_allclose(psi_b_2,psi_e_2,atol=1e-7)



# same for iterate=False
expH = exp_op(H,a=-1j,start=0,stop=10,iterate=False,num=50,endpoint=True)

times = np.linspace(0,10,num=50,endpoint=True)

psi0 = np.random.ranf(H.Ns)
psi0 /= np.linalg.norm(psi0)

psi_exact_1 = H.evolve(psi0,0,times,iterate=False,atol=1e-15,rtol=1e-15)
psi_block_1 = block_op.evolve(psi0,0,times,iterate=False,atol=1e-15,rtol=1e-15)

psi_exact_2 = expH.dot(psi0,time=0.3)
psi_block_2 = block_op.expm(psi0,H_time_eval=0.3,start=0,stop=10,iterate=False,num=50,endpoint=True)

for psi_e_1,psi_e_2,psi_b_1,psi_b_2 in izip(psi_exact_1,psi_exact_2,psi_block_1,psi_block_2):
	np.testing.assert_allclose(psi_b_1,psi_e_1,atol=1e-7)
	np.testing.assert_allclose(psi_b_2,psi_e_2,atol=1e-7)



