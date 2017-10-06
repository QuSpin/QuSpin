import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np
from itertools import product
try:
	from functools import reduce
except ImportError:
	pass



dtypes = [(np.float32,np.complex64),(np.float64,np.complex128)]


spin_ops={}
spins=['1/2','1','3/2','2']

spin_ops['1/2']={}
spin_ops['1/2']["I"]=np.array([[1,0],[0,1]]) + 0.0j
spin_ops['1/2']['x']=(1.0/2.0)*np.array([[0,1],[1,0]]) + 0.0j
spin_ops['1/2']['y']=(1.0j/2.0)*np.array([[0,-1],[1,0]]) + 0.0j
spin_ops['1/2']['z']=(1.0/2.0)*np.array([[1,0.0],[0.0,-1]]) + 0.0j

spin_ops['1']={}
spin_ops['1']['I']=np.array([[1,0,0],[0,1,0],[0,0,1]]) + 0.0j
spin_ops['1']['x']=(1.0/np.sqrt(2))*np.array([[0,1,0],[1,0,1],[0,1,0]]) + 0.0j
spin_ops['1']['y']=(1.0j/np.sqrt(2))*np.array([[0,-1,0],[1,0,-1],[0,1,0]]) +0.0j
spin_ops['1']['z']=np.array([[1,0,0],[0,0,0],[0,0,-1]]) + 0.0j

spin_ops['3/2']={}
spin_ops['3/2']['I']=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
spin_ops['3/2']['x']=(1.0/2.0)*np.array([[0,np.sqrt(3),0,0],[np.sqrt(3),0,2,0],[0,2,0,np.sqrt(3)],[0,0,np.sqrt(3),0]]) + 0j
spin_ops['3/2']['y']=(1.0j/2.0)*np.array([[0,-np.sqrt(3),0,0],[np.sqrt(3),0,-2,0],[0,2,0,-np.sqrt(3)],[0,0,np.sqrt(3),0]])
spin_ops['3/2']['z']=(1.0/2.0)*np.array([[3,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-3]])+ 0.0j

spin_ops['2']={}
spin_ops['2']['I']=np.array([[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
spin_ops['2']['x']=(1.0/2.0)*np.array([[0,2.0,0,0,0],[2,0,np.sqrt(6),0,0],[0,np.sqrt(6),0,np.sqrt(6),0],[0,0,np.sqrt(6),0,2],[0,0,0,2,0]])
spin_ops['2']['y']=(1.0j/2.0)*np.array([[0,-2.0,0,0,0],[2,0,-np.sqrt(6),0,0],[0,np.sqrt(6),0,-np.sqrt(6),0],[0,0,np.sqrt(6),0,-2],[0,0,0,2,0]])
spin_ops['2']['z']=np.array([[2,0,0,0,0],[0,1,0,0,0],[0,0,0,0,0],[0,0,0,-1,0],[0,0,0,0,-2]])+ 0.0j




L_max = 4


for S in spins:
	for L in range(1,L_max+1):
		basis = spin_basis_1d(L,S=S,pauli=False)
		J = [1.0]
		J.extend(range(L))
		for p in product(*[spin_ops[S].items() for i in range(L)]):
			opstr,ops = zip(*list(p))
			opstr = "".join(opstr)
			static = [[opstr,[J]]]
			static,_ = basis.expanded_form(static,[])
			quspin_op = hamiltonian(static,[],basis=basis,check_symm=False,check_herm=False)
			op = reduce(np.kron,ops)
			np.testing.assert_allclose(quspin_op.toarray(),op,atol=1e-14,err_msg="failed test for S={} operator {}".format(S,opstr))

	print("spin-{} operators comparisons passed!".format(S))
