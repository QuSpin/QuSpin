from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import operator,hamiltonian
from quspin.basis import spin_basis_1d
import numpy as np
import scipy.sparse as sp



eps=1e-13
L = 8


def f(t):
	return 1+t


J = [[1.0,i,(i+1)%L] for i in range(L)]

static=[["yy",J],["xx",J]]
dynamic=[["zz",J,f,()],]

op_dict = {
	"Jz":[["zz",J]],
	"Jxx":[["yy",J],["xx",J]],
}

Op = operator(op_dict,N=L,dtype=np.float64)
H = hamiltonian(static,dynamic,N=L,dtype=np.float64)

E ,V  = Op.eigh()
E1,V1 = H.eigh()

assert(np.linalg.norm(E1-E)/(1<<L)<eps)

k=20
E ,V  = Op.eigsh(k=k,which="SA")
E1,V1 = H.eigsh(k=k,which="SA")



assert(np.linalg.norm(E1-E)/k<eps)

V1 = Op.dot(V)
V2 = H.dot(V)
assert(np.linalg.norm(V1-V2)/(1<<L)<eps)

V1 = Op.rdot(V.T)
V2 = H.rdot(V.T)
assert(np.linalg.norm(V1-V2)/(1<<L)<eps)

V1 = Op.matrix_ele(V,V)
V2 = H.matrix_ele(V,V)
assert(np.linalg.norm(V1-V2)/(1<<L)<eps)

H2 = Op.todense()
H3 = H.todense()
assert(np.linalg.norm(H2-H3)<eps)

H2 = Op.toarray()
H3 = H.toarray()
assert(np.linalg.norm(H2-H3)<eps)

H2 = Op.tocsr()
H3 = H.tocsr()
assert(np.linalg.norm((H2-H3).todense())<eps)

H2 = Op.tocsc()
H3 = H.tocsc()
assert(np.linalg.norm((H2-H3).todense())<eps)

H1 = Op.tohamiltonian(Jz=(f,()),Jxx=1)
H2 = H - H1

for t in np.linspace(-10,10,num=1000):
	assert(np.linalg.norm(H2.todense(t))<eps)
