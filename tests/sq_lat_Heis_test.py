from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis.general_basis.transformations import square_lattice_trans
from quspin.basis import general_hcb,spin_basis_1d
from quspin.operators import hamiltonian
import numpy as np

Lx=4
Ly=4
N = Lx*Ly
tr = square_lattice_trans(Lx,Ly)

basis = general_hcb(Lx*Ly,kx=(tr.T_x,0),ky=(tr.T_y,0),pd=(tr.P_d,0),py=(tr.P_y,0),px=(tr.P_x,0))
basis_full = spin_basis_1d(N,pauli=False)

J = [[1.0,i,tr.T_x[i]] for i in range(N)]
J.extend([1.0,i,tr.T_y[i]] for i in range(N))

static = [["xx",J],["yy",J],["zz",J]]

H = hamiltonian(static,[],basis=basis,dtype=np.float64)
H_full = hamiltonian(static,[],basis=basis_full,dtype=np.float64)
[E2],_= H.eigsh(k=1,which="SA")
[E1],_= H_full.eigsh(k=1,which="SA")

assert(np.abs(E2-E1)<1.1e-14)
