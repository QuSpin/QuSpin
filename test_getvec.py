from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import photon_basis,spin_basis_1d
import numpy as np





L=8
Nph = 10
Ntot = 5

J = [[-1.0,i,(i+1)%L] for i in xrange(L)]
K = [[-1.0,i,(i+1)%L,(i+2)%L] for i in xrange(L)]
h = [[-0.5,i] for i in xrange(L)]

basis1 = photon_basis(spin_basis_1d,L,Ntot=Ntot,kblock=0,pblock=1)
basis2 = photon_basis(spin_basis_1d,L,Nph=Nph)

static = [["zz|",J],["+z-|",K],["-z+|",K],["-|+",h],["+|-",h]]
dynamic = []

H1 = hamiltonian(static,dynamic,basis=basis1)
H2 = hamiltonian(static,dynamic,basis=basis2)


V = basis1.get_proj(np.float64,Nph=Nph)
H2 = H2.project_to(V)

print H1-H2


























