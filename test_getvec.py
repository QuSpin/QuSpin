from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import photon_basis,spin_basis_1d
import numpy as np





L=6
Nph = 10

J = [[-1.0,i,(i+1)%L] for i in xrange(L)]
h = [[-0.5,i] for i in xrange(L)]

basis1 = photon_basis(spin_basis_1d,L,Nph=Nph,kblock=L/2,pblock=-1)
basis2 = photon_basis(spin_basis_1d,L,Nph=Nph)

static = [["zz|",J],["y|",h],["x|+",h],["x|-",h]]
dynamic = []

H1 = hamiltonian(static,dynamic,basis=basis1)
H2 = hamiltonian(static,dynamic,basis=basis2).todense()


E1,V1 = H1.eigh()

V1_tot = basis1.get_vec(V1,sparse=True).todense()


E2 = np.einsum('ji,jk,ki->i',V1_tot.conj(),H2,V1_tot)

print np.linalg.norm(E1-E2)



























