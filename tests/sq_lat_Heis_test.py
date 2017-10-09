from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_general
from quspin.basis.transformations import square_lattice_trans
from quspin.operators import hamiltonian
import numpy as np
from itertools import product
import os


os.environ["OMP_NUM_THREADS"]="1"

def test(S,Lx,Ly):

	N = Lx*Ly

	nmax = int(eval("2*"+S))
	sps = nmax+1
	tr = square_lattice_trans(Lx,Ly)


	basis_dict = {}
	Nups=range(nmax*N+1)

	for Nup in Nups:
		basis_blocks=[]
		pcon_basis = spin_basis_general(N,Nup=Nup,S=S)
		Ns_block = 0
		for blocks in tr.allowed_blocks_spin_inversion_iter(Nup,sps):
			basis =  spin_basis_general(N,Nup=Nup,S=S,**blocks)
			Ns_block += basis.Ns
			basis_blocks.append(basis)

		try:
			assert(Ns_block == pcon_basis.Ns)
		except AssertionError:
			print(Nup,Ns_block,pcon_basis.Ns)
			raise AssertionError("reduced blocks don't sum to particle sector.")


		basis_dict[Nup] = (pcon_basis,basis_blocks)
	
	J = [[1.0,i,tr.T_x[i]] for i in range(N)]
	J.extend([[1.0,i,tr.T_y[i]] for i in range(N)])

	static = [["zz",J],["+-",J],["-+",J]]

	E_symm = {}

	for Nb,(pcon_basis,basis_blocks) in basis_dict.items():
		H_pcon = hamiltonian(static,[],basis=pcon_basis,dtype=np.float64)
		if H_pcon.Ns>0:
			E_pcon = np.linalg.eigvalsh(H_pcon.todense())
		else:
			E_pcon = np.array([])

		E_block = []
		for basis in basis_blocks:
			H = hamiltonian(static,[],basis=basis,dtype=np.complex128)
			if H.Ns>0:
				E_block.append(np.linalg.eigvalsh(H.todense()))

		E_block = np.hstack(E_block)
		E_block.sort()
		np.testing.assert_allclose(E_pcon,E_block,atol=1e-13)
		print("passed Nb={} sector".format(Nb))

test("1/2",3,3)
test("1",3,3)
test("1/2",3,2)
test("1",3,2)

