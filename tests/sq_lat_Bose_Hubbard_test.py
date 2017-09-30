from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import boson_basis_general
from quspin.basis.transformations import square_lattice_trans
from quspin.operators import hamiltonian
import numpy as np




def test(sps):
	Lx = 3
	Ly = 3

	N = Lx*Ly
	nmax = sps-1
	tr = square_lattice_trans(Lx,Ly)

	basis_dict = {}
	Nb_list=range(nmax*N+1)

	for Nb in Nb_list:
		basis_dict[(Nb,None,None)] = boson_basis_general(N,Nb=Nb,sps=sps)
		for kx in range(Lx):
			for ky in range(Ly):
				basis_dict[(Nb,kx,ky)] = boson_basis_general(N,Nb=Nb,sps=sps,kxblock=(tr.T_x,kx),kyblock=(tr.T_y,ky))
				# print basis_dict[(Nb,kx,ky)]._n

	J = [[1.0,i,tr.T_x[i]] for i in range(N)]
	J.extend([[1.0,i,tr.T_y[i]] for i in range(N)])

	static = [["nn",J],["+-",J],["-+",J]]

	E_symm = {}

	for key,basis in basis_dict.items():
		H = hamiltonian(static,[],basis=basis,dtype=np.complex128)
		E_symm[key] = H.eigvalsh()

	for Nb in Nb_list:
		E_full_block = E_symm[(Nb,None,None)]
		E_block = []
		for kx in range(Lx):
			for ky in range(Ly):
				E_block.append(E_symm[(Nb,kx,ky)])

		E_block = np.hstack(E_block)
		E_block.sort()
		np.testing.assert_allclose(E_full_block,E_block,atol=1e-13)
		print("passed Nb={} sector".format(Nb))


test(2)
test(3)
