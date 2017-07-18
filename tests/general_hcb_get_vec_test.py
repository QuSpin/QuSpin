from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_1d
from quspin.basis import general_hcb
import numpy as np
from itertools import product

def check_gen_basis_hcb():
	L=8
	kblocks = [None]
	kblocks.extend(range(L))
	pblocks = [None,0,1]
	zblocks = [0,1]

	Nups = [None,L//2]
	
	t = np.array([(i+1)%L for i in range(L)])
	p = np.array([L-i-1 for i in range(L)])
	z = np.array([-(i+1) for i in range(L)])

	for Nup,kblock,pblock,zblock in product(Nups,kblocks,pblocks,zblocks):
		gen_blocks = {}
		basis_blocks = {"pauli":False}
		dtype=np.complex128
		if kblock==0 or kblock==L//2:
			if pblock is not None:
				dtype=np.float64
				basis_blocks["pblock"] = (-1)**pblock
				gen_blocks["pblock"] = (p,pblock)
			else:
				basis_blocks["pblock"] = None
				gen_blocks["pblock"] = None
		else:
			basis_blocks["pblock"] = None
			gen_blocks["pblock"] = None

		if zblock is not None:
			basis_blocks["zblock"] = (-1)**zblock
			gen_blocks["zblock"] = (z,zblock)
		else:
			basis_blocks["zblock"] = None
			gen_blocks["zblock"] = None

		if kblock is not None:
			basis_blocks["kblock"] = kblock
			gen_blocks["kblock"] = (t,kblock)
		else:
			basis_blocks["kblock"] = None
			gen_blocks["kblock"] = None

		print("checking Nup={Nup:6} kblock={kblock:6} pblock={pblock:6} zblock={zblock:6}".format(Nup=Nup,**basis_blocks))

		basis_1d = spin_basis_1d(L,Nup=Nup,**basis_blocks)
		gen_basis = general_hcb(L,Np=Nup,**gen_blocks)


		P1 = basis_1d.get_proj(dtype)
		P2 = gen_basis.get_proj(dtype)

		np.testing.assert_allclose((P1-P2).data,0,atol=1e-14,err_msg="failed projector")

		v = np.random.ranf(size=(basis_1d.Ns,)).astype(dtype)
		vs = np.random.ranf(size=(basis_1d.Ns,100)).astype(dtype)

		v1 = basis_1d.get_vec(v,sparse=False)
		v2 = gen_basis.get_vec(v,sparse=False)

		np.testing.assert_allclose((v1-v2),0,atol=1e-14,err_msg="failed single vector dense")


		v1 = basis_1d.get_vec(v,sparse=True)
		v2 = gen_basis.get_vec(v,sparse=True)

		np.testing.assert_allclose((v1-v2).data,0,atol=1e-14,err_msg="failed single vector sparse")

		vs1 = basis_1d.get_vec(vs,sparse=False)
		vs2 = gen_basis.get_vec(vs,sparse=False)

		np.testing.assert_allclose((vs1-vs2),0,atol=1e-14,err_msg="failed multi vector dense")

		vs1 = basis_1d.get_vec(vs,sparse=True)
		vs2 = gen_basis.get_vec(vs,sparse=True)

		np.testing.assert_allclose((vs1-vs2).data,0,atol=1e-14,err_msg="failed multi vector sparse")


check_gen_basis_hcb()