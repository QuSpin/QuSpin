from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian,quantum_operator
import numpy as np
import scipy.sparse as sp



L = 10


for Nup in [0,1,2]:
	t = (np.arange(L)+1)%L
	basis = spin_basis_general(L,Nup=Nup,t=(t,1))


	J_list = [[1.0,i,(i+1)%L] for i in range(L)]
	static = [[op,J_list] for op in ["xx","yy","zz"]]


	H = hamiltonian(static,[],basis=basis,dtype=np.complex128)
	O = quantum_operator(dict(J=static),basis=basis,dtype=np.complex128)
	ns = 10


	pure_sp = sp.random(H.Ns,1,format="csr")
	pure_sp_many = sp.random(H.Ns,ns,format="csr")
	mixed_sp = sp.random(H.Ns,H.Ns,format="csr")

	pure = np.random.normal(0,1,size=(H.Ns,))
	pure_many = np.random.normal(0,1,size=(H.Ns,ns))
	pure_sq = np.random.normal(0,1,size=(H.Ns,H.Ns))

	mixed = np.random.normal(0,1,size=(H.Ns,H.Ns))
	mixed_many = np.random.normal(0,1,size=(H.Ns,H.Ns,ns))

	times = np.linspace(0,1,ns)


	vl_iter = [pure_many,pure_many,pure_sp_many,pure_sp_many]
	vr_iter = [pure_many,pure_sp_many,pure_many,pure_sp_many]

	for vl,vr in zip(vl_iter,vr_iter):
		r = H.matrix_ele(vl,vr)
		assert(r.ndim==2 and r.shape==(ns,ns))

		r = H.matrix_ele(vl,vr,diagonal=True)
		assert(r.ndim==1 and r.shape==(ns,))

		r = O.matrix_ele(vl,vr)
		assert(r.ndim==2 and r.shape==(ns,ns))

		r = O.matrix_ele(vl,vr,diagonal=True)
		assert(r.ndim==1 and r.shape==(ns,))



	for func in [H.expt_value,H.quant_fluct,O.expt_value,O.quant_fluct]:
		r = func(pure_sp)
		assert(r.ndim==0)

		r = func(pure)
		assert(r.ndim==0)

		r = func(mixed)
		assert(r.ndim==0)

		r = func(pure_sp_many)
		assert(r.ndim==1 and r.shape[0] == ns)

		r = func(pure_many)
		assert(r.ndim==1 and r.shape[0] == ns)

		r = func(pure_sq,enforce_pure=True)
		assert(r.ndim==1 and r.shape[0] == H.Ns)

		r = func(mixed_many)
		assert(r.ndim==1 and r.shape[0] == ns)

	for func in [H.expt_value,H.quant_fluct]:
		r = func(pure_sp_many,time=times)
		assert(r.ndim==1 and r.shape[0] == ns)

		r = func(pure_many,time=times)
		assert(r.ndim==1 and r.shape[0] == ns)

		r = func(mixed_many,time=times)
		assert(r.ndim==1 and r.shape[0] == ns)
