from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian # Hamiltonian and observables
from quspin.tools.misc import project_op, KL_div, mean_level_spacing
import numpy as np
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers
seed()


"""
This test checks the functions 'obs_vs_time', 'KZ_div' and 'mean_level_spacing'.
"""

dtypes={"float32":np.float32,
		"float64":np.float64,
		"complex64":np.complex64,
		"complex128":np.complex128,
	}

atols={"float32":1E-4,"float64":1E-13,
		"complex64":1E-4,"complex128":1E-13}


L=16
basis = spin_basis_1d(L)
basis2 = spin_basis_1d(L,Nup=L//2,kblock=0,pblock=1,zblock=1)


J_zz = [[1.0,i,(i+1)%L] for i in range(0,L)] 
J_xy = [[0.5,i,(i+1)%L] for i in range(0,L)]

static = [["+-",J_xy],["-+",J_xy],["zz",J_zz]]
dynamic=[]


for _i in dtypes.keys():
	dtype = dtypes[_i]
	atol = atols[_i]

	H=hamiltonian(static,dynamic,basis=basis,dtype=dtype,check_herm=False,check_symm=False)
	H2=hamiltonian(static,dynamic,basis=basis2,dtype=dtype,check_herm=False,check_symm=False,check_pcon=False)

	Proj = basis2.get_proj(dtype)

	H_proj = project_op(H.tocsr(),Proj,dtype=dtype)['Proj_Obs']
	H_proj2 = project_op(H,basis2,dtype=dtype)['Proj_Obs']
	H_proj3 = project_op(H.tocsr(),basis2,dtype=dtype)['Proj_Obs']
	H_proj4 = project_op(H,Proj,dtype=dtype)['Proj_Obs']

	np.testing.assert_allclose(H_proj.todense(),H_proj2.todense(),atol=atol,err_msg='Failed projectors comparison!')
	np.testing.assert_allclose(H_proj3.todense(),H_proj4.todense(),atol=atol,err_msg='Failed projectors comparison!')
	np.testing.assert_allclose(H_proj.todense(),H_proj3.todense(),atol=atol,err_msg='Failed projectors comparison!')


	H2_proj = project_op(H2.tocsr(),Proj,dtype=dtype)['Proj_Obs']
	H2_proj2 = project_op(H2,basis2,dtype=dtype)['Proj_Obs']
	H2_proj3 = project_op(H2.tocsr(),basis2,dtype=dtype)['Proj_Obs']
	H2_proj4 = project_op(H2,Proj,dtype=dtype)['Proj_Obs']

	np.testing.assert_allclose(H_proj.todense(),H_proj2.todense(),atol=atol,err_msg='Failed projectors comparison!')
	np.testing.assert_allclose(H_proj3.todense(),H_proj4.todense(),atol=atol,err_msg='Failed projectors comparison!')
	np.testing.assert_allclose(H_proj.todense(),H_proj3.todense(),atol=atol,err_msg='Failed projectors comparison!')


	### test KL div
	p1 = np.random.uniform(size=H.Ns)
	p1 *= 1.0/sum(p1)

	p2 = np.random.uniform(size=H.Ns)
	p2 *= 1.0/sum(p2)  

	KL_div(p1,p2)


	### test mean_level_spacing
	E2 = H2.eigvalsh() 
	mean_level_spacing(E2)

print("project_op, KZ_div and mean_level_spacing checks passed!")
