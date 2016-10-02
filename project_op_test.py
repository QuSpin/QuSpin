from qspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from qspin.operators import hamiltonian # Hamiltonian and observables
from qspin.tools.measurements import project_op, KL_div, mean_level_spacing
import numpy as np
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers
seed()


"""
This test only makes sure the function 'obs_vs_time' runs properly.
"""

dtypes={"float32":np.float32,"float64":np.float64,"float128":np.float128,
		"complex64":np.complex64,"complex128":np.complex128,"complex256":np.complex256}

atols={"float32":1E-4,"float64":1E-13,"float128":1E-13,
		"complex64":1E-4,"complex128":1E-13,"complex256":1E-13}


L=12
basis = spin_basis_1d(L)
basis2 = spin_basis_1d(L,Nup=L/2,kblock=0,pblock=1,zblock=1)


J_zz = [[1.0,i,(i+1)%L] for i in range(0,L)] 
J_xy = [[1.0,i,(i+1)%L] for i in range(0,L)]

static = [["+-",J_xy],["-+",J_xy],["zz",J_zz]]
dynamic=[]


dtype=dtypes["float64"]
atol=atols["float64"]

H=hamiltonian(static,dynamic,basis=basis,dtype=dtype,check_herm=False,check_symm=False)
H2=hamiltonian(static,dynamic,basis=basis2,dtype=dtype,check_herm=False,check_symm=False,check_pcon=False)

Proj = basis2.get_proj(dtype)

H_proj = project_op(H.tocsr(),Proj,dtype=dtype)['Proj_Obs']
H_proj2 = project_op(H,basis2,dtype=dtype)['Proj_Obs']

np.testing.assert_allclose(H_proj.todense(),H_proj2.todense(),atol=atol,err_msg='Failed projectors comparison!')


### test KL div
E = H.eigvalsh()

a = uniform(4.0)
KL_div(abs(E)/sum(abs(E)),abs(E-a)/sum(abs(E-a)))


### test mean_level_spacing
E2 = H2.eigvalsh()
mean_level_spacing(E2)


