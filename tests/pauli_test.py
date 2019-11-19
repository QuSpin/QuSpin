from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_general,spin_basis_1d
from quspin.operators import hamiltonian
from quspin.operators._make_hamiltonian import _consolidate_static

import numpy as np



L=2

sc_list=[[1.0,i,(i+1)%L] for i in range(L)]

opstr_pm=[['+-',sc_list]]
opstr_mp=[['-+',sc_list]]
opstr_xx=[['xx',sc_list]]
opstr_yy=[['yy',sc_list]]
opstr_zz=[['zz',sc_list]]

opstr_lists=[opstr_pm, opstr_mp, opstr_xx, opstr_yy, opstr_zz]

def compare_Op_bra_ket(static_list,basis_1,basis_2,ratio_12,test_ratio=True):
	for opstr,indx,J in static_list:
		ME_1,_,_ = basis_1.Op_bra_ket(opstr,indx,J,np.float64,basis_1.states)
		ME_2,_,_ = basis_2.Op_bra_ket(opstr,indx,J,np.float64,basis_2.states)

		if test_ratio and (opstr=='+-' or opstr=='-+'):
			ME_2/=ratio_12

		np.testing.assert_allclose(ME_1 - ratio_12*ME_2,0.0,atol=1E-5)

def compare_Op(static_list,basis_1,basis_2,ratio_12,test_ratio=True):
	for opstr,indx,J in static_list:
		ME_1,_,_ = basis_1.Op(opstr,indx,J,np.float64)
		ME_2,_,_ = basis_2.Op(opstr,indx,J,np.float64)

		if test_ratio and (opstr=='+-' or opstr=='-+'):
			ME_2/=ratio_12
		np.testing.assert_allclose(ME_1 - ratio_12*ME_2,0.0,atol=1E-5)

def compare_inplace_Op(static_list,basis_1,basis_2,ratio_12,test_ratio=True):
	for opstr,indx,J in static_list:
		
		dtype=np.float64

		v_in=np.random.uniform(size=basis_1.Ns)
		#v_in/=np.linalg.norm(v_in)

		op_list = [[opstr,indx,J]]
		v_out_1=basis_1.inplace_Op(v_in,op_list,dtype,transposed=False,conjugated=False,v_out=None)
		v_out_2=basis_2.inplace_Op(v_in,op_list,dtype,transposed=False,conjugated=False,v_out=None)

		if test_ratio and (opstr=='+-' or opstr=='-+'):
			v_out_2/=ratio_12

		np.testing.assert_allclose(v_out_1 - ratio_12*v_out_2,0.0,atol=1E-5)


def compare_hamiltonian(basis_1,basis_2,ratio_12,test_ratio=True):

	pm_1=hamiltonian(opstr_pm,[],basis=basis_1,**no_checks).toarray()
	mp_1=hamiltonian(opstr_mp,[],basis=basis_1,**no_checks).toarray()
	xx_1=hamiltonian(opstr_xx,[],basis=basis_1,**no_checks).toarray()
	yy_1=hamiltonian(opstr_yy,[],basis=basis_1,**no_checks).toarray()
	zz_1=hamiltonian(opstr_zz,[],basis=basis_1,**no_checks).toarray()

	
	pm_2=hamiltonian(opstr_pm,[],basis=basis_2,**no_checks).toarray()
	mp_2=hamiltonian(opstr_mp,[],basis=basis_2,**no_checks).toarray()
	xx_2=hamiltonian(opstr_xx,[],basis=basis_2,**no_checks).toarray()
	yy_2=hamiltonian(opstr_yy,[],basis=basis_2,**no_checks).toarray()
	zz_2=hamiltonian(opstr_zz,[],basis=basis_2,**no_checks).toarray()

	if test_ratio:
		np.testing.assert_allclose(pm_1, ratio_12*pm_2,atol=1e-13)
		np.testing.assert_allclose(mp_1, ratio_12*mp_2,atol=1e-13)
		
	else:

		np.testing.assert_allclose(pm_1, ratio_12*pm_2,atol=1e-13)
		np.testing.assert_allclose(mp_1, ratio_12*mp_2,atol=1e-13)
		
		np.testing.assert_allclose(xx_1, ratio_12*xx_2,atol=1e-13)
		np.testing.assert_allclose(yy_1, ratio_12*yy_2,atol=1e-13)
		np.testing.assert_allclose(zz_1, ratio_12*zz_2,atol=1e-13)

	

basis_1d_S=spin_basis_1d(L=L,pauli=0)
basis_1d_pauli_1=spin_basis_1d(L=L)
basis_1d_pauli_2=spin_basis_1d(L=L,pauli=-1)

spin_basis_general_S=spin_basis_general(N=L,pauli=0)
spin_basis_general_pauli_1=spin_basis_general(N=L)
spin_basis_general_pauli_2=spin_basis_general(N=L,pauli=-1)

bases=[(basis_1d_S,basis_1d_pauli_1,basis_1d_pauli_2), (spin_basis_general_S,spin_basis_general_pauli_1,spin_basis_general_pauli_2)]


no_checks=dict(check_herm=False,check_pcon=False,check_symm=False,dtype=np.float64)


for basis_S, basis_pauli_1, basis_pauli_2 in bases:

	# check Op and Op_bra_ket functions
	for static in opstr_lists:

		static_list = _consolidate_static(static)

		compare_inplace_Op(static_list,basis_S,basis_pauli_1,1/2**2, test_ratio=False)
		compare_inplace_Op(static_list,basis_S,basis_pauli_2,1/2**2, test_ratio=True)
	
		compare_Op(static_list,basis_S,basis_pauli_1,1/2**2, test_ratio=False)
		compare_Op(static_list,basis_S,basis_pauli_2,1/2**2, test_ratio=True)

		if isinstance(basis_S,spin_basis_general):

			compare_Op_bra_ket(static_list,basis_S,basis_pauli_1,0.25, test_ratio=False)
			compare_Op_bra_ket(static_list,basis_S,basis_pauli_2,0.25, test_ratio=True)
			

	# check quantum operators
	compare_hamiltonian(basis_S,basis_pauli_1,1/2**2, test_ratio=False)
	compare_hamiltonian(basis_S,basis_pauli_1,1/2**2, test_ratio=True)
	

