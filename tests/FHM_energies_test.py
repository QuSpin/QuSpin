from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import tensor_basis, spinless_fermion_basis_1d, spinful_fermion_basis_1d#, spinful_fermion_basis_general, spinless_fermion_basis_general
from quspin.basis.transformations import square_lattice_trans
from quspin.operators import hamiltonian,quantum_operator
import numpy as np


Lx=Ly=4
N = Lx*Ly

# energies from arXiv:cond-mat/0604319, Sec III A
E_paper=-np.array([12.0000000000000000,
				   11.9126285145094126,
				   11.8364690235642218,
				   11.7695877912829268,
				   11.7104580743242632,
				   11.6578605150913841,
				   11.6108110503797608,
				   11.5685079573602838,
				   11.5302924026297138,
				   ])


tr = square_lattice_trans(Lx,Ly)

Jp = [[1.0,i,tr.T_x[i]] for i in range(N)]
Jp.extend([[1.0,i,tr.T_y[i]] for i in range(N)])

Jm = [[-1.0,i,tr.T_x[i]] for i in range(N)]
Jm.extend([[-1.0,i,tr.T_y[i]] for i in range(N)])

U_onsite = [[1.0,i,i] for i in range(N)]

operator_list_0=[["+-|",Jp],["-+|",Jm],["|+-",Jp],["|-+",Jm]]
operator_list_1=[["n|n",U_onsite]]

operator_dict=dict(H0=operator_list_0,H1=operator_list_1)



basis_f=spinless_fermion_basis_1d(L=N,Nf=2)
basis_1=tensor_basis(basis_f,basis_f)

# basis_f=spinless_fermion_basis_general(N,Nf=2)
# basis_2=tensor_basis(basis_f,basis_f)


basis_3=spinful_fermion_basis_1d(L=N,Nf=(2,2))
# basis_4=spinful_fermion_basis_general(N,Nf=(2,2))

basis_dict=dict(tensored_spinless_fermion_basis_1d=basis_1,
			spinful_fermion_basis_1d=basis_3,
			# tensored_spinless_fermion_basis_general=basis_2,
			# spinful_fermion_basis_general=basis_4
			)
for basis_name,basis in basis_dict.items():

	H_U = quantum_operator(operator_dict,basis=basis,dtype=np.float64,check_pcon=False,check_symm=False,check_herm=False)

	E_quspin=np.zeros(E_paper.shape,)
	for j, U in enumerate(np.linspace(0.0,4.0,9)):

		params_dict=dict(H0=1.0,H1=U)
		H=H_U.tohamiltonian(params_dict)

		E_quspin[j]=H.eigsh(k=1,which="SA",maxiter=1E4,return_eigenvectors=False)

	np.testing.assert_allclose(E_quspin,E_paper,atol=1e-13)
	print("passed Fermi Hubbard model test for " + basis_name)



