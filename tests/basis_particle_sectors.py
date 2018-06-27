from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_1d, boson_basis_1d, spinless_fermion_basis_1d, spinful_fermion_basis_1d
from quspin.basis import spin_basis_general, boson_basis_general, spinless_fermion_basis_general, spinful_fermion_basis_general
from itertools import product
import numpy as np

for L in [6,7]:

	# symmetry-free

	basis_1=spin_basis_1d(L=L,Nup=range(0,L,2))
	basis_1g=spin_basis_general(N=L,Nup=range(0,L,2))

	basis_2=boson_basis_1d(L=L,Nb=range(0,L,2))
	basis_2g=boson_basis_general(N=L,Nb=range(0,L,2))

	basis_3=spinless_fermion_basis_1d(L=L,Nf=range(0,L,2))
	basis_3g=spinless_fermion_basis_general(N=L,Nf=range(0,L,2))

	basis_4=spinful_fermion_basis_1d(L=L,Nf=product(range(0,L,2),range(0,L,2)) )
	basis_4g=spinful_fermion_basis_general(N=L,Nf=product(range(0,L,2),range(0,L,2)) )


	# symmetry-ful

	t = (np.arange(L)+1)%L

	basis_1=spin_basis_1d(L=L,Nup=range(0,L,2),kblock=0)
	basis_1g=spin_basis_general(N=L,Nup=range(0,L,2),kblock=(t,0))

	basis_2=boson_basis_1d(L=L,Nb=range(0,L,2),kblock=0)
	basis_2g=boson_basis_general(N=L,Nb=range(0,L,2),kblock=(t,0))

	basis_3=spinless_fermion_basis_1d(L=L,Nf=range(0,L,2),kblock=0)
	basis_3g=spinless_fermion_basis_general(N=L,Nf=range(0,L,2),kblock=(t,0))

	basis_4=spinful_fermion_basis_1d(L=L,Nf=product(range(0,L,2),range(0,L,2)),kblock=0 )
	basis_4g=spinful_fermion_basis_general(N=L,Nf=product(range(0,L,2),range(0,L,2)),kblock=(t,0))


	print("passed particle number sectors test")


