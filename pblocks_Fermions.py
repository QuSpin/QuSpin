from quspin.basis import spinless_fermion_basis_1d
from quspin.operators import hamiltonian
import numpy as np
import scipy.sparse as sp



dtype=np.complex128

L = 18
Nf = 9
a = 1

basis = spinless_fermion_basis_1d(L,Nf=Nf,cblock=1)


# basis_full = spinless_fermion_basis_1d(L,Nf=Nf)
# b1 = basis_full._basis.copy()
# b2 = basis_full._basis.copy()
# signs1 = np.ones_like(b1,dtype=np.int8)
# signs2 = np.ones_like(b2,dtype=np.int8)
# basis._bitops.py_shift(b1,-a,L,basis._pars,signs=signs1)
# basis._bitops.py_flip_all(b1,L,basis._pars,signs=signs1)
# basis._bitops.py_flip_all(b2,L,basis._pars,signs=signs2)
# basis._bitops.py_shift(b2,-a,L,basis._pars,signs=signs2)
# print (signs1-signs2)//2
# print np.linalg.norm((signs1-signs2)//(basis.Ns*2))**2
#exit()



# basis_hcb = boson_basis_1d(L,Nb=Nf,sps=2)

j = 1
t_l = [[-1.0,i,(i+j)%L] for i in range(L)]
t_r = [[ 1.0,i,(i+j)%L] for i in range(L)]

static = [["+-",t_r],["-+",t_l]]
M = sp.csr_matrix((basis.Ns,basis.Ns),dtype=np.complex128)
for opstr,indxs in static:
	for indx in indxs:
		J = indx[0]
		indx = np.array(indx[1:])

		ME,row,col = basis.Op(opstr,indx,J,dtype)
		MM = sp.csr_matrix((ME,(row,col)),shape=M.shape,dtype=np.complex128)
		M += MM
		print opstr,indx
		print MM.todense()
		print

print M.todense()