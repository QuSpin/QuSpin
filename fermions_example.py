from quspin.basis import spinless_fermion_basis_1d,boson_basis_1d,spin_basis_1d
from quspin.operators import hamiltonian
import numpy as np


np.set_printoptions(linewidth=100000,precision=2)

L = 8
a = 2
Nf = 4

basis = spinless_fermion_basis_1d(L,Nf=Nf)
# basis_hcb = boson_basis_1d(L,Nb=Nf,sps=2)


j = 1
t_l = [[-1.0,i,(i+j)%L] for i in range(L)]
t_r = [[ 1.0,i,(i+j)%L] for i in range(L)]

static = [["+-",t_r],["-+",t_l],["zz",t_r]]


checks = dict(check_symm=False,check_herm=False,check_pcon=False)

H = hamiltonian(static,[],basis=basis,**checks)

E_tot = H.eigvalsh()
E_symm = np.array([])

# static_hcb = [["+-",t_r],["-+",t_r]]
# H_hcb = hamiltonian(static_hcb,[],basis=basis_hcb,**checks)
# E_hcb = H_hcb.eigvalsh()
# assert(np.linalg.norm(E_tot-E_hcb)<1e-10)

#for k in [None]:
for k in range(L//a):
	k_basis = spinless_fermion_basis_1d(L,Nf=Nf,kblock=k,a=a)
	print k,k_basis
	H = hamiltonian(static,[],basis=k_basis,**checks)
	E_k = H.eigvalsh()


	M_list = []
	E_kp = np.array([],dtype=np.complex128)
	for p in [-1,1]:
		basis_symm = spinless_fermion_basis_1d(L,Nf=Nf,kblock=k,cblock=p,a=a)
		print k,p,basis_symm

		H = hamiltonian(static,[],basis=basis_symm,**checks)
		M = H.todense()
		M_list.append(M)
		
		E = H.eigvalsh()

		try:
			assert(np.linalg.norm(M-M.H)<1e-10)
		except AssertionError:
			print M
			print "kblock={} pblock={} failed hermitian test:".format(k,p)

		if k_basis.Ns == basis_symm.Ns and (2*a*k) % L != 0:
			if p == -1:
				E_symm = np.hstack((E_symm,E))
				E_kp = E
		else:
			E_symm = np.hstack((E_symm,E))
			E_kp = np.hstack((E_kp,E))

	E_kp.sort()

	try:
		assert(np.linalg.norm(E_k-E_kp)<1e-10)
	except:
		print "\n\n\n\n--------------------------------------"
		print "kblock={} failed at energies:".format(k)
		for M in M_list:
			print M
		print E_k
		print E_kp
		print "\n\n\n\n--------------------------------------"






E_symm.sort()
print np.linalg.norm(E_tot-E_symm)/basis.Ns