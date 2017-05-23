from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d, fermion_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt # figure/plot library
##### define model parameters #####
L=8 # system size
J=1.0 # spin zz interaction
h=np.sqrt(2) # z magnetic field strength
# loop over spin inversion symmetry block variable and boundary conditions
for zblock,PBC in zip([-1,1],[1,-1]):
	##### define spin model
	# site-coupling lists (PBC for both spin inversion sectors)
	h_field=[[-h,i] for i in range(L)]
	J_zz=[[-J,i,(i+1)%L] for i in range(L)] # PBC
	# define spin static and dynamic lists
	static_spin =[["zz",J_zz],["x",h_field]] # static part of H
	dynamic_spin=[] # time-dependent part of H
	# construct spin basis in pos/neg spin inversion sector depending on APBC/PBC
	basis_spin = spin_basis_1d(L=L,zblock=zblock) 
	# build spin Hamiltonians
	H_spin=hamiltonian(static_spin,dynamic_spin,basis=basis_spin,dtype=np.float64)
	# calculate spin energy levels
	E_spin=H_spin.eigvalsh()
	##### define fermion model
	# define site-coupling lists for external field
	h_pot=[[2.0*h,i] for i in range(L)]
	if PBC==1: # periodic BC: odd particle number subspace only
		# define site-coupling lists (including boudary couplings)
		J_pm=[[-J,i,(i+1)%L] for i in range(L)] # PBC
		J_mp=[[+J,i,(i+1)%L] for i in range(L)] # PBC
		J_pp=[[-J,i,(i+1)%L] for i in range(L)] # PBC
		J_mm=[[+J,i,(i+1)%L] for i in range(L)] # PBC
		# construct fermion basis in the odd particle number subsector
		basis_fermion = fermion_basis_1d(L=L,Nf=range(1,L+1,2))
	elif PBC==-1: # anti-periodic BC: even particle number subspace only
		# define bulk site coupling lists
		J_pm=[[-J,i,i+1] for i in range(L-1)]
		J_mp=[[+J,i,i+1] for i in range(L-1)]
		J_pp=[[-J,i,i+1] for i in range(L-1)]
		J_mm=[[+J,i,i+1] for i in range(L-1)]
		# add boundary coupling between sites (L-1,0)
		J_pm.append([+J,L-1,0]) # APBC
		J_mp.append([-J,L-1,0]) # APBC
		J_pp.append([+J,L-1,0]) # APBC
		J_mm.append([-J,L-1,0]) # APBC
		# construct fermion basis in the even particle number subsector
		basis_fermion = fermion_basis_1d(L=L,Nf=range(0,L+1,2))
	# define fermionic static and dynamic lists
	static_fermion =[["+-",J_pm],["-+",J_mp],["++",J_pp],["--",J_mm],['z',h_pot]]
	dynamic_fermion=[]
	# build fermionic Hamiltonian
	H_fermion=hamiltonian(static_fermion,dynamic_fermion,basis=basis_fermion,
							dtype=np.float64,check_pcon=False,check_symm=False)
	# calculate fermionic energy levels
	E_fermion=H_fermion.eigvalsh()
	##### plot spectra
	plt.plot(np.arange(H_fermion.Ns),E_fermion/L,marker='o'
									,color='b',label='fermion')
	plt.plot(np.arange(H_spin.Ns),E_spin/L,marker='x'
									,color='r',markersize=2,label='spin')
	plt.xlabel('state number',fontsize=16)
	plt.ylabel('energy',fontsize=16)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.legend(fontsize=16)
	plt.grid()
	plt.tight_layout()
	plt.show()