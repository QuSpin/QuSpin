from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_1d, boson_basis_1d, fermion_basis_1d # Hilbert space spin basis
import numpy as np # generic math functions
import matplotlib.pyplot as plt
##### define model parameters #####
L=8 # system size
J=1.0 # spin interaction
hz=np.sqrt(2) # magnetic field
# loop ober boundary conditions/spin inversion block variable
for PBC in [-1,1]: # periodic or antiperiodic BC
	##### define fermion model
	# define site-coupling lists for external field
	x_field=[[2.0*hz,i] for i in range(L)]
	if PBC==1: # periodic boundary conditions, include odd particle number subspace only
		# define site-coupling lists (including boudary couplings)
		J_pm=[[-J,i,(i+1)%L] for i in range(L)] # PBC
		J_mp=[[+J,i,(i+1)%L] for i in range(L)] # PBC
		J_pp=[[-J,i,(i+1)%L] for i in range(L)] # PBC
		J_mm=[[+J,i,(i+1)%L] for i in range(L)] # PBC
		# construct fermion basis in the odd particle number subsector
		basis_fermion = fermion_basis_1d(L=L,Nf=range(1,L+1,2))
	elif PBC==-1: # anti-periodic boundary conditions, include even particle number subspace only
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
	# define fermionic static list
	static_fermion =[["+-",J_pm],["-+",J_mp],["++",J_pp],["--",J_mm],['z',x_field]]
	# build fermionic Hamiltonian
	H_fermion=hamiltonian(static_fermion,[],basis=basis_fermion,dtype=np.float64,check_pcon=False)
	# calculate fermionic energy levels
	E_fermion=H_fermion.eigvalsh()
	##### define spin model
	# site-coupling lists (PBC in both cases)
	J_zz=[[-J,i,(i+1)%L] for i in range(L)] # PBC
	x_field=[[-hz,i] for i in range(L)]
	# determine Hilbert space symemtries
	if PBC==1: # include odd spin inversion sector only
		basis_spin = spin_basis_1d(L=L,zblock=-1)
	elif PBC==-1: # include even spin inversion sector only
		basis_spin = spin_basis_1d(L=L,zblock=1)
	# define spin static list
	static_spin =[["zz",J_zz],["x",x_field]]
	# build spin Hamiltonian
	H_spin=hamiltonian(static_spin,[],basis=basis_spin,dtype=np.float64)
	# calculate spin energy levels
	E_spin=H_spin.eigvalsh()
	##### plot spectra
	plt.plot(np.arange(H_fermion.Ns),E_fermion/L,marker='o',color='b',label='fermion')
	plt.plot(np.arange(H_spin.Ns),E_spin/L,marker='x',color='r',markersize=2,label='spin')
	plt.xlabel('state number',fontsize=16)
	plt.ylabel('energy',fontsize=16)
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.legend(fontsize=16)
	plt.grid()
	plt.tight_layout()
	plt.show()