from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)
#######################################################################
#                            example 26                               #
# This example shows how to use the `Op-shit_sector` method of the    #
# general basis class to compute spectral functions using symmetries. #
#######################################################################
from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian,quantum_LinearOperator
import scipy.sparse as sp
import numexpr,cProfile
import numpy as np
import matplotlib.pyplot as plt
#
#
# define custom LinearOperator object that generates the left hand side of the equation.
#
class LHS(sp.linalg.LinearOperator):
	#
	def __init__(self,H,omega,eta,E0,kwargs={}):
		self._H = H # Hamiltonian
		self._z = omega +1j*eta + E0 # complex energy
		self._kwargs = kwargs # arguments
	#
	@property
	def shape(self):
		return (self._H.Ns,self._H.Ns)
	#
	@property
	def dtype(self):
		return np.dtype(self._H.dtype)
	#
	def _matvec(self,v):
		# left multiplication
		return self._z * v - self._H.dot(v,**self._kwargs)
	#
	def _rmatvec(self,v):
		# right multiplication
		return self._z.conj() * v - self._H.dot(v,**self._kwargs)
#
##### calculate action without constructing the Hamiltonian matrix
#
on_the_fly = False # toggles between using a `hamiltnian` or a `quantum_LinearOperator` object
# chain length
L = 12
# on-site spin size 
S = "1/2"
# translation transformation on the lattice sites [0,...,L-1]
T = (np.arange(L)+1)%L
# this example does not work under these conditions because ground-state sector is not q=0
if (L//2)%2 != 0:
	raise ValueError("Example requires modifications for Heisenberg chains with L=4*n+2.")
if L%2 != 0:
	raise ValueError("Example requires modifications for Heisenberg chains with odd number of sites.")
# construct basis
basis0 = spin_basis_general(L,S=S,m=0,pauli=False,kblock=(T,0))
# construct static list for Heisenberg chain
Jzz_list = [[1.0,i,(i+1)%L] for i in range(L)]
Jxy_list = [[0.5,i,(i+1)%L] for i in range(L)]
static = [["zz",Jzz_list],["+-",Jxy_list],["-+",Jxy_list]]
# construct operator for Hamiltonian in the ground state sector
if on_the_fly:
	H0 = quantum_LinearOperator(static,basis=basis0,dtype=np.float64)
else:
	H0 = hamiltonian(static,[],basis=basis0,dtype=np.float64)
# calculate ground state
[E0],psi0 = H0.eigsh(k=1,which="SA")
psi0 = psi0.ravel()
# define all possible momentum sectors excluding q=L/2 (pi-momentum) where the peak is abnormally large
qs = np.arange(-L//2+1,L//2,1)
# define frequencies to calculate spectral function for
omegas = np.arange(0,4,0.05)
# spectral peaks broadening factor
eta = 0.1
# allocate arrays to store data
Gzz = np.zeros(omegas.shape+qs.shape,dtype=np.complex128)
Gpm = np.zeros(omegas.shape+qs.shape,dtype=np.complex128)
# loop over momentum sectors
for j,q in enumerate(qs):
	print("computing momentum block q={}".format(q) )
	# define block
	block = dict(qblock=(T,q))
	#
	####################################################################
	#
	# ---------------------------------------------------- #
	#                 calculation but for SzSz             #
	# ---------------------------------------------------- #
	#
	# define operator list for Op_shift_sector
	f = lambda i:np.exp(-2j*np.pi*q*i/L)/np.sqrt(L)
	Op_list = [["z",[i],f(i)] for i in range(L)]
	# define basis
	basisq = spin_basis_general(L,S=S,m=0,pauli=False,**block)
	# define operators in the q-momentum sector
	if on_the_fly:
		Hq = quantum_LinearOperator(static,basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	else:
		Hq = hamiltonian(static,[],basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	# shift sectors
	psiA = basisq.Op_shift_sector(basis0,Op_list,psi0)
	#
	### apply vector correction method
	#
	# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
	for i,omega in enumerate(omegas):
		lhs = LHS(Hq,omega,eta,E0)
		x,*_ = sp.linalg.bicg(lhs,psiA)
		Gzz[i,j] = -np.vdot(psiA,x)/np.pi
	#
	#####################################################################
	#
	# ---------------------------------------------------- #
	#            same calculation but for S-S+             #
	# ---------------------------------------------------- #
	#
	# divide by extra sqrt(2) to get extra factor of 1/2 when taking sandwich: needed since H = 1/2 (S^+_i S^-_j + h.c.) + S^z_j S^z_j
	f = lambda i:np.exp(-2j*np.pi*q*i/L)*np.sqrt(1.0/(2*L))
	Op_list = [["+",[i],f(i)] for i in range(L)]
	# change S_z projection up by S for action of S+ operator
	S_z_tot = 0 + eval(S)
	# calculate magnetization density from S_z_tot as defined in the documentation
	# m = S_z_tot / (S * N), S: local spin (as number), N: total spins
	m = S_z_tot/(eval(S)*L) 
	basisq = spin_basis_general(L,S=S,m=m,pauli=False,**block)
	# define operators in the q-momentum sector
	if on_the_fly:
		Hq = quantum_LinearOperator(static,basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	else:
		Hq = hamiltonian(static,[],basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	# shift sectors
	psiA = basisq.Op_shift_sector(basis0,Op_list,psi0)
	#
	### apply vector correction method
	#
	# solve (z-H)|x> = |A> solve for |x>  using iterative solver for each omega
	for i,omega in enumerate(omegas):
		lhs = LHS(Hq,omega,eta,E0)
		x,*_ = sp.linalg.bicg(lhs,psiA)
		Gpm[i,j] = -np.vdot(psiA,x)/np.pi
#
##### plot results
#
ks = 2*np.pi*qs/L # compute physical momentum values
f,(ax1,ax2) = plt.subplots(2,1,figsize=(3,4),sharex=True)
ax1.pcolormesh(ks,omegas,Gzz.imag,shading='nearest')
ax2.pcolormesh(ks,omegas,Gpm.imag,shading='nearest')
ax2.set_xlabel('$k$')
ax2.set_ylabel('$\\omega$')
ax1.set_ylabel('$\\omega$')
ax1.set_title('$G_{zz}(\\omega,k)$')
ax2.set_title('$G_{+-}(\\omega,k)$')
#plt.show()
plt.close()