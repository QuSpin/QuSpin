from __future__ import print_function, division
#
import sys,os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
#
quspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,quspin_path)


from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian,quantum_LinearOperator
import scipy.sparse as sp
import numexpr,cProfile
import numpy as np
import matplotlib.pyplot as plt




class LHS(sp.linalg.LinearOperator):
	# LinearOperator that generates the left hand side of the equation.
	def __init__(self,H,omega,eta,E0,kwargs={}):
		self._H = H
		self._S = omega +1j*eta + E0 
		self._kwargs = kwargs

	@property
	def shape(self):
		return (self._H.Ns,self._H.Ns)
	
	@property
	def dtype(self):
		return np.dtype(self._H.dtype)
	
	def _matvec(self,v):
		return self._S * v - self._H.dot(v,**self._kwargs)

	def _rmatvec(self,v):
		return self._S.conj() * v - self._H.dot(v,**self._kwargs)

# calculate action without constructing the Hamiltonian matrix
on_the_fly = False
# chain length
L = 12
# local spin 
S = "1/2"
# translation transformation
T = (np.arange(L)+1)%L
# this example does not work for these conditions because ground-state sector is no q=0 and S=1/2
if (L//2)%2 != 0:
	raise ValueError("Example is not written for heisenberg chains with L=4*n+2.")
if L%2 != 0:
	raise ValueError("Example is not written for heisenberg chains with odd number of sites.")
# construct basis
basis0 = spin_basis_general(L,S=S,m=0,pauli=False,kblock=(T,0))
# construct static list for Heisenberg chain
Jzz_list = [[1.0,i,(i+1)%L] for i in range(L)]
Jxy_list = [[0.5,i,(i+1)%L] for i in range(L)]
static = [["zz",Jzz_list],["+-",Jxy_list],["-+",Jxy_list]]
# construct operator for Hamiltonian in ground state sector
if on_the_fly:
	H0 = quantum_LinearOperator(static,basis=basis0,dtype=np.float64)
else:
	H0 = hamiltonian(static,[],basis=basis0,dtype=np.float64)
# calculate ground state.
[E0],psi0 = H0.eigsh(k=1,which="SA")
psi0 = psi0.ravel()
# list of possible momentum sectors
# excluding k=pi because the peak is large
qs = np.arange(-L//2+1,L//2,1)
# list of omegas to calculate spectral function
omegas = np.arange(0,4,0.05)
# broadening factor
eta = 0.1
# allocate arrays to store data
Gzz = np.zeros(omegas.shape+qs.shape,dtype=np.complex128)
Gpm = np.zeros(omegas.shape+qs.shape,dtype=np.complex128)
# looping over momentum sectors
for j,q in enumerate(qs):
	print("computing momentum block q={}".format(q) )
	# define block
	block = dict(kblock=(T,q))
	# define operator list to Op_shift_sector
	f = lambda i:np.exp(-2j*np.pi*q*i/L)/np.sqrt(L)
	Op_list = [["z",[i],f(i)] for i in range(L)]
	# define basis
	basisq = spin_basis_general(L,S=S,m=0,pauli=False,**block)
	# define operators for the sector.
	if on_the_fly:
		Hq = quantum_LinearOperator(static,basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	else:
		Hq = hamiltonian(static,[],basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	# shift sectors
	psiA = basisq.Op_shift_sector(basis0,Op_list,psi0)
	# use vector correction method:
	# solve (z-H)|x> = |A> solve for |x> 
	# using iterative solver for each omega
	for i,omega in enumerate(omegas):
		lhs = LHS(Hq,omega,eta,E0)
		x,*_ = sp.linalg.bicg(lhs,psiA)
		Gzz[i,j] = -np.vdot(psiA,x)/np.pi
	# ---------------------------------------------------- #
	#            same calculation but for S-S+             #
	# ---------------------------------------------------- #
	# divide by extra sqrt(2) to get extra factor of 1/2 when taking sandwich
	f = lambda i:np.exp(-2j*np.pi*q*i/L)*np.sqrt(1.0/(2*L))
	Op_list = [["+",[i],f(i)] for i in range(L)]
	# change S_z projection up by S for action of S+ operator
	S_z_tot = 0 + eval(S)
	# calculate magnetization density from S_z_tot as define in docs
	# m = S_z_tot / (S * N), S: local spin (as number), N: total spins
	m = S_z_tot/(eval(S)*L) 
	basisq = spin_basis_general(L,S=S,m=m,pauli=False,**block)
	# define operators for the sector.
	if on_the_fly:
		Hq = quantum_LinearOperator(static,basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	else:
		Hq = hamiltonian(static,[],basis=basisq,dtype=np.complex128,
			check_symm=False,check_pcon=False,check_herm=False)
	# shift sectors
	psiA = basisq.Op_shift_sector(basis0,Op_list,psi0)
	# use vector correction method:
	# solve (z-H)|x> = |A> solve for |x> 
	# using iterative solver
	for i,omega in enumerate(omegas):
		lhs = LHS(Hq,omega,eta,E0)
		x,*_ = sp.linalg.bicg(lhs,psiA)
		Gpm[i,j] = -np.vdot(psiA,x)/np.pi
# plot results
qs = 2*np.pi*qs/L
f,(ax1,ax2) = plt.subplots(2,1,figsize=(3,4),sharex=True)
ax1.pcolormesh(qs,omegas,Gzz.imag,shading='nearest')
ax2.pcolormesh(qs,omegas,Gpm.imag,shading='nearest')
f.subplots_adjust(hspace=0.01)
plt.show()