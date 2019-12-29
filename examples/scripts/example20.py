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
#                            example 20                               #	
# This example shows how to use the `Lanczox` submodule of the        #
# `tools` module to compute the time evolution of a quantum state     #
# and how to find ground states of hermitian Hamiltonians.            #
#######################################################################
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian
from scipy.sparse.linalg import expm_multiply
from quspin.tools.lanczos import lanczos_full,lanczos_iter,lin_comb_Q,expm_lanczos
import numpy as np
import scipy.linalg as sla
#
np.random.seed(17) # set random seed, cf initial state below
#
##### Heisenberg model
L = 20 # system size
dt= 0.1 # unitary evolution time step
# basis object
basis = spin_basis_1d(L,m=0,kblock=0,pblock=1,zblock=1)
print("\nHilbert space dimension: {}.\n".format(basis.Ns))
# Heisenberg Hamiltonian
J_list = [[1.0,i,(i+1)%L] for i in range(L)]
static = [[op,J_list] for op in ["xx","yy","zz"]]
H = hamiltonian(static,[],basis=basis,dtype=np.float64)
#
##### Lanczos time evolution calculation
#
m_evo=20 # Krylov subspace dimension
#
# initial states
v0 = np.random.normal(0,1,size=basis.Ns)
v0 /= np.linalg.norm(v0)
# make copies to test the lanczos routines independently
v0_lanczos_full = v0.copy()
v0_lanczos_iter = v0.copy()
#
print("\nchecking lanczos matrix exponential calculation:\n")
for i in range(100):
	# compute Lanczos decomposition
	E1,V1,Q_full = lanczos_full(H,v0_lanczos_full,m_evo) # all Lanczps vectors at once
	E2,V2,Q_iter = lanczos_iter(H,v0_lanczos_iter,m_evo) # Lanczos vectors as an iterator
	# evolve quantum state using different routines
	v_expm_multiply_t = expm_multiply(-1j*dt*H.static,v0) # cf tools.expm_multiply_parallel with OMP speedup
	v_lanczos_full_t = expm_lanczos(E1,V1,Q_full,a=-1j*dt)
	v_lanczos_iter_t = expm_lanczos(E2,V2,Q_iter,a=-1j*dt)
	# test results against each other
	np.testing.assert_allclose(v_lanczos_full_t,v_expm_multiply_t,atol=1e-10,rtol=0)
	np.testing.assert_allclose(v_lanczos_iter_t,v_expm_multiply_t,atol=1e-10,rtol=0)
	#
	print("finished unitary evolution step: {0:d}.".format(i))
#
print("\ntime evolution complete.\n")
#
###### Lanczos ground state calculation
#
m_GS=50 # Krylov subspace dimension
#
E_GS,psi_GS = H.eigsh(k=1,which="SA")

psi_GS = psi_GS.ravel()

v0 = np.random.normal(0,1,size=basis.Ns)
v0 /= np.linalg.norm(v0)

E,V,Q = lanczos_full(-H,v0,m_GS,full_ortho=False)
E=-E[::-1]
V=V[:,::-1]


try:
	dE = np.abs(E[0]-E_GS[0])
	assert(dE < 1e-10)
except AssertionError:
	raise AssertionError("Energy failed to converge |E_lanczos-E_exact| = {} > 1e-10".format(dE))

v1 = lin_comb_Q(V[:,0],Q)
try:
	F = np.abs(np.log(np.abs(np.vdot(v1,psi_GS))))
	assert(F < 1e-10)
except AssertionError:
	raise AssertionError("wavefunction failed to converge Fedility = {} > 1e-10".format(F))

E,V,Q_iter = lanczos_iter(H,v0,m_GS,return_vec_iter=True)

v2 = lin_comb_Q(V[:,0],Q_iter)

np.testing.assert_allclose(v2,v1,atol=1e-10,rtol=0)


