from qspin.operators import hamiltonian,exp_op # Hamiltonians and operators
from qspin.basis import spin_basis_1d # Hilbert space spin basis
from qspin.tools.measurements import ent_entropy, diag_ensemble, obs_vs_time # entropies
from numpy.random import ranf,seed # pseudo random numbers
from joblib import delayed,Parallel # parallelisation
import numpy as np # generic math functions
import scipy.sparse as sp
from time import time # timing package

#
##### define model parameters #####
L=10 # system size
Jxy=1.0 # xy interaction
Jzz_0=1.0 # zz interaction at time t=0
h_MBL=3.72 # MBL disorder strength
h_ETH=0.1 # delocalised disorder strength
start = 0
stop = 0.5
num = 101

n = sum((2**i for i in xrange(0,L/2,1))) # initial state
#

dtype = np.float64
##### set up Heisenberg Hamiltonian with linearly varying zz-interaction #####
# compute basis in the 0-total magnetisation sector (requires L even)
basis = spin_basis_1d(L,Nup=L/2,pauli=False)

n = basis._basis.searchsorted(n)

psi_0 = np.zeros((basis.Ns,),dtype=dtype)
psi_0[n] = 1.0

Sent_args = {}
Sent_args["basis"] = basis

# define operators with OBC using site-coupling lists
J_zz = [[Jzz_0,i,i+1] for i in range(0,L-1)] # OBC
J_xy = [[Jxy/2.0,i,i+1] for i in range(0,L-1)] # OBC

# static and dynamic lists
static = [["zz",J_zz],["+-",J_xy],["-+",J_xy]]
# compute the time-dependent Heisenberg Hamiltonian
H_XXZ = hamiltonian(static,[],basis=basis,dtype=np.float64)
#
psi_0 = np.zeros((H_XXZ.Ns,),dtype=dtype)
psi_0[n] = 1.0

##### calculate diagonal and entanglement entropies #####
def relization(H_0,basis,w,psi_0,start,stop,num,real):
	seed() # the random number needs to be seeded for each parallel process

	fields=w*(-1+2*ranf((L,)))

	h_z=[[fields[i],i] for i in range(basis.L)]
	disorder_field = [["z",h_z]]

	Hz = hamiltonian(disorder_field,[],basis=basis,dtype=np.float64,check_herm=False,check_pcon=False,check_symm=False)

	H = H_0 + Hz

#	Emin,Emax=H.eigsh(k=2,which="BE",maxiter=1E4,return_eigenvectors=False)
#	E_inf_temp=(Emax+Emin)/2.0

	[Emin] = H.eigsh(k=1,which="SA",maxiter=1E4,return_eigenvectors=False)
	H -= Emin * sp.identity(H.Ns,dtype = H.dtype)

#	expO = exp_op(H,a=-1,start=start,stop=stop,num=num,endpoint=True,iterate=True)
#	psi_t = expO.dot(psi_0)
#	times = np.array(expO.grid)

	times = np.linspace(start,stop,num=num,endpoint=True)
	psi_t = H.evolve(psi_0,0,times,iterate=True,imag_time=True)

	Sent_args = {}
	Sent_args["basis"] = basis
	Sent_args["chain_subsys"] = range(basis.L/2)

	dyn = obs_vs_time(psi_t,times,{},Sent_args=Sent_args)
	return times,dyn["Sent_time"]["Sent"]

#	Sent_before = 0
#	for i,psi in enumerate(psi_t):
#		Sent = ent_entropy(psi,basis,chain_subsys=range(basis.L/2))["Sent"]
#		print Sent,Sent_before
#		if Sent_before > Sent:
#			return times[i]*L
#		Sent_before = Sent

#	return np.nan




#
##### plot results #####
import matplotlib.pyplot as plt

"""
data = Parallel(n_jobs=4)(delayed(relization)(H_XXZ,basis,h_ETH,psi_0,start,1,num,i) for i in range(100))
print np.nanstd(data),np.nanmean(data)
data = Parallel(n_jobs=4)(delayed(relization)(H_XXZ,basis,h_MBL,psi_0,start,1,num,i) for i in range(100))
print np.nanstd(data),np.nanmean(data)
data = Parallel(n_jobs=4)(delayed(relization)(H_XXZ,basis,20.0,psi_0,start,1,num,i) for i in range(100))
print np.nanstd(data),np.nanmean(data)
"""
n_jobs = 4
n_iter = 100
tf = 30

data = Parallel(n_jobs=n_jobs)(delayed(relization)(H_XXZ,basis,h_ETH,psi_0,start,tf,num,i) for i in range(n_iter))
times,Sent = zip(*data)
times = np.mean(times,axis=0)
Sent = np.mean(Sent,axis=0)
plt.plot(times,Sent,color="blue",marker=".")
#for i in range(n_iter):
#	plt.plot(times[i],Sent[i],color="blue",marker=".")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$S_\mathrm{ent}$")
plt.figure()

data = Parallel(n_jobs=n_jobs)(delayed(relization)(H_XXZ,basis,10.0,psi_0,start,tf,num,i) for i in range(n_iter))
times,Sent = zip(*data)
times = np.mean(times,axis=0)
Sent = np.mean(Sent,axis=0)
plt.plot(times,Sent,color="green",marker=".")
#for i in range(n_iter):
#	plt.plot(times[i],Sent[i],color="green",marker=".")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$S_\mathrm{ent}$")
plt.figure()

data = Parallel(n_jobs=n_jobs)(delayed(relization)(H_XXZ,basis,h_MBL,psi_0,start,tf,num,i) for i in range(n_iter))
times,Sent = zip(*data)
times = np.mean(times,axis=0)
Sent = np.mean(Sent,axis=0)
plt.plot(times,Sent,color="red",marker=".")
#for i in range(n_iter):
#	plt.plot(times[i],Sent[i],color="red",marker=".")
plt.xlabel(r"$\tau$")
plt.ylabel(r"$S_\mathrm{ent}$")
plt.show()




