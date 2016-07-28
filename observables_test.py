from exact_diag_py.tools import observables

from exact_diag_py.tools.Floquet import Floquet

from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

from numpy.linalg import norm
from numpy.random import random,seed


from joblib import delayed,Parallel
from numpy import vstack



# parallelisation params
U_jobs = 1
n_jobs = 1

# system size
L = 4
block = 2

### static model params
J = 1.0
hx = 0.809
hz = 0.9045

### dynamic model params
def f(t,Omega):
	return np.cos(Omega*t)
Omega = 10.0
A = 1.0
def t_vec(Omega,N_const,len_T=100,N_up=0,N_down=0):
	# define dynamics params for a time step of 'da=1/len_T'
	# N_up, N_down: ramp-up (-down) periods
	# N_const: number of periods of constant evolution
	# Omega: drive frequency

	t_vec.N_up=N_up
	t_vec.N_down=N_down
	t_vec.N_const=N_const

	t_vec.len_T = len_T
	t_vec.T = 2.0*np.pi/Omega # driving period T

	
	t_up = N_up*t_vec.T # ramp-up t_vec
	t_down = N_down*t_vec.T # ramp-down t_vec

	#a = -mL:da:n+mR;
	da = 1./t_vec.len_T
	a = np.arange(-N_up, N_const+N_down+da, da)
	t = t_vec.T*a

	t_vec.len = t.size
	t_vec.dt = t_vec.T*da
	# define stroboscopic t_vec T
	# ind0 = find(a==-mL);
	ind0 = (a==-N_up).nonzero()[0]
	#indT = ind0:1/da:length(a);
	t_vec.indT = np.arange(ind0,a.size+1, 1.0/da).astype(int)
	#discrete stroboscopic t_vecs
	t_vec.T_n = t.take(t_vec.indT)
	#tend = nT(end-mR);
	t_end = t_vec.T_n[-N_down]

	t_vec.up = t[:t_vec.indT[N_up]]
	if t_vec.N_up > 0:
		t_vec.indT_up = t_vec.indT[:N_up]
		t_vec.T_up = t[t_vec.indT_up]

	if t_vec.N_down > 0 or t_vec.N_up > 0:
		t_vec.const = t[t_vec.indT[N_up]:t_vec.indT[N_up+N_const+1]]
		t_vec.indT_const = t_vec.indT[N_up:N_up+N_const+1]
		t_vec.T_const = t[t_vec.indT_const]

	if t_vec.N_down > 0:
		t_vec.down = t[t_vec.indT[N_up+N_const+1]:t_vec.indT[-1]]
		t_vec.indT_down = t_vec.indT[N_up+N_const+1:]
		t_vec.T_down = t[t_vec.indT_down]

	return t
t = t_vec(Omega,1)

### set up operator strings
# static
x_field=[[-hx,i] for i in xrange(L)]
z_field=[[-hz,i] for i in xrange(L)]
J_nn=[[-J,i,(i+1)%L] for i in xrange(L)]
# dynamic
drive_coupling=[[-A,i] for i in xrange(L)]

### build spin operator lists
static = [["zz",J_nn],["z",z_field], ["x",x_field]]
dynamic = [["x",drive_coupling,f,[Omega]]]


def symm_sector(kblock,pblock):

	################################################################
	##################   set up Hamiltonian    #####################
	################################################################

	# build spin basis
	basis = spin_basis_1d(L=L,kblock=kblock,pblock=pblock)


	# build spin Hamiltonian and operators
	H=hamiltonian(static,dynamic,L=L,dtype=np.float64,basis=basis,check_symm=False,check_herm=False)
	HF0=hamiltonian(static,[],L=L,dtype=np.float64,basis=basis,check_symm=False,check_herm=False)

	Ns = HF0.Ns
	
	################################################################
	################  calculate FLoquet operator  ##################
	################################################################

	Floq = Floquet(H,t_vec.T,VF=True,n_jobs=U_jobs)
	VF = Floq.VF
	EF = Floq.EF

	del Floq

	### diagonalise infinite-frequency Hamiltonian
	EF0, VF0 = HF0.eigh()

	################################################################
	###################  calculate observables  ####################
	################################################################
	#"""
	betavec = [1.0, 0.235]
	Diag_Ens = observables.Diag_Ens_Observables_old(L,VF0,EF0,VF,betavec=betavec,Sd_Renyi=True,Ed=True,deltaE=True,rho_d=True)

	psi0 = VF0[:,0]

	rho0 = np.outer(VF0[:,0].conj(), VF0[:,0]) #+ np.outer(VF0[:,1].conj(), VF0[:,1])

	#Diag_Ens2 = observables.Diag_Ens_Observables(L,{'V1':VF0,'E1':EF0,'f_args':[betavec],'V1_state':0},VF,Sd_Renyi=True,Obs=HF0,deltaObs=True,rho_d=True)
	#Diag_Ens2 = observables.Diag_Ens_Observables(L,rho0,VF,Sd_Renyi=True,Obs=HF0,deltaObs=True,rho_d=True)
	Diag_Ens2 = observables.Diag_Ens_Observables(L,psi0,VF,rho_d=True,Sent_Renyi=True,Sent_args=(basis))
	

	print "old", Diag_Ens
	print '---------------------'
	print "new", Diag_Ens2
	#s"""

	'''
	# calculate entanglement entropy of L/2 the chain
	v = observables.Entanglement_entropy2(L,VF0[:,0],basis=basis)

	v2,_ = observables.reshape_as_subsys({'V_rho':VF0,'rho_d':EF0},basis)
	#v2 = observables.reshape_as_subsys(VF0[:,0],basis)

	#"""
	print "++++++++++++"
	print v.shape
	print v
	print "-----------"
	print v2.shape
	print v2[0,:]
	#"""
	exit()
	'''

	'''
	Sent = observables.Entanglement_entropy2(L,VF0[:,0],basis=basis,DM=True)

	Sent_new = observables.Entanglement_Entropy({'V_rho':VF0,'rho_d':abs(EF0)/sum(abs(EF0))},basis,DM='both',svd_return_vec=[True,True,True])
	#Sent_new = observables.Entanglement_Entropy(VF0[:,0],basis)

	#"""
	print "++++++++++++"
	print Sent
	print "-----------"
	print Sent_new
	#"""
	exit()
	'''


sectors = [(0,-1)]
[sectors.append((kblock, 1)) for kblock in xrange( (L+2)/2)]


# parallel-loop over the symmetry sectors
#Data = np.vstack( Parallel(n_jobs=n_jobs)(delayed(symm_sector)(kblock,pblock) for kblock, pblock in sectors) )
Data = symm_sector(*sectors[block])


			