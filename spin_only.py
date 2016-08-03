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

import time
import sys
import os

####################################################################
start_time = time.time()
####################################################################

# parallelisation params
U_jobs = 1
n_jobs = 1

# system size
L = int(sys.argv[1])
block = int(sys.argv[3]) - 1

### static model params
J = 1.0
hx = 0.809
hz = 0.9045

### dynamic model params
def f(t,Omega):
	return np.cos(Omega*t)
Omega = int(sys.argv[2])
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
	print "spin H-space size is", Ns

	################################################################
	################  calculate Floquet operator  ##################
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

	# calculate Diag Ensemble quantitties
	Diag_Ens = observables.Diag_Ens_Observables(L,VF0[:,0],VF,Sent_Renyi=True,Sd_Renyi=True,Obs=HF0,delta_t_Obs=True,Sent_args={'basis':basis})
	
	
	Sd = Diag_Ens['Sd_pure']
	Sent = Diag_Ens['Sent_pure']
	S_Tinf = np.log(2)
	Ed = Diag_Ens['Obs_pure']
	E_Tinf = sum(EF0)/Ns/L
	deltaE = Diag_Ens['delta_t_Obs_pure']

	# calculate finite-temperater diag ensemble values
	beta = [0.005, 1.0, 20.0]
	Diag_Ens_T = observables.Diag_Ens_Observables(L,{'V1':VF0,'E1':EF0,'f_args':[beta],'f_norm':False},VF,Sd_Renyi=True)
	
	print 'print keys:', Diag_Ens_T

	# read off Floquet diagonal entropy for the three values of 'beta'
	Sd_T = Diag_Ens_T['Sd_thermal']

	print 'pure', Sd
	print 'thermal', Sd_T

	exit()


	# calculate entanglement entropy of HF0 GS
	Sent0 = observables.Entanglement_Entropy(VF0[:,0],basis)['Sent']
	
	# calculate normalised Q quantities
	Q_E = (Ed - EF0[0]/L)/(E_Tinf- EF0[0]/L) 
	Q_SF = Sd/S_Tinf
	Q_Sent = (Sent - Sent0)/(S_Tinf - Sent0)


	# calculate mean level spacing of HF and HF0
	folded_EF0 = sorted( np.real( 1j/t_vec.T*np.log(np.exp(-1j*EF0*t_vec.T)) ) )
	rave_F0 = observables.Mean_Level_Spacing( folded_EF0 )
	rave_F = observables.Mean_Level_Spacing(EF)


	################################################################
	###################     store data        ######################
	################################################################

	data = np.zeros((17,),dtype=np.float64)

	data[0] = Q_E
	data[1] = Q_SF
	data[2] = Q_Sent
	data[3] = deltaE
	data[4] = rave_F
	data[5] = rave_F0
	data[6] = Ed
	data[7] = Sd
	data[8] = Sent
	data[9] = E_Tinf
	data[10] = S_Tinf
	data[11] = Sent0
	data[12] = EF0[0]/L
	data[13]= (EF0[-1]-EF0[0])/L # MB bandwdith per site
	data[14]= Ns
	data[15]= kblock
	data[16]= pblock

	return data


# define list will all symmetry sectors
#sectors = [(0,-1)]
#[sectors.append((kblock, 1)) for kblock in xrange(int(np.ceil((L+1)/2.0)))]

sectors = [(0,-1)]
[sectors.append((kblock, 1)) for kblock in xrange( (L+2)/2)]


# parallel-loop over the symmetry sectors
#Data = np.vstack( Parallel(n_jobs=n_jobs)(delayed(symm_sector)(kblock,pblock) for kblock, pblock in sectors) )
Data = symm_sector(*sectors[block])


################################################################
###################     save data        #######################
################################################################

# file name
save_params = (L,) + tuple(np.around([Omega,A],2)) + sectors[block]
data_name = "data_driven_chain_L=%s_Omega=%s_A=%s_kblock=%s_pblock=%s.txt" %(save_params)

# display full strings
np.set_printoptions(threshold='nan')


# read in local directory path
str1=os.getcwd()
str2=str1.split('\\')
n=len(str2)
my_dir = str2[n-1]

# define save data directory
save_dir = "%s/data" %(my_dir)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#change path to save directory
os.chdir(save_dir)

# save to .txt
np.savetxt(data_name, Data, delimiter=" ", fmt="%s")

print "Calculation took",("--- %s seconds ---" % (time.time() - start_time))


"""
import matplotlib.pyplot as plt
import pylab

#psi = H.evolve(psi0,t[-1],t,rtol=1E-12,atol=1E-12)

# calculate energy of spin chain
Energy = np.real( np.einsum("ij,jk,ik->i", psi.conj(),HF0.todense(),psi) )

# plot results
title_params = tuple(np.around([hz/J,hx/J,A/Omega,Omega/J],2) ) + (L,)
titlestr = "$h_z/J=%s,\\ h_x/J=%s,\\ A/\\Omega=%s,\\ \\Omega/J=%s,\\ L=%s$" %(title_params)

plt.plot(t,Energy/L,'r--',linewidth=2)

plt.xlabel('$t/T$', fontsize=18)
plt.ylabel('$\\eta(t)$', fontsize=20)


plt.legend(loc='upper right')
plt.title(titlestr, fontsize=18)
plt.tick_params(labelsize=16)
plt.grid(True)


plt.show()

"""
			