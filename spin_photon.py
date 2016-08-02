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

Nph_tot = int(sys.argv[4]) #120 #total number of photon states
Nph = Nph_tot/2 # mean number of photons in initial state

### static model params
J = 1.0
hx = 0.809
hz = 0.9045

### dynamic model params
Omega = float(sys.argv[2])
A = 1.0
T = 2*np.pi/Omega
'''
def f(t,Omega):
	return np.cos(Omega*t)
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
t = t_vec(Omega,len_T=10)
'''

### set up operator strings
# spin
x_field=[[-hx,i] for i in xrange(L)]
z_field=[[-hz,i] for i in xrange(L)]
J_nn=[[-J,i,(i+1)%L] for i in xrange(L)]
# spin-photon
absorb=[[-0.5*A/np.sqrt(Nph),i] for i in xrange(L)]
emit=[[-0.5*np.conj(A)/np.sqrt(Nph),i] for i in xrange(L)]
# photon
ph_energy = [[Omega/L,i] for i in xrange(L)]

### build spin-photon operator lists
static = [["zz|",J_nn], ["z|",z_field], ["x|",x_field], ["x|-",absorb], ["x|+",emit], ["I|n",ph_energy]]
H_ph_list = [["I|n",ph_energy]]
N_op_list = [["I|n",[[1.0/L,i] for i in xrange(L)] ]]
# spin-only operator list
static_sp = [["zz",J_nn],["z",z_field], ["x",x_field]]



def symm_sector(kblock,pblock):

	################################################################
	##################   set up Hamiltonian    #####################
	################################################################

	# build spin-photon basis
	basis = photon_basis(spin_basis_1d,L=L,kblock=kblock,pblock=pblock,Nph=Nph_tot)
	# build spin basis
	basis_sp = spin_basis_1d(L=L,kblock=kblock,pblock=pblock)


	# build spin-photon Hamiltonian and operators
	H=hamiltonian(static,[],L=L,dtype=np.float64,basis=basis,check_symm=False,check_herm=False)
	H_ph=hamiltonian(H_ph_list,[],L=L,dtype=np.float64,basis=basis,check_symm=False,check_herm=False)
	N_op = hamiltonian(N_op_list,[],L=L,dtype=np.float64,basis=basis,check_symm=False,check_herm=False)
	# build spin Hamiltonian and operators
	HF0_sp=hamiltonian(static_sp,[],L=L,dtype=np.float64,basis=basis_sp,check_symm=False,check_herm=False)
	
	Ns = H.Ns
	print "spin (total, spin) H-space sizes are", (Ns, HF0_sp.Ns)

	################################################################
	################  calculate initial state  #####################
	################################################################

	# get GS of spin chain
	#EF0_sp, psiF0_sp = HF0_sp.eigsh(k=1)
	EF0_sp, VF0_sp = HF0_sp.eigh()
	psiF0_sp = VF0_sp[:,0]
	# get coherent state of photon mode
	psi0_ph = observables.coherent_state(np.sqrt(Nph),Nph_tot+1)
	print 'Norm of coherent state is:', np.linalg.norm(psi0_ph)
	psi0_ph *= 1.0/np.linalg.norm(psi0_ph)
	# calculate total initial state
	psi0 = np.kron(psiF0_sp,psi0_ph)
	#print 'Norm of initial state is:', np.linalg.norm(psi0)
	print 'Energy of initial state is', H_ph.matrix_ele(psi0,psi0), EF0_sp[-1] - EF0_sp[0], H_ph.matrix_ele(psi0,psi0) - (EF0_sp[-1] - EF0_sp[0])

	print psi0.conj().T.dot( N_op.tocsr().dot(psi0) ), Nph
	print psi0.conj().T.dot( N_op.tocsr().dot( N_op.tocsr().dot(psi0) ) )  - psi0.conj().T.dot( N_op.tocsr().dot(psi0) )**2, Nph



	### diagonalise spin-photon Hamiltonian
	E, V = H.eigh()

	################################################################
	###################  calculate observables  ####################
	################################################################

	# calculate Diag Ensemble quantities of full system
	Diag_Ens = observables.Diag_Ens_Observables(L,psi0,V,rho_d=True,densities=False,Obs=N_op,delta_t_Obs=True)
	
	N_op_d = Diag_Ens['Obs_pure']
	delta_t_N_op_d = Diag_Ens['delta_t_Obs_pure']
	delta_q_N_op_d = Diag_Ens['delta_q_Obs_pure']
	# get diagonal DM
	rho_d = Diag_Ens['rho_d']


	# calculate reduced DM of spin chain from diagonal DM
	Chain = observables.Entanglement_Entropy({'V_rho':V,'rho_d':rho_d},basis,DM='chain_subsys')
	Sent_spins = Chain['Sent']
	rho_d_sp = Chain['DM_chain_subsys']

	# calculate infinite time expectation of spin chain
	
	Sent_sp = observables.Entanglement_Entropy(rho_d_sp,basis_sp)['Sent']
	S_Tinf_sp = np.log(2)
	Ed_sp = np.trace(HF0_sp.dot(rho_d_sp))/L
	E_Tinf_sp = sum(EF0_sp)/L
	deltaE_sp = np.sqrt( np.trace(HF0_sp.dot(HF0_sp.tocsr()).dot(rho_d_sp))/L**2 - Ed_sp**2 )


	# calculate entanglement entropy of HF0 GS.
	Diag_Ens_sp = observables.Diag_Ens_Observables(L,rho_d_sp,VF0_sp,Sent_Renyi=True,Sent_args={'basis':basis_sp})
	Sent_sp_subsys = Diag_Ens_sp['Sent_DM']
	Sent_sp_subsys_0 = observables.Entanglement_Entropy(psiF0_sp,basis_sp)['Sent']
	
	
	# calculate normalised Q quantities
	Q_E_sp = (Ed_sp - EF0_sp[0]/L)/(E_Tinf_sp- EF0_sp[0]/L) 
	Q_Sent_sp = Sent_sp/S_Tinf_sp
	Q_Sent_subsys = (Sent_sp_subsys - Sent_sp_subsys_0)/(S_Tinf_sp - Sent_sp_subsys_0)


	# calculate mean level spacing of HF and HF0
	folded_E = sorted( np.real( 1j/T*np.log(np.exp(-1j*E*T)) ) )
	rave_folded = observables.Mean_Level_Spacing( folded_E )
	rave = observables.Mean_Level_Spacing(E)

	exit()

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

br
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
			