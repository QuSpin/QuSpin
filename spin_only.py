from exact_diag_py.tools import observables

from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la

from numpy.linalg import norm
from numpy.random import random,seed

import matplotlib.pyplot as plt
import pylab

import time
import sys
import os

####################################################################
start_time = time.time()
####################################################################

L = 6

J = 1.0
hx = 0.809
hz = 0.9045


def f(t,Omega):
	return 2*np.cos(Omega*t)
Omega = 16.0
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
t = t_vec(Omega,11)


x_field=[[hx,i] for i in xrange(L)]
z_field=[[hz,i] for i in xrange(L)]
J_nn=[[-J,i,(i+1)%L] for i in xrange(L)]

drive_coupling=[[A,i] for i in xrange(L)]

### build spin operator lists

static = [["zz",J_nn],["z",z_field], ["x",x_field]]
dynamic = [["x",drive_coupling,f,[Omega]]]

# build spin basis
basis = spin_basis_1d(L=L,kblock=0,pblock=1)


# build spin Hamiltonian and operators
H=hamiltonian(static,dynamic,L=L,dtype=np.float64,basis=basis)
HF0=hamiltonian(static,[],L=L,dtype=np.float64,basis=basis)

Ns = HF0.Ns
print "spin H-space size is", Ns


######### calculate FLoquet operator

# this function evolves the ith local basis state with Hamiltonian H
# this is used to construct the stroboscpoic evolution operator
def evolve(i,H,T):
	from numpy import zeros
	from scipy.integrate import complex_ode

	nsteps=sum([2**_i for _i in xrange(32,63)]) # huge number to make sure solver is successful.
	psi0=zeros((Ns,),dtype=np.complex128); psi0[i]=1.0
	solver=complex_ode(H._hamiltonian__SO)
	solver.set_integrator('dop853', atol=1E-12,rtol=1E-12,nsteps=nsteps) 
	solver.set_initial_value(psi0,t=0.0)
	solver.integrate(T)
	if solver.successful():
		return solver.y
	else:
		raise Exception('failed to integrate')
### USING JOBLIB ###
def get_U(H,n_jobs,T): 
	from joblib import delayed,Parallel
	from numpy import vstack # or hstack, I can't remember

	sols=Parallel(n_jobs=n_jobs)(delayed(evolve)(i,H,T) for i in xrange(H.Ns))

	return vstack(sols)

n_jobs = 2
UF = get_U(H,n_jobs,t_vec.T)
# find Floquet states and phases
thetaF, VF = la.eig(UF)
# calculate and order q'energies
EF = np.real( 1j/t_vec.T*np.log(thetaF) )
ind_EF = np.argsort(EF)
EF = EF[ind_EF]
VF = VF[:,ind_EF]

del ind_EF, UF


### diagonalise Hamiltonian
EF0, VF0 = HF0.eigh()
print "MB band width per site is", (EF0[-1]-EF0[0])/L


Diag_Ens = observables.Diag_Ens_Observables(L,VF,EF,VF0,Sd_Renyi=True,Ed=True,deltaE=True,rho_d=True,state=Ns)

print Diag_Ens.keys()
print Diag_Ens['rho_d'].sum()

Sd = Diag_Ens['Sd_Renyi_state']
S_Tinf = Diag_Ens['S_Tinf']
Ed = Diag_Ens['Ed_state']
E_Tinf = Diag_Ens['E_Tinf']
deltaE = Diag_Ens['deltaE_state']


Q = (Ed - EF[0]/L)/(E_Tinf- EF[0]/L) 
S = Sd/S_Tinf

print Q, S

print "Calculation took",("--- %s seconds ---" % (time.time() - start_time))


exit()


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


			