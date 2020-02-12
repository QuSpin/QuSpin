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
#                            example 201                              #	
# This example shows how to use the `Lanczos` submodule of the        #
# `tools` module to compute finite temperature expecation values      #
# using FTLM_statc_iteration and LTLM_statiic_iteration.              #
#######################################################################
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian,quantum_operator
from quspin.tools.lanczos import lanczos_full,FTLM_static_iteration,LTLM_static_iteration
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def bootstrap_mean(N,D,n_bs=100):
	# uses boostraping to esimate the error. 
    N = np.asarray(N)
    D = np.asarray(D)

    avg = np.nanmean(N,axis=0)/np.nanmean(D,axis=0)
    n_D = D.shape[0]
    n_N = N.shape[0]

    i_iter = (np.random.randint(n_D,size=n_D) for i in range(n_bs))

    bs_iter = (np.nanmean(N[i,...],axis=0)/np.nanmean(D[i,...],axis=0) for i in i_iter)
    diff_iter = ((bs-avg)**2 for bs in bs_iter)
    err = np.sqrt(sum(diff_iter)/n_bs)

    return avg,err
#
def get_operators(L):
	# generates hamiltonian for TFIM, see quspin papers to learn more aabout this
	basis = spin_basis_1d(L,pauli=True)

	J_list = [[-1.0,i,(i+1)%L] for i in range(L)]
	h_list = [[-1.0,i] for i in range(L)]
	M_list = [[1.0/L,i] for i in range(L)]

	ops_dict = dict(J=[["zz",J_list]],h=[["x",h_list]])
	M = hamiltonian([["z",M_list]],[],basis=basis,dtype=np.float64)
	M2 = M**2
	H = quantum_operator(ops_dict,basis=basis,dtype=np.float64)

	return M2,H
#
class lanczos_wrapper(object):
	# class that contains minimum requirments to use Lanczos 
	def __init__(self,A,**kwargs):
		self._A = A
		self._kwargs = kwargs

	def dot(self,v,out=None):
		# calls the `dot` method of quantum_operator 
		# with the parameters fixed to a given value.
		return self._A.dot(v,out=out,pars=self._kwargs)

	@property
	def dtype(self):
		# dtype required to figure out result types in lanczos calculations.
		return self._A.dtype
#
np.random.seed(0)
#
##### define parameters #####
#
L = 10 
nv = 50
s = 0.6
nsample = 100
T = np.logspace(-3,3,51,base=10)
beta = 1.0/(T+1e-15)
#
##### get operators #####
#
M2,H = get_operators(L)
# crate wrapper for quantum_operator
H_w = lanczos_wrapper(H,J=s,h=(1-s))
# calculate ground state energy to use as shift that will prevent overflows 
[E0] = H.eigsh(k=1,which="SA",pars=dict(J=s,h=1-s),return_eigenvectors=False)
#
##### finite temperature methods #####
# 
# lists to store results from iterations
M2_FT_list = []
M2_LT_list = []
Z_FT_list = []
Z_LT_list = []
#
# allocate memory for lanczos vectors
out = np.zeros((nv,H.Ns),dtype=np.float64)
# calculate iterations
for i in range(nsample):
	# generate random vector
	r = np.random.normal(0,1,size=H.Ns)
	r /= np.linalg.norm(r)
	# get lanczos basis
	E,V,lv = lanczos_full(H_w,r,nv,eps=1e-8,full_ortho=True)
	# shift energy to avoid overflows
	E -= E0
	# calculate iteration
	results_FT,Z_FT = FTLM_static_iteration({"M2":M2},E,V,lv,beta=beta)
	results_LT,Z_LT = LTLM_static_iteration({"M2":M2},E,V,lv,beta=beta)
	# save results to a list
	M2_FT_list.append(results_FT["M2"])
	Z_FT_list.append(Z_FT)
	M2_LT_list.append(results_LT["M2"])
	Z_LT_list.append(Z_LT)
#
# calculating error bars
m2_FT,dm2_FT = bootstrap_mean(M2_FT_list,Z_FT_list)
m2_LT,dm2_LT = bootstrap_mean(M2_LT_list,Z_LT_list)
#
##### plotting results #####
#
# setting up plot and inset
h=3.2
f,ax = plt.subplots(figsize=(1.5*h,h))
axinset = inset_axes(ax, width="45%", height="65%", loc="upper right")
axs = [ax,axinset]
#
# plot results for FTLM and LTLM.
for a in axs:
	a.errorbar(T,m2_LT,dm2_LT,marker=".",label="LTLM",zorder=-1)
	a.errorbar(T,m2_FT,dm2_FT,marker=".",label="FTLM",zorder=-2)
#
if H.Ns < 2000: # hilbert space is not too big to diagonalize
	#
	##### calculating exact results from full diagonalization ####
	#
	# adding more points for smooth line
	T_new = np.logspace(np.log10(T.min()),np.log10(T.max()),10*len(T))
	beta_new = 1.0/(T_new+1e-15)

	# full diagonaization of H
	E,V = H.eigh(pars=dict(J=s,h=1-s))
	# shift energy to avoid overflows
	E -= E[0]
	# get boltzmann weights for each temperature
	W = np.exp(-np.outer(E,beta_new))
	# get diagonal matrix elements for trace
	O = M2.matrix_ele(V,V,diagonal=True) 
	# calculate trace
	O = np.einsum("j...,j->...",W,O)/np.einsum("j...->...",W)
	# plot results
	for a in axs:
		a.plot(T_new,O,label="exact",zorder=0)
#
# max axis log-scale along x-axis
for a in axs:
	a.set_xscale("log")
#
# adding space for inset by expanding x limits.
xmin,xmax = ax.get_xlim()
ax.set_xlim((xmin,10*xmax))
ax.legend(loc="lower left")
#
# inset adjustment to zoom in low-temp limit.
xmin,xmax = axinset.get_xlim()
#
a = -0.6
m = np.logical_and(T>=xmin,T<=10**(a))
axinset.set_xlim((xmin,10**(a+0.1)))
ymin = min(m2_LT[m].min(),m2_FT[m].min())
ymax = max(m2_LT[m].max(),m2_FT[m].max())
ywin = ymax-ymin
boundy = 0.1*ywin
axinset.set_ylim((ymin-boundy,ymax+boundy))
#
# display plot
f.tight_layout()
plt.show()
