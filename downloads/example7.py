from __future__ import print_function, division
import sys,os
# line 4 and line 5 below are for development purposes and can be removed
qspin_path = os.path.join(os.getcwd(),"../../")
sys.path.insert(0,qspin_path)
#####################################################################
#                            example 7                              #
#   In this script we demonstrate how to use QuSpin's to create	    #
#   a ladder geometry and study quanch dynamics in the Bose-Hubbard #
#   model. We also show how to compute the entanglement entropy of  #
#   bosonic systems. Last, we demonstrate how to use the block_ops  #
#   tool to decompose a state in the symemtry sectors of a          #
#   Hamiltonian, evolve the separate parts, and put back the state  #
#   in the end.                                                     #
#####################################################################
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import boson_basis_1d # bosonic Hilbert space
from quspin.tools.block_tools import block_ops # dynamics in symmetry blocks
import numpy as np # general math functions
import matplotlib.pyplot as plt # plotting library
import matplotlib.animation as animation # animating movie of dynamics
#
##### define model parameters
# initial seed for random number generator
np.random.seed(0) # seed is 0 to produce plots from QuSpin2 paper
# setting up parameters of simulation
L = 6 # length of chain
N = 2*L # number of sites
nb = 0.5 # density of bosons
sps = 3 # number of states per site
J_par_1 = 1.0 # top side of ladder hopping
J_par_2 = 1.0 # bottom side of ladder hopping
J_perp =  0.5 # rung hopping
U = 20.0 # Hubbard interaction
#
##### set up Hamiltonian and observables
# define site-coupling lists
int_list_1 = [[-0.5*U,i] for i in range(N)] # interaction $-U/2 \sum_i n_i$
int_list_2 = [[0.5*U,i,i] for i in range(N)] # interaction: $U/2 \num_i n_i^2$
# setting up hopping lists
hop_list = [[-J_par_1,i,(i+2)%N] for i in range(0,N,2)] # PBC bottom leg
hop_list.extend([[-J_par_2,i,(i+2)%N] for i in range(1,N,2)]) # PBC top leg
hop_list.extend([[-J_perp,i,i+1] for i in range(0,N,2)]) # perp/rung hopping
hop_list_hc = [[J.conjugate(),i,j] for J,i,j in hop_list] # add h.c. terms
# set up static and dynamic lists
static = [
			["+-",hop_list], # hopping
			["-+",hop_list_hc], # hopping h.c.
			["nn",int_list_2], # U n_i^2
			["n",int_list_1] # -U n_i
		]
dynamic = [] # no dynamic operators
# create block_ops object
blocks=[dict(kblock=kblock) for kblock in range(L)] # blocks to project on to
baisis_args = (N,) # boson_basis_1d manditory arguments
basis_kwargs = dict(nb=nb,sps=sps,a=2) # boson_basis_1d optional args
get_proj_kwargs = dict(pcon=True) # set projection to full particle basis
H_block = block_ops(blocks,static,dynamic,boson_basis_1d,baisis_args,np.complex128,
					basis_kwargs=basis_kwargs,get_proj_kwargs=get_proj_kwargs)
# setting up local Fock basis
basis = boson_basis_1d(N,nb=nb,sps=sps)
# setting up observables
no_checks = dict(check_herm=False,check_symm=False,check_pcon=False)
n_list = [hamiltonian([["n",[[1.0,i]]]],[],basis=basis,dtype=np.float64,**no_checks) for i in range(N)]
##### do time evolution
# set up initial state
i0 = np.random.randint(basis.Ns) # pick random state from basis set
psi = np.zeros(basis.Ns,dtype=np.float64)
psi[i0] = 1.0
# print info about setup
state_str = "".join(str(int((basis[i0]//basis.sps**(L-i-1)))%basis.sps) for i in range(N))
print("total H-space size: {}, initial state: |{}>".format(basis.Ns,state_str))
# setting up parameters for evolution
start,stop,num = 0,30,301 # 0.1 equally spaced points
times = np.linspace(start,stop,num)
# calculating the evolved states
n_jobs = 1 # paralelisation: increase to see if calculation runs faster!
psi_t = H_block.expm(psi,a=-1j,start=start,stop=stop,num=num,block_diag=False,n_jobs=n_jobs)
# calculating the local densities as a function of time
expt_n_t = np.vstack([n.expt_value(psi_t).real for n in n_list]).T
# reshape data for plotting
n_t = np.zeros((num,2,L))
n_t[:,0,:] = expt_n_t[:,0::2]
n_t[:,1,:] = expt_n_t[:,1::2]
# calculating entanglement entropy 
sub_sys_A = range(0,N,2) # bottom side of ladder 
gen = (basis.ent_entropy(psi,sub_sys_A=sub_sys_A)["Sent_A"] for psi in psi_t.T[:])
ent_t = np.fromiter(gen,dtype=np.float64,count=num)
# plotting static figures
#"""
fig, ax = plt.subplots(nrows=5,ncols=1)
im=[]
im_ind = []
for i,t in enumerate(np.logspace(-1,np.log10(stop-1),5,base=10)):
	j = times.searchsorted(t)
	im_ind.append(j)
	im.append(ax[i].imshow(n_t[j],cmap="hot",vmax=n_t.max(),vmin=0))
	ax[i].tick_params(labelbottom=False,labelleft=False)
cax = fig.add_axes([0.85, 0.1, 0.03, 0.8])
fig.colorbar(im[2],cax)
plt.savefig("boson_density.pdf")
plt.figure()
plt.plot(times,ent_t,lw=2)
plt.plot(times[im_ind],ent_t[im_ind],marker="o",linestyle="",color="red")
plt.xlabel("$Jt$",fontsize=20)
plt.ylabel("$s_\mathrm{ent}(t)$",fontsize=20)
plt.grid()
plt.savefig("boson_entropy.pdf")
#plt.show()
plt.close()
#"""
# setting up two plots to animate side by side
fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(10, 5)
ax1.set_xlabel(r"$Jt$",fontsize=18)
ax1.set_ylabel(r"$s_\mathrm{ent}$",fontsize=18)
ax1.grid()
line1, = ax1.plot(times, ent_t, lw=2)
line1.set_data([],[])
im = ax2.matshow(n_t[0],cmap="hot")
fig.colorbar(im)
def run(i): # function to update frame
	# set new data for plots
	line1.set_data(times[:i],ent_t[:i])
	im.set_data(n_t[i])
	return im, line1
# CAUTION: these are many animations
ani = animation.FuncAnimation(fig, run, range(num),interval=50)
plt.show()
plt.close()
#
""" 
###### ladder lattice 
# hopping coupling parameters:
# - : J_par_1
# = : J_par_2
# | : J_perp
#
# lattice graph
#
 = 1 = 3 = 5 = 7 = 9 =
   |   |   |   |   |
 - 0 - 2 - 4 - 6 - 8 -
#
# translations along leg-direction (i -> i+2):
#
 = 9 = 1 = 3 = 5 = 7 =
 |   |   |   |   | 
 - 8 - 0 - 2 - 4 - 6 -
#
# if J_par_1=J_par_2, one can use regular chain parity (i -> N - i) as combination 
# of the two ladder parity operators:
#
 - 8 - 6 - 4 - 2 - 0 - 
 |   |   |   |   | 
 - 9 - 7 - 5 - 3 - 1 -
"""