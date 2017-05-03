from __future__ import print_function, division

import sys,os
import argparse

# qspin_path = os.path.join(os.getcwd(),"../")
# sys.path.insert(0,qspin_path)


from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from quspin.tools.block_tools import block_ops
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
ladder:
-: t_par_1
-: t_par_2
|: t_perp

-1-3-5-7-9-
 | | | | |
-0-2-4-6-8-

translations (i -> i+2):

-9-1-3-5-7-
 | | | | | 
-8-0-2-4-6-


parity (i -> N - i):

-8-6-4-2-0-
 | | | | |
-9-7-5-3-1-

"""

L = 20
N = 2*L
Nb = 2
sps = Nb+1

t_par_1 = -1.0j
t_par_2 = 1.0j
t_perp = -1.0
U = 1.0
h=np.pi/2.0


basis = boson_basis_1d(N,Nb=Nb,sps=sps)

U_2 = [[U**2,i,i] for i in range(N)]
U_1 = [[-U,i] for i in range(N)]

t = [[t_par_1,i,(i+2)%N] for i in range(0,N,2)]
t.extend([[t_par_2,i,(i+2)%N] for i in range(1,N,2)])
t.extend([[t_perp,i,i+1] for i in range(0,N,2)])

t_hc = [[J.conjugate(),i,j] for J,i,j in t]




static = [["+-",t],["-+",t_hc],["nn",U_2],["n",U_1]]
dynamic = []

no_checks = dict(check_herm=False,check_symm=False,check_pcon=False)
n = [hamiltonian([["n",[[1.0,i]]]],[],basis=basis,dtype=np.float64,**no_checks) for i in range(N)]


# setting up initial state
state = [("10" if i%2 else "01") for i in range(L)]
state = ["0" for i in range(N)]
state[L] = str(Nb)

state_str = "".join(state)
i0 = basis.index(state_str)

psi = np.zeros(basis.Ns,dtype=np.float64)
psi[i0] = 1.0



print("H-space size: {}, initial state: |{}>".format(basis.Ns,state_str))

blocks=[]

for kblock in range(L):
	blocks.append(dict(Nb=Nb,sps=sps,a=2,kblock=kblock))

U_block = block_ops(blocks,static,dynamic,boson_basis_1d,(N,),np.complex128,get_proj_kwargs=dict(pcon=True))

psi_t = U_block.expm(psi,start=0,stop=1000,num=100000,iterate=True,block_diag=False)
times = np.linspace(0,100,1001)


psi = next(psi_t)
ns = np.array([n[i].expt_value(psi).real for i in range(N)])
n0 = ns[L]
ns = ns.reshape((-1,2))

fig = plt.figure()
ax = plt.gca()
im = ax.matshow(ns,cmap='hot')
fig.colorbar(im)
time_text = ax.text(-L//2,L//2,"$t= {:.2f}$\n$n_{{ {:d} }}={:.2f}$".format(times[0],L,n0))



def updatefig(i):
	psi = next(psi_t)
	ns = np.array([n[j].expt_value(psi).real for j in range(N)])
	n0 = ns[L]
	ns = ns.reshape((-1,2))

	im.set_array(ns)

	st = "$t= {:.2f}$\n$n_{{ {:d} }}={:.2f}$".format(times[i+1],L,n0)
	time_text.set_text(st)

	return im, time_text


ani = animation.FuncAnimation(fig, updatefig, interval=50)
plt.show()