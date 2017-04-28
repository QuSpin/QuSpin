from __future__ import print_function, division

import sys,os
import argparse

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)


from quspin.operators import hamiltonian
from quspin.basis import boson_basis_1d
from quspin.tools.block_tools import block_ops
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
ladder:
-: t_par
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

L = 5
N = 2*L
Nb = L

t_par = 1.0
t_perp = 1.0
U = 1.0

basis = boson_basis_1d(N,Nb=Nb)

U_2 = [[U**2,i,i] for i in range(N)]
U_1 = [[-U,i] for i in range(N)]
t = [[t_par,i,(i+2)%N] for i in range(N)]
t.extend([[t_perp,i,i+1] for i in range(0,N,2)])




static = [["+-",t],["-+",t],["nn",U_2],["n",U_1]]
dynamic = []

no_checks = dict(check_herm=False,check_symm=False,check_pcon=False)
n = [hamiltonian([["n",[[1.0,i]]]],[],basis=basis,dtype=np.float64,**no_checks) for i in range(N)]


# setting up initial state
state = ["0" for i in range(N)]
state[L] = str(Nb)
state_str = "".join(state)

print("H-space size: {}, initial state: |{}>".format(basis.Ns,state_str))
i0 = basis.index(state_str)
psi = np.zeros(basis.Ns,dtype=np.float64)
psi[i0] = 1.0

blocks=[]

for kblock in range(L//2+1):
	blocks.append(dict(Nb=Nb,a=2,kblock=kblock,pblock=1))
	blocks.append(dict(Nb=Nb,a=2,kblock=kblock,pblock=-1))


U_block = block_ops(blocks,static,dynamic,boson_basis_1d,(N,),np.float64,get_proj_kwargs=dict(pcon=True))

psi_t = U_block.expm(psi,start=0,stop=1000,num=10000,iterate=True)
times = np.linspace(0,1000,num=10000)


import matplotlib.pyplot as plt
import matplotlib.animation as animation


psi = next(psi_t)
ns = np.array([n[i].expt_value(psi).real for i in range(N)])
ns = ns.reshape((-1,2))

fig = plt.figure()
ax = plt.gca()
im = ax.matshow(ns, animated=True,cmap='hot')


def updatefig(i):
	psi = next(psi_t)
	ns = np.array([n[j].expt_value(psi).real for j in range(N)])
	ns = ns.reshape((-1,2))
	im.set_array(ns)

	return im,

ani = animation.FuncAnimation(fig, updatefig, interval=50, blit=True)
plt.show()