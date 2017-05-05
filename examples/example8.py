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

L = 6
N = 2*L
nb = 0.5


t_par_1 = 1.0j
t_par_2 = -1.0j
t_perp =  0.6
U = 3.0
h=np.pi/2.0


basis = boson_basis_1d(N,nb=nb)

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

state = [("2000" if i%2 else "0200") for i in range(L//2)]
# setting up initial state
state_str = "".join(state)
i0 = basis.index(state_str)

psi = np.zeros(basis.Ns,dtype=np.float64)
psi[i0] = 1.0

state_str = np.array([int(s) for s in state_str])
ind_start = np.argwhere(state_str==2)

print("H-space size: {}, initial state: |{}>".format(basis.Ns,state_str))

blocks=[]

for kblock in range(L):
	blocks.append(dict(Nb=Nb,sps=sps,a=2,kblock=kblock))

U_block = block_ops(blocks,static,dynamic,boson_basis_1d,(N,),np.complex128,get_proj_kwargs=dict(pcon=True))

start,stop,num = 0,1000,30001
psi_t = U_block.expm(psi,start=start,stop=stop,num=num,iterate=True,block_diag=False)
times = np.linspace(start,stop,num)

n_eq = np.full((N,),float(Nb/N))

fig, (ax1,ax2) = plt.subplots(1,2)
fig.set_size_inches(10, 5)
line, = ax1.plot([], [], lw=2)
ax1.set_xlabel(r"$t/J$",fontsize=18)
ax1.set_ylabel(r"$n_{L}$",fontsize=18)
im = ax2.matshow(np.zeros((L,2)),cmap="hot",vmin=0,vmax=2)


ax1.grid()
xdata, ydata = [], []

def init():
	ax1.set_ylim(-0.1, 1.1)
	ax1.set_xlim(0, 5)


	del xdata[:]
	del ydata[:]
	line.set_data(xdata, ydata)
	im.set_data(np.zeros((L,2)))
	return im, line


def data_gen(t=0):
	for i,psi in enumerate(psi_t):
		ns = np.fromiter((n[j].expt_value(psi).real for j in range(N)),count=N,dtype=np.float64)
		n0 = ns[ind_start].sum()/Nb
		ns = ns.reshape((-1,2))

		yield times[i],n0,ns


def run(data):
	t,n0,ns = data

	xdata.append(t)
	ydata.append(n0)

	xmin, xmax = ax1.get_xlim()
	ymin, ymax = ax1.get_xlim()
	if t >= xmax:
		ax1.set_xlim(xmin, 2*xmax)
		ax1.figure.canvas.draw()

	line.set_data(xdata,ydata)
	im.set_data(ns)

	return im, line


ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=50,
								repeat=False, init_func=init)
plt.show()