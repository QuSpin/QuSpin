from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
import numpy as np
import scipy.sparse as sp

from quspin.operators._make_hamiltonian import _consolidate_static


#
###### define model parameters ######
J1=1.0 # spin=spin interaction
J2=0.5 # magnetic field strength
Lx, Ly = 4, 4 # linear dimension of 2d lattice
N_2d = Lx*Ly # number of sites
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites

T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction

T_a = (x+1)%Lx + Lx*((y+1)%Ly) # translation along anti-diagonal
T_d = (x-1)%Lx + Lx*((y+1)%Ly) # translation along diagonal

R = np.rot90(s.reshape(Lx,Ly), axes=(0,1)).reshape(N_2d) # rotate anti-clockwise

P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis

Z   = -(s+1) # spin inversion

#
###### setting up bases ######
basis_2d = spin_basis_general(N_2d, pauli=False, make_basis=True,
									Nup=N_2d//2,
									kxblock=(T_x,0),kyblock=(T_y,0),
									rblock=(R,0),
									pxblock=(P_x,0),pyblock=(P_y,0),
									zblock=(Z,0)
								)


###### setting up hamiltonian ######
# setting up site-coupling lists
J1_list=[[J1,i,T_x[i]] for i in range(N_2d)] + [[J1,i,T_y[i]] for i in range(N_2d)]
J2_list=[[J2,i,T_d[i]] for i in range(N_2d)] + [[J2,i,T_a[i]] for i in range(N_2d)]
#
static=[ ["xx",J1_list],["yy",J1_list],["zz",J1_list],  
		 ["xx",J2_list],["yy",J2_list],["zz",J2_list]
		]


static_list = _consolidate_static(static)

H_new=sp.csr_matrix((basis_2d.Ns,basis_2d.Ns),dtype=np.float64)

for opstr,indx,J in static_list:
	
	ME,ket,bra = basis_2d._Op_int_state(opstr,indx,J,basis_2d.states,dtype=np.float64)


	#print(ME)

	#print(np.searchsorted(ket,basis_2d.states))

	print(np.unique(ket))
	print()
	print(basis_2d.states)
	#print(bra)

	exit()
	
exit()

# build hamiltonian
H=hamiltonian(static,[],basis=basis_2d,dtype=np.float64)




