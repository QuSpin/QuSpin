from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general
import numpy as np
import pickle

#np.set_printoptions(threshold=np.nan)

'''
N, Nup = 34, 4
basis=spin_basis_general(N,Nup=Nup,make_basis=False)
print(basis)

basis.make(N,50000,Nup)
print(basis)

exit()
'''

#
###### define model parameters ######
J1=1.0 # spin=spin interaction
J2=0.5 # magnetic field strength
Lx, Ly = 4, 4 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
#
###### setting up user-defined symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,....]
x = s%Lx # x positions for sites
y = s//Lx # y positions for sites

T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction

T_a = (x+1)%Lx + Lx*((y+1)%Ly) # translation along anti-diagonal
T_d = (x-1)%Lx + Lx*((y+1)%Ly) # translation along diagonal

R_a = np.rot90(s.reshape(Lx,Ly), axes=(0,1)).reshape(N_2d) # rotate anti-clockwise
R_c = np.rot90(s.reshape(Lx,Ly), axes=(1,0)).reshape(N_2d) # rotate clockwise

P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis

Z   = -(s+1) # spin inversion

#
###### setting up bases ######
basis_2d = spin_basis_general(N_2d, pauli=False,
									Nup=N_2d//2,
									kxblock=(T_x,0),kyblock=(T_y,0),
									rcblock=(R_c,0),rablock=(R_a,0),
									pxblock=(P_x,0),pyblock=(P_y,0),
									zblock=(Z,0)
								)

print(basis_2d)
exit()