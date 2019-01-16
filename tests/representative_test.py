from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_general, spinless_fermion_basis_general
import numpy as np



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

R = np.rot90(s.reshape(Lx,Ly), axes=(0,1)).reshape(N_2d) # rotate

P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis

Z   = -(s+1) # spin inversion

#
###### setting up bases ######
#'''
basis_2d = spin_basis_general(N_2d, pauli=False, make_basis=False,
									Nup=N_2d//2,
									kxblock=(T_x,0),kyblock=(T_y,0),
									rblock=(R,0),
									pxblock=(P_x,0),pyblock=(P_y,0),
									zblock=(Z,0)
								)

basis_2d_full = spin_basis_general(N_2d, pauli=False, make_basis=True,
									Nup=N_2d//2,
								)
'''

basis_2d = spinless_fermion_basis_general(N_2d, make_basis=False,
									Nf=N_2d//2,
									kxblock=(T_x,0),kyblock=(T_y,0),
									rblock=(R,0),
									pxblock=(P_x,0),pyblock=(P_y,0),
								)

basis_2d_full = spinless_fermion_basis_general(N_2d, make_basis=True,
									Nf=N_2d//2,
								)
'''

# grab states of full basis
states=basis_2d_full.states

# check function
ref_states=basis_2d.representative(states)
ref_states=np.sort( np.unique(ref_states) )[::-1]

# check inplace function
ref_states_inplace=np.zeros_like(states)
basis_2d.representative(states,out=ref_states_inplace)
ref_states_inplace=np.sort( np.unique(ref_states_inplace) )[::-1]

# make full basis to compare to
basis_2d.make(Ns_block_est=20000)


np.testing.assert_allclose(basis_2d.states - ref_states,0.0,atol=1E-5,err_msg='failed representative test!')
np.testing.assert_allclose(basis_2d.states - ref_states_inplace,0.0,atol=1E-5,err_msg='failed inplace representative test!')


# check g_out and sign_out flags
ref_states,g_out,sign_out=basis_2d.representative(states,return_g=True,return_sign=True)
ref_states,g_out=basis_2d.representative(states,return_g=True)
ref_states,sign_out=basis_2d.representative(states,return_sign=True)

r_out=np.zeros_like(states)
g_out,sign_out=basis_2d.representative(states,out=r_out,return_g=True,return_sign=True)
g_out=basis_2d.representative(states,out=r_out,return_g=True)
sign_out=basis_2d.representative(states,out=r_out,return_sign=True)

