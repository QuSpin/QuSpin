from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian
from quspin.basis import spin_basis_general, boson_basis_general, spinless_fermion_basis_general, spinful_fermion_basis_general
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
#P_d = y + Lx*x

P_x = x + Lx*(Ly-y-1) # reflection about x-axis
P_y = (Lx-x-1) + Lx*y # reflection about y-axis

Z   = -(s+1) # spin inversion

#per_factor=Lx*Ly*2*2*2*2

J_p=[[np.sqrt(7),i,T_x[i]] for i in range(N_2d)] + [[np.sqrt(7),i,T_y[i]] for i in range(N_2d)]
J_n=[[-np.sqrt(7),i,T_x[i]] for i in range(N_2d)] + [[-np.sqrt(7),i,T_y[i]] for i in range(N_2d)]


for q in [0,Lx-1]: # loop only over quantum numbers 0, pi so all symmetries commute

	###### setting up bases ######

	basis_boson = boson_basis_general(N_2d, make_basis=False,
										Nb=N_2d//4,sps=2,
										kxblock=(T_x,q),kyblock=(T_y,q),
										rblock=(R,q),
										pxblock=(P_x,q),pyblock=(P_y,q),
									)

	basis_boson_full = boson_basis_general(N_2d, make_basis=False,
										Nb=N_2d//4,sps=2,
									)


	basis_spin = spin_basis_general(N_2d, pauli=False, make_basis=False,
										Nup=N_2d//2,
										kxblock=(T_x,q),kyblock=(T_y,q),
										rblock=(R,q),
										pxblock=(P_x,q),pyblock=(P_y,q),
										zblock=(Z,q)
									)

	basis_spin_full = spin_basis_general(N_2d, pauli=False, make_basis=False,
										Nup=N_2d//2,
									)

	basis_fermion = spinless_fermion_basis_general(N_2d, make_basis=False,
										Nf=N_2d//2,
										kxblock=(T_x,q),kyblock=(T_y,q),
										rblock=(R,q),
										pxblock=(P_x,q),pyblock=(P_y,q),
									)

	basis_fermion_full = spinless_fermion_basis_general(N_2d, make_basis=False,
										Nf=N_2d//2,
									)



	basis_spinful_fermion = spinful_fermion_basis_general(N_2d, make_basis=False,
										Nf=(N_2d//8,N_2d//8),
										kxblock=(T_x,0),kyblock=(T_y,0),
										rblock=(R,0),
										pxblock=(P_x,0),pyblock=(P_y,0),
									)

	basis_spinful_fermion_full = spinful_fermion_basis_general(N_2d, make_basis=False,
										Nf=(N_2d//8,N_2d//8),
									)


	bases_2d=[basis_boson,basis_spin,basis_fermion,basis_spinful_fermion]
	bases_2d_full=[basis_boson_full,basis_spin_full,basis_fermion_full,basis_spinful_fermion_full]


	for i,(basis_2d,basis_2d_full,basis_2d_made,basis_2d_full_made) in enumerate(zip(bases_2d,bases_2d_full,bases_2d,bases_2d_full)):

		basis_2d_made.make(Ns_block_est=16000)
		basis_2d_full_made.make(Ns_block_est=16000)

		if i in [2,6]: # fermionic
			static=[['zz',J_p],['+-',J_p],['-+',J_n]]
		else:
			static=[['zz',J_p],['+-',J_p],['-+',J_p]]

		print('# of states', i, basis_2d_made.Ns, basis_2d_full_made.Ns)

		H=hamiltonian(static,[],basis=basis_2d_made,dtype=np.float64)
		H_full=hamiltonian(static,[],basis=basis_2d_full_made,dtype=np.float64)

		E_GS,V_GS=H.eigsh(k=1)
		E_GS_full,V_GS_full=H_full.eigsh(k=1)

		np.testing.assert_allclose(E_GS - E_GS_full,0.0,atol=1E-5,err_msg='failed energies comparison!')



		amps=np.zeros(basis_2d.Ns,dtype=H.dtype)
		states=basis_2d_made.states.copy()
		inds=[np.where(basis_2d_full.states==r)[0][0] for r in states]
		psi_GS=V_GS[:,0]
		print(psi_GS[0:4])
		out=basis_2d.get_amp(states,amps=psi_GS,mode='representative')
		print(psi_GS[0:4])
		print(V_GS_full[inds,0][:4])
		np.testing.assert_allclose(psi_GS - V_GS_full[inds,0],0.0,atol=1E-5,err_msg='failed representative mode comparison!')

		exit()

		amps_full=np.zeros(basis_2d_full.Ns,dtype=H_full.dtype)
		states_full=basis_2d_full_made.states.copy()
		psi_GS_full=V_GS_full[:,0]
		basis_2d.get_amp(states_full,amps=psi_GS_full,mode='full_basis')
		ref_states=np.sort( np.unique(states_full) )[::-1]
		psi_GS_full=psi_GS_full[inds]
		np.testing.assert_allclose(psi_GS_full - V_GS[:,0],0.0,atol=1E-5,err_msg='failed full_basis mode comparison!')



		print('test {} passed'.format(i))

