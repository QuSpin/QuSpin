from __future__ import print_function, division

import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

# return line number
import inspect
def lineno():
    """Returns the current line number in our program."""
    return inspect.currentframe().f_back.f_lineno

from quspin.basis import spin_basis_1d,photon_basis # Hilbert space bases
from quspin.operators import hamiltonian # Hamiltonian and observables
from quspin.tools.Floquet import  Floquet, Floquet_t_vec
import numpy as np
from numpy.random import uniform,seed,shuffle,randint # pseudo random numbers
seed()

"""
This script tests the Floquet class.
"""

# matrix scipy's logm does not support complex256 and float128
dtypes={"float32":np.float32,"float64":np.float64,"complex64":np.complex64,"complex128":np.complex128}
atols={"float32":1E-4,"float64":1E-13,"complex64":1E-4,"complex128":1E-13}

def drive(t,Omega):
	return np.sign(np.cos(Omega*t))

def test():
	for _r in range(10): # 10 random realisations

		##### define model parameters #####
		L=4 # system size
		J=1.0 # spin interaction
		g=uniform(0.2,1.5) # transverse field
		h=uniform(0.2,1.5) # parallel field
		Omega=uniform(8.0,10.0) # drive frequency
		#
		##### set up alternating Hamiltonians #####
		# define time-reversal symmetric periodic step drive

		drive_args=[Omega]
		# compute basis in the 0-total momentum and +1-parity sector
		basis=spin_basis_1d(L=L,a=1,kblock=0,pblock=1)
		# define PBC site-coupling lists for operators
		x_field_pos=[[+g,i]	for i in range(L)]
		x_field_neg=[[-g,i]	for i in range(L)]
		z_field=[[h,i]		for i in range(L)]
		J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC
		# static and dynamic lists for time-dep H
		static=[["zz",J_nn],["z",z_field],["x",x_field_pos]]
		dynamic=[["zz",J_nn,drive,drive_args],
				 ["z",z_field,drive,drive_args],["x",x_field_neg,drive,drive_args]]
		# static and dynamic lists for step drive
		static1=[["zz",J_nn],["z",z_field]]
		static2=[["x",x_field_pos]]

		# loop over dtypes
		for _i in dtypes.keys():
			
			dtype = dtypes[_i]
			atol = atols[_i]

			# compute Hamiltonians
			H=0.5*hamiltonian(static,dynamic,dtype=dtype,basis=basis)
			H1=hamiltonian(static1,[],dtype=dtype,basis=basis)
			H2=hamiltonian(static2,[],dtype=dtype,basis=basis)
			#
			##### define time vector of stroboscopic times with 100 cycles #####
			t=Floquet_t_vec(Omega,20,len_T=1) # t.vals=times, t.i=init. time, t.T=drive period
			#
			##### calculate exact Floquet eigensystem #####
			t_list=np.array([0.0,t.T/4.0,3.0*t.T/4.0])+np.finfo(float).eps # times to evaluate H
			dt_list=np.array([t.T/4.0,t.T/2.0,t.T/4.0]) # time step durations to apply H for


			###
			# call Floquet class for evodict a coutinous H from a Hamiltonian object
			Floq_Hevolve=Floquet({'H':H,'T':t.T,'atol':1E-16,'rtol':1E-16},n_jobs=2) 
			EF_Hevolve=Floq_Hevolve.EF # read off quasienergies
			# call Floquet class for evodict a step H from a Hamiltonian object
			Floq_H=Floquet({'H':H,'t_list':t_list,'dt_list':dt_list},n_jobs=2) 
			EF_H=Floq_H.EF # read off quasienergies
			# call Floquet class for evodict a step H from a list of Hamiltonians
			Floq_Hlist=Floquet({'H_list':[H1,H2,H1],'dt_list':dt_list},n_jobs=2) # call Floquet class
			EF_Hlist=Floq_Hlist.EF

			try:
				np.testing.assert_allclose(EF_H,EF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(EF_H,EF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
			except AssertionError:
				print('dtype, (g,h,Omega) =', dtype, (g,h,Omega))
				print('exiting in line', lineno()+1)
				exit()
			###
			# call Floquet class for evodict a coutinous H from a Hamiltonian object
			Floq_Hevolve=Floquet({'H':H,'T':t.T,'atol':1E-16,'rtol':1E-16},n_jobs=randint(2)+1,VF=True) 
			EF_Hevolve=Floq_Hevolve.EF
			VF_Hevolve=Floq_Hevolve.VF
			# call Floquet class for evodict a step H from a Hamiltonian object
			Floq_H=Floquet({'H':H,'t_list':t_list,'dt_list':dt_list},n_jobs=randint(2)+1,VF=True) 
			EF_H=Floq_H.EF # read off quasienergies
			VF_H=Floq_H.VF # read off Floquet states
			# call Floquet class for evodict a step H from a list of Hamiltonians
			Floq_Hlist=Floquet({'H_list':[H1,H2,H1],'dt_list':dt_list},n_jobs=randint(2)+1,VF=True) # call Floquet class
			EF_Hlist=Floq_Hlist.EF
			VF_Hlist=Floq_Hlist.VF

			try:
				np.testing.assert_allclose(EF_H,EF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(VF_H,VF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')

				np.testing.assert_allclose(EF_H,EF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(VF_H,VF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
			except AssertionError:
				print('dtype, (g,h,Omega) =', dtype, (g,h,Omega))
				print('exiting in line', lineno()+1)
				exit()

			###
			# call Floquet class for evodict a coutinous H from a Hamiltonian object
			Floq_Hevolve=Floquet({'H':H,'T':t.T,'atol':1E-16,'rtol':1E-16},n_jobs=randint(2)+1,VF=True,UF=True) 
			EF_Hevolve=Floq_Hevolve.EF
			VF_Hevolve=Floq_Hevolve.VF
			UF_Hevolve=Floq_Hevolve.UF
			# call Floquet class for evodict a step H from a Hamiltonian object
			Floq_H=Floquet({'H':H,'t_list':t_list,'dt_list':dt_list},n_jobs=randint(2)+1,VF=True,UF=True) 
			EF_H=Floq_H.EF # read off quasienergies
			VF_H=Floq_H.VF # read off Floquet states
			UF_H=Floq_H.UF
			# call Floquet class for evodict a step H from a list of Hamiltonians
			Floq_Hlist=Floquet({'H_list':[H1,H2,H1],'dt_list':dt_list},n_jobs=randint(2)+1,VF=True,UF=True) # call Floquet class
			EF_Hlist=Floq_Hlist.EF
			VF_Hlist=Floq_Hlist.VF
			UF_Hlist=Floq_Hlist.UF

			try:
				np.testing.assert_allclose(EF_H,EF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(VF_H,VF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(UF_H,UF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')

				np.testing.assert_allclose(EF_H,EF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(VF_H,VF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(UF_H,UF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
			except AssertionError:
				print('dtype, (g,h,Omega) =', dtype, (g,h,Omega))
				print('exiting in line', lineno()+1)
				exit()

			###
			# call Floquet class for evodict a coutinous H from a Hamiltonian object
			Floq_Hevolve=Floquet({'H':H,'T':t.T,'atol':1E-16,'rtol':1E-16},n_jobs=randint(2)+1,VF=True,UF=True,HF=True) 
			EF_Hevolve=Floq_Hevolve.EF
			VF_Hevolve=Floq_Hevolve.VF
			UF_Hevolve=Floq_Hevolve.UF
			HF_Hevolve=Floq_Hevolve.HF
			# call Floquet class for evodict a step H from a Hamiltonian object
			Floq_H=Floquet({'H':H,'t_list':t_list,'dt_list':dt_list},n_jobs=randint(2)+1,VF=True,UF=True,HF=True) 
			EF_H=Floq_H.EF # read off quasienergies
			VF_H=Floq_H.VF # read off Floquet states
			UF_H=Floq_H.UF
			HF_H=Floq_H.HF
			# call Floquet class for evodict a step H from a list of Hamiltonians
			Floq_Hlist=Floquet({'H_list':[H1,H2,H1],'dt_list':dt_list},n_jobs=randint(2)+1,VF=True,UF=True,HF=True) # call Floquet class
			EF_Hlist=Floq_Hlist.EF
			VF_Hlist=Floq_Hlist.VF
			UF_Hlist=Floq_Hlist.UF
			HF_Hlist=Floq_Hlist.HF

			try:
				np.testing.assert_allclose(EF_H,EF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(VF_H,VF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(UF_H,UF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(HF_H,HF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')

				np.testing.assert_allclose(EF_H,EF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(VF_H,VF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(UF_H,UF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(HF_H,HF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
			except AssertionError:
				print('dtype, (g,h,Omega) =', dtype, (g,h,Omega))
				print('exiting in line', lineno()+1)
				exit()

			###
			# call Floquet class for evodict a coutinous H from a Hamiltonian object
			Floq_Hevolve=Floquet({'H':H,'T':t.T,'atol':1E-16,'rtol':1E-16},n_jobs=randint(2)+1,VF=True,UF=True,HF=True,thetaF=True) 
			EF_Hevolve=Floq_Hevolve.EF
			VF_Hevolve=Floq_Hevolve.VF
			UF_Hevolve=Floq_Hevolve.UF
			HF_Hevolve=Floq_Hevolve.HF
			thetaF_Hevolve=Floq_Hevolve.thetaF
			# call Floquet class for evodict a step H from a Hamiltonian object
			Floq_H=Floquet({'H':H,'t_list':t_list,'dt_list':dt_list},n_jobs=randint(2)+1,VF=True,UF=True,HF=True,thetaF=True) 
			EF_H=Floq_H.EF # read off quasienergies
			VF_H=Floq_H.VF # read off Floquet states
			UF_H=Floq_H.UF
			HF_H=Floq_H.HF
			thetaF_H=Floq_H.thetaF
			# call Floquet class for evodict a step H from a list of Hamiltonians
			Floq_Hlist=Floquet({'H_list':[H1,H2,H1],'dt_list':dt_list},n_jobs=randint(2)+1,VF=True,UF=True,HF=True,thetaF=True) # call Floquet class
			EF_Hlist=Floq_Hlist.EF
			VF_Hlist=Floq_Hlist.VF
			UF_Hlist=Floq_Hlist.UF
			HF_Hlist=Floq_Hlist.HF
			thetaF_Hlist=Floq_Hlist.thetaF

			try:
				np.testing.assert_allclose(EF_H,EF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(VF_H,VF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(UF_H,UF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(HF_H,HF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(thetaF_H,thetaF_Hlist,atol=atol,err_msg='Failed Floquet object comparison!')

				np.testing.assert_allclose(EF_H,EF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(VF_H,VF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(UF_H,UF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(HF_H,HF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
				np.testing.assert_allclose(thetaF_H,thetaF_Hevolve,atol=atol,err_msg='Failed Floquet object comparison!')
			except AssertionError:
				print('dtype, (g,h,Omega) =', dtype, (g,h,Omega))
				print('exiting in line', lineno()+1)
				exit()

		print("Floquet class random check {} finished successfully".format(_r))


if __name__ == '__main__':
	test()