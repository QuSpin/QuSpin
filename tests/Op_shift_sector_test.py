from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_general
import numpy as np


L = 13


for k in range(L):
	for q in range(L):
		print("testing k={} -> k+q={}".format(k,(k+q)%L))

		# use standard static list for this. 
		# use generators to generate coupling list
		op_list = [["z",[i],np.exp(-2j*np.pi*q*i/L)] for i in range(L)]


		t = (np.arange(L)+1)%L



		b = spin_basis_general(L)
		b1 = spin_basis_general(L,kblock=(t,k))
		b2 = spin_basis_general(L,kblock=(t,k+q))

		# print(b1)
		# print(b2)

		P1 = b1.get_proj(np.complex128)
		P2 = b2.get_proj(np.complex128)

		v_in = np.random.normal(0,1,size=b1.Ns) + 1j*np.random.normal(0,1,size=b1.Ns)
		v_in /= np.linalg.norm(v_in)


		v_in_full = P1.dot(v_in)
		v_out_full = b.inplace_Op(v_in_full,op_list,np.complex128)
		v_out_proj = P2.H.dot(v_out_full)

		# swap b1 with op_list in arguments. 
		v_out = b2.Op_shift_sector(b1,op_list,v_in)

		np.testing.assert_allclose(v_out,v_out_proj,rtol=0, atol=1e-13)



for q1 in [0,1]:
	for q2 in [0,1]:
		i = 1
		print("testing q1={} -> q2={}".format(q1,q2))
		op_list = [["z",[i],1],["z",[L-i-1],(-1)**q2]]

		p = np.arange(L)[::-1]
		# z = -(np.arange(L)+1)


		b = spin_basis_general(L)
		b1 = spin_basis_general(L,block=(p,q1))
		b2 = spin_basis_general(L,block=(p,q1+q2))

		# print(b1)
		# print(b2)

		P1 = b1.get_proj(np.complex128)
		P2 = b2.get_proj(np.complex128)

		v_in = np.random.normal(0,1,size=b1.Ns) + 1j*np.random.normal(0,1,size=b1.Ns)
		v_in /= np.linalg.norm(v_in)


		v_in_full = P1.dot(v_in)
		v_out_full = b.inplace_Op(v_in_full,op_list,np.complex128)
		v_out_proj = P2.H.dot(v_out_full)

		v_out = b2.Op_shift_sector(b1,op_list,v_in)
		np.testing.assert_allclose(v_out,v_out_proj,rtol=0, atol=1e-13)



for Nup in range(0,L):
		print("testig Nup={} -> Nup={}".format(Nup,Nup+1))
		opp_list = [["+",[i],1.0] for i in range(L)]


		b = spin_basis_general(L)
		b1 = spin_basis_general(L,Nup=Nup)
		b2 = spin_basis_general(L,Nup=Nup+1)


		P1 = b1.get_proj(np.complex128)
		P2 = b2.get_proj(np.complex128)

		v_in = np.random.normal(0,1,size=b1.Ns) + 1j*np.random.normal(0,1,size=b1.Ns)
		v_in /= np.linalg.norm(v_in)


		v_in_full = P1.dot(v_in)
		v_out_full = b.inplace_Op(v_in_full,op_list,np.complex128)
		v_out_proj = P2.H.dot(v_out_full)

		v_out = b2.Op_shift_sector(b1,op_list,v_in)
		np.testing.assert_allclose(v_out,v_out_proj,rtol=0, atol=1e-13)

for Nup in range(1,L+1):
		print("testig Nup={} -> Nup={}".format(Nup,Nup-1))
		opm_list = [["-",[i],1.0] for i in range(L)]


		b = spin_basis_general(L)
		b1 = spin_basis_general(L,Nup=Nup)
		b2 = spin_basis_general(L,Nup=Nup-1)


		P1 = b1.get_proj(np.complex128)
		P2 = b2.get_proj(np.complex128)

		v_in = np.random.normal(0,1,size=b1.Ns) + 1j*np.random.normal(0,1,size=b1.Ns)
		v_in /= np.linalg.norm(v_in)


		v_in_full = P1.dot(v_in)
		v_out_full = b.inplace_Op(v_in_full,op_list,np.complex128)
		v_out_proj = P2.H.dot(v_out_full)

		v_out = b2.Op_shift_sector(b1,op_list,v_in)
		np.testing.assert_allclose(v_out,v_out_proj,rtol=0, atol=1e-13)
