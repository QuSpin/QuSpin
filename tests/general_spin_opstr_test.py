from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_1d
from quspin.basis import spin_basis_general
from quspin.basis import basis_int_to_python_int
import numpy as np
from itertools import product

try:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in xrange(1,10001)}
except NameError:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in range(1,10001)}

def check_ME(basis_1d,basis_gen,opstr,indx,dtype,err_msg):

	ME1,row1,col1=basis_1d.Op(opstr,indx,1.0,dtype)
	ME2,row2,col2=basis_gen.Op(opstr,indx,1.0,dtype)

	if len(ME1) != len(ME2):
		print(opstr,list(indx))
		print(basis_1d)
		print("spin_basis_1d:")
		print(ME1)
		print(row1)
		print(col1)
		print()
		print("spin_basis_general")
		print(ME2)
		print(row2)
		print(col2)
		raise Exception("number of matrix elements do not match.")

	if len(ME1)>0 and len(ME2)>0:
		row1 = row1.astype(np.min_scalar_type(row1.max()))
		row2 = row2.astype(np.min_scalar_type(row2.max()))
		col1 = row2.astype(np.min_scalar_type(col1.max()))
		col2 = row2.astype(np.min_scalar_type(col2.max()))
		try:
			np.testing.assert_allclose(row1,row2,atol=1e-6,err_msg=err_msg)
			np.testing.assert_allclose(col1,col2,atol=1e-6,err_msg=err_msg)
			np.testing.assert_allclose(ME1,ME2,atol=1e-6,err_msg=err_msg)
		except AssertionError:
			print(err_msg)
			print(basis_1d)
			print("difference:")
			print(ME1-ME2)
			print(row1-row2)
			print(col1-col2)
			print("spin_basis_1d:")
			print(ME1)
			print(row1)
			print(col1)
			print("spin_basis_general")
			print(ME2)
			print(row2)
			print(col2)
			raise Exception

def test_gen_basis_spin(l_max,S="1/2"):
	L=6
	kblocks = [None]
	kblocks.extend(range(L))
	pblocks = [None,0,1]
	zblocks = [None,0,1]

	if S=="1/2":
		ops = ["x","y","z","+","-","I"]
	else:
		ops = ["z","+","-","I"]

	sps,s=S_dict[S]

	Nups = [None,int(s*L)]
	
	t = np.array([(i+1)%L for i in range(L)])
	p = np.array([L-i-1 for i in range(L)])
	z = np.array([-(i+1) for i in range(L)])

	for Nup,kblock,pblock,zblock in product(Nups,kblocks,pblocks,zblocks):
		gen_blocks = {"pauli":False,"S":S}
		basis_blocks = {"pauli":False,"S":S}

		if kblock==0 or kblock==L//2:
			if pblock is not None:
				basis_blocks["pblock"] = (-1)**pblock
				gen_blocks["pblock"] = (p,pblock)
			else:
				basis_blocks["pblock"] = None
				gen_blocks["pblock"] = None
		else:
			if pblock is not None:
				continue
			basis_blocks["pblock"] = None
			gen_blocks["pblock"] = None

		if zblock is not None:
			basis_blocks["zblock"] = (-1)**zblock
			gen_blocks["zblock"] = (z,zblock)
		else:
			basis_blocks["zblock"] = None
			gen_blocks["zblock"] = None

		if kblock is not None:
			basis_blocks["kblock"] = kblock
			gen_blocks["kblock"] = (t,kblock)
		else:
			basis_blocks["kblock"] = None
			gen_blocks["kblock"] = None

		print("checking S={S:} Nup={Nup:} kblock={kblock:} pblock={pblock:} zblock={zblock:}".format(Nup=Nup,**basis_blocks))

		basis_1d = spin_basis_1d(L,Nup=Nup,**basis_blocks)
		gen_basis = spin_basis_general(L,Nup=Nup,**gen_blocks)
		n = basis_1d._get_norms(np.float64)**2
		n_gen = (gen_basis._n.astype(np.float64))*gen_basis._pers.prod()


		if basis_1d.Ns != gen_basis.Ns:
			print(L,basis_blocks)
			print(basis_1d)
			print(gen_basis)
			raise ValueError("basis size mismatch")

		try:
			np.testing.assert_allclose(basis_1d.states-gen_basis.states,0,atol=1e-6)
			np.testing.assert_allclose(n , n_gen,atol=1e-6)
		except:
			print(basis_1d.states)
			print(gen_basis.states)
			print(n)
			print(n_gen)
			raise Exception

		for l in range(1,l_max+1):
			for i0 in range(0,L-l+1,1):
				indx = range(i0,i0+l,1)
				for opstr in product(*[ops for i in range(l)]):
					opstr = "".join(list(opstr))
					printing = dict(basis_blocks)
					printing["opstr"]=opstr
					printing["indx"]=indx
					printing["Nup"]=Nup
					printing["S"]=S

					err_msg="testing: {opstr:} {indx:}  S={S:} Nup={Nup:} kblock={kblock:} pblock={pblock:} zblock={zblock:}".format(**printing)

					check_ME(basis_1d,gen_basis,opstr,indx,np.complex128,err_msg)


def test_gen_basis_spin_boost(L,Nups,l_max,S="1/2"):
	kblocks = [None]
	kblocks.extend(range(0,L,(L//4)))
	pblocks = [None,0,1]
	if S=="1/2":
		ops = ["x","y","z","+","-","I"]
	else:
		ops = ["z","+","-","I"]

	sps,s=S_dict[S]
	
	t = np.array([(i+1)%L for i in range(L)])
	p = np.array([L-i-1 for i in range(L)])

	for Nup,kblock,pblock in product(Nups,kblocks,pblocks):
		gen_blocks = {"pauli":False,"S":S}
		basis_blocks = {"pauli":False,"S":S}

		if kblock==0 or kblock==L//2:
			if pblock is not None:
				basis_blocks["pblock"] = (-1)**pblock
				gen_blocks["pblock"] = (p,pblock)
			else:
				basis_blocks["pblock"] = None
				gen_blocks["pblock"] = None
		else:
			if pblock is not None:
				continue
			basis_blocks["pblock"] = None
			gen_blocks["pblock"] = None

		if kblock is not None:
			basis_blocks["kblock"] = kblock
			gen_blocks["kblock"] = (t,kblock)
		else:
			basis_blocks["kblock"] = None
			gen_blocks["kblock"] = None

		basis_1d = spin_basis_1d(L,Nup=Nup,**basis_blocks)
		gen_basis = spin_basis_general(L,Nup=Nup,**gen_blocks)
		n = basis_1d._get_norms(np.float64)**2
		n_gen = (gen_basis._n.astype(np.float64))*gen_basis._pers.prod()

		print("checking S={S:} Nup={Nup:} kblock={kblock:} pblock={pblock:}".format(Nup=Nup,**basis_blocks))
		if basis_1d.Ns != gen_basis.Ns:
			print(L,basis_blocks)
			print(basis_1d)
			print(gen_basis)
			raise ValueError("basis size mismatch")

		try:
			for s_general,s_1d in zip(gen_basis.states,basis_1d.states):
				assert(basis_int_to_python_int(s_general)==s_1d)
			np.testing.assert_allclose(n-n_gen ,0,atol=1e-6)
		except:
			print(basis_1d)
			print(n)

			print(gen_basis)
			print(n_gen)
			raise Exception

		for l in range(1,l_max+1):
			for i0 in range(0,L-l+1,1):
				indx = list(range(i0,i0+l,1))
				for opstr in product(*[ops for i in range(l)]):
					opstr = "".join(list(opstr))
					printing = dict(basis_blocks)
					printing["opstr"]=opstr
					printing["indx"]=indx
					printing["Nup"]=Nup
					printing["S"]=S

					err_msg="testing: {opstr:} {indx:}  S={S:} Nup={Nup:} kblock={kblock:} pblock={pblock:}".format(**printing)

					check_ME(basis_1d,gen_basis,opstr,indx,np.complex128,err_msg)


print("testing S=1/2")
test_gen_basis_spin(3,S="1/2")
test_gen_basis_spin_boost(66,[1,-2],2,S="1/2")
print("testing S=1")
test_gen_basis_spin(3,S="1")
test_gen_basis_spin_boost(40,[1,-2],2,S="1")
print("testing S=3/2")
test_gen_basis_spin(3,S="3/2")
test_gen_basis_spin_boost(34,[1,-2],2,S="3/2")
print("testing S=2")
test_gen_basis_spin(3,S="2")
test_gen_basis_spin_boost(30,[1,-2],2,S="2")
