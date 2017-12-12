from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.basis import spin_basis_1d
from quspin.basis import spin_basis_general
import numpy as np
from itertools import product

try:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in xrange(1,10001)}
except NameError:
	S_dict = {(str(i)+"/2" if i%2==1 else str(i//2)):(i+1,i/2.0) for i in range(1,10001)}

def check_ME(b1,b2,opstr,indx,dtype,err_msg):

	if b1.Ns != b2.Ns:
		print(b1._basis)
		print(b2._basis)
		raise Exception("number of states do not match.")

	ME1,row1,col1=b1.Op(opstr,indx,1.0,dtype)
	ME2,row2,col2=b2.Op(opstr,indx,1.0,dtype)

	if len(ME1) != len(ME2):
		print(ME1)
		print(row1)
		print(col1)
		print()
		print(ME2)
		print(row2)
		print(col2)
		raise Exception("number of matrix elements do not match.")

	if len(ME1)>0 and len(ME2)>0:
		try:
			np.testing.assert_allclose(row1-row2,0,atol=1e-6,err_msg=err_msg)
			np.testing.assert_allclose(col1-col2,0,atol=1e-6,err_msg=err_msg)
			np.testing.assert_allclose(ME1-ME2,0,atol=1e-6,err_msg=err_msg)
		except:
			print(ME1)
			print(row1)
			print(col1)
			print()
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

		basis_1d = spin_basis_1d(L,Nup=Nup,**basis_blocks)
		gen_basis = spin_basis_general(L,Nup=Nup,**gen_blocks)
		n = basis_1d._get_norms(np.float64)**2

		if basis_1d.Ns != gen_basis.Ns:
			print(basis_1d)
			print(gen_basis)
			raise ValueError("basis size mismatch")
		np.testing.assert_allclose(basis_1d._basis-gen_basis._basis,0,atol=1e-6)
		np.testing.assert_allclose(n- gen_basis._n,0,atol=1e-6)

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

print("testing S=1/2")
test_gen_basis_spin(3,S="1/2")
print("testing S=1")
test_gen_basis_spin(3,S="1")
print("testing S=3/2")
test_gen_basis_spin(3,S="3/2")
print("testing S=2")
test_gen_basis_spin(3,S="2")
