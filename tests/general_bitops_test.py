from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

import numpy as np
from quspin.basis import spin_basis_general
from quspin.basis import bitwise_not,bitwise_and,bitwise_or,bitwise_xor,bitwise_left_shift,bitwise_right_shift

def test(y1,y2):
	np.testing.assert_allclose(y1,y2)

def initiate(x1):
	where=np.ones(x1.shape,dtype=bool)
	out=np.zeros_like(x1)
	return out,where


for N in [4,6,8]:

	basis=spin_basis_general(N)
	x1=basis.states[:2**(N-1)]
	x2=basis.states[2**(N-1):]
	b=1*np.ones(x1.shape,dtype=np.uint32) # shift by b bits


	# test NOT
	out,where=initiate(x1)
	y1_np=np.invert(x1)
	y1=bitwise_not(x1)
	y1_where=bitwise_not(x1,where=where)
	bitwise_not(x1,where=where,out=out)
	test(y1,y1_np)
	test(y1,y1_where)
	test(y1,out)


	# test AND, OR, XOR
	funcs=[(np.bitwise_and,bitwise_and),(np.bitwise_or,bitwise_or),(np.bitwise_xor,bitwise_xor)]
	for numpy_func, quspin_func in funcs:
		out,where=initiate(x1)
		y1_np=numpy_func(x1,x2)
		y1=quspin_func(x1,x2)
		y1_where=quspin_func(x1,x2,where=where)
		quspin_func(x1,x2,where=where,out=out)
		test(y1,y1_np)
		test(y1,y1_where)
		test(y1,out)


	# test shifts
	funcs=[(np.left_shift,bitwise_left_shift),(np.right_shift,bitwise_right_shift)]
	for numpy_func, quspin_func in funcs:
		out,where=initiate(x1)
		y1_np=numpy_func(x1,b)
		y1=quspin_func(x1,b)
		y1_where=quspin_func(x1,b,where=where)
		quspin_func(x1,b,where=where,out=out)
		test(y1,y1_np)
		test(y1,y1_where)
		test(y1,out)


	print("passed bitwise_ops test for N={0:d}".format(N))
