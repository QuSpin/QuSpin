from __future__ import print_function
import sys,os

qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)
import numpy as np
from quspin.operators import hamiltonian



def test_trace():
	M = np.arange(9).reshape((3,3)).astype(np.complex128)
	H = hamiltonian([M],[])
	H_trace = H.trace()
	trace = np.trace(M)
	np.testing.assert_equal(H_trace,trace)





test_trace()