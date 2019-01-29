from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

from quspin.tools.misc import csr_matvec
import scipy.sparse as sp
import numpy as np

N=1000

for i in range(10):
	print("random matrix test: {}".format(i))

	A = sp.random(N,N,format="csr")
	a = np.random.uniform(-1,1)
	V = np.random.uniform(-1,1,size=N)
	V3 = np.zeros_like(V)

	V1 = a*A.dot(V)
	V2 = csr_matvec(A,V,a=a)
	csr_matvec(A,V,a=a,out=V3)
	np.testing.assert_allclose(V1,V2,atol=1e-15,rtol=1e-7)
	np.testing.assert_allclose(V1,V3,atol=1e-15,rtol=1e-7)


	V = V + 1j*np.random.uniform(-1,1,size=N)
	V3 = np.zeros_like(V)
	V1 = a*A.dot(V)
	V2 = csr_matvec(A,V,a=a)
	csr_matvec(A,V,out=V3,a=a)
	np.testing.assert_allclose(V1,V2,atol=1e-15,rtol=1e-7)
	np.testing.assert_allclose(V1,V3,atol=1e-15,rtol=1e-7)


	A = A + 1j*sp.random(N,N,format="csr")
	a = a + 1j*np.random.uniform(-1,1)
	V1 = a*A.dot(V)
	V2 = csr_matvec(A,V,a=a)
	csr_matvec(A,V,out=V3,a=a)
	np.testing.assert_allclose(V1,V2,atol=1e-15,rtol=1e-7)
	np.testing.assert_allclose(V1,V3,atol=1e-15,rtol=1e-7)
