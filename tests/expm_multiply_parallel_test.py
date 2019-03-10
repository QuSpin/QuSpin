from __future__ import print_function, division

import sys,os
qspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,qspin_path)

#print(os.environ["OMP_NUM_THREADS"])

from quspin.tools.evolution import expm_multiply_parallel
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import random,identity
import numpy as np

seed=np.random.randint(10000000) #9513926
np.random.seed(seed)



N = 3500
for i in range(10):
	print("testing random matrix {}".format(i+1))
	A = (random(N,N) + 1j*random(N,N))
	A = A.tocsr()
	A = (A + A.H)/2.0
	v = np.random.uniform(-1,1,size=N) + 1j * np.random.uniform(-1,1,size=N)

	v1 = expm_multiply(-1j*A,v)
	v2 = expm_multiply_parallel(A,a=-1j).dot(v)
	
	np.testing.assert_allclose(v1-v2,0,atol=1e-10,err_msg='failed seed {:d}'.format(seed) )


print("expm_multiply_parallel tests passed!")





