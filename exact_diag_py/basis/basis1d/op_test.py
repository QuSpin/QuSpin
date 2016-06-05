import numpy as np
from numpy.linalg import norm
from basis1d import basis1d
from fortran_constructors import op as f_op
from constructors import op as op
from time import time



L=12
Nup=L/2
kblock=None
pblock=None
zblock=None
pzblock=None
a=2


J=-1.0


opstr = "yxxy"
indx = np.array([4,5,6,7],dtype=np.int32)
dtype=np.float64

b = basis1d(L,Nup=Nup,kblock=kblock,pblock=pblock,zblock=zblock,pzblock=pzblock,a=a)


times= []
for i in xrange(100):
	t = time()
	ME1,row1,col1 = f_op(opstr,indx,J,dtype,False,b.basis)#,**b.blocks)
	times.append(time()-t)
print np.mean(times)


times= []
for i in xrange(100):
	t = time()
	ME1,row1,col1 = op(opstr,indx,J,dtype,False,b.basis)#,**b.blocks)
	times.append(time()-t)
print np.mean(times)













