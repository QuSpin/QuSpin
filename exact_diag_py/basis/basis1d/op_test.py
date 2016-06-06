import numpy as np
from numpy.linalg import norm
from basis1d import basis1d
from fortran_constructors import op as f_op
from constructors import op as op
from time import time



L=4
Nup=None
kblock=None
pblock=None
zblock=None
pzblock=None
a=2


J=1.0


opstr = "zz"
indx = np.array([0,1],dtype=np.int32)
dtype=np.complex64

b = basis1d(L,Nup=Nup,kblock=kblock,pblock=pblock,zblock=zblock,pzblock=pzblock,a=a)

if b.Ns > 0:
	basis = np.array(b.basis,dtype=np.int32)
	ME1,row1,col1 = f_op(opstr,indx,J,dtype,False,basis)
	ME2,row2,col2 = op(opstr,indx,J,dtype,False,b.basis)

	print np.linalg.norm(ME1-ME2)
	print ME1-ME2
	print
	print row1-row2
	print col1-col2


"""
times= []
for i in xrange(100):
	t = time()
	ME1,row1,col1 = f_op(opstr,indx,J,dtype,False,b.N,b.basis,L,**b.blocks)
	times.append(time()-t)
print np.mean(times)


times= []
for i in xrange(100):
	t = time()
	ME1,row1,col1 = op(opstr,indx,J,dtype,False,b.N,b.basis,L,**b.blocks)
	times.append(time()-t)
print np.mean(times)

"""











