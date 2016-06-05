import basis_ops
import numpy as np
from exact_diag_py.basis import basis1d

typecode = {np.float32:"s",np.float64:"d",np.complex64:"c",np.complex128:"z"}

def ncr(n, r):
	import operator as op
	r = min(r, n-r)
	if r == 0: return 1
	numer = reduce(op.mul, xrange(n, n-r, -1))
	denom = reduce(op.mul, xrange(1, r+1))
	return numer//denom




L=10
Nup = L/2
kblock = 5
pblock = 1
zblock = None
pzblock = None
a=1
if Nup is None:
	Ns = 2**L
else:
	Ns = ncr(L,Nup)

dtype = np.complex128


b = basis1d(L,Nup=Nup,kblock=kblock,pblock=pblock,pzblock=pzblock,zblock=zblock,a=a)
N = np.zeros((Ns,),dtype=np.int8)

indx = np.array([0,1,3,4],dtype=np.int32)
opstr = "+zz-"

basis = np.zeros((Ns,),dtype=np.int32)



if Nup is None:
	if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
		m = np.zeros((Ns,),dtype=np.int16)
		Ns = basis_ops.make_t_p_z_basis(L,pblock,zblock,kblock,a,N,m,basis)
		m = m[:Ns]
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_p_z_op"](N,m,basis,opstr,indx,L,pblock,zblock,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 

	elif (type(kblock) is int) and (type(pblock) is int):
		m = np.zeros((Ns,),dtype=np.int8)
		Ns = basis_ops.make_t_p_basis(L,pblock,kblock,a,N,m,basis)
		m = m[:Ns]
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_p_op"](N,m,basis,opstr,indx,L,pblock,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 

	elif (type(kblock) is int) and (type(zblock) is int):
		m = np.zeros((Ns,),dtype=np.int8)
		Ns = basis_ops.make_t_z_basis(L,zblock,kblock,a,N,m,basis)
		m = m[:Ns]
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_z_op"](N,m,basis,opstr,indx,L,zblock,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 

	elif (type(kblock) is int) and (type(pzblock) is int):
		m = np.zeros((Ns,),dtype=np.int8)
		Ns = basis_ops.make_t_pz_basis(L,pzblock,kblock,a,N,m,basis)
		m = m[:Ns]
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_pz_op"](N,m,basis,opstr,indx,L,pzblock,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 

	elif (type(pblock) is int) and (type(zblock) is int):
		Ns = basis_ops.make_p_z_basis(L,pblock,zblock,N,basis)
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_p_z_op"](N,basis,opstr,indx,L,pblock,zblock)
		print b.basis-basis
		print 
		print b.N-N
		print 

	elif (type(kblock) is int):
		Ns = basis_ops.make_t_basis(L,kblock,a,N,basis)
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_op"](N,basis,opstr,indx,L,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
	
	elif (type(pzblock) is int):
		Ns = basis_ops.make_pz_basis(L,pzblock,N,basis)
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_pz_op"](N,basis,opstr,indx,L,pzblock)
		print b.basis-basis
		print 
		print b.N-N
		print 

	elif (type(pblock) is int):
		Ns = basis_ops.make_p_basis(L,pblock,N,basis)
		basis = basis[:Ns]
		N = N[:Ns]
		func = basis_ops.__dict__[typecode[dtype]+"_t_op"]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_p_op"](N,basis,opstr,indx,L,pblock)
		print b.basis-basis
		print 
		print b.N-N
		print 

	elif (type(zblock) is int):
		Ns = basis_ops.make_z_basis(L,zblock,basis)
		basis = basis[:Ns]
		func = basis_ops.__dict__[typecode[dtype]+"_t_op"]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_z_op"](basis,opstr,indx,L,zblock)
		print b.basis-basis
		print 
	else:
		basis = np.arange(2**L,dtype=basis.dtype)
		col_test,ME_test,error = col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_spinop"](basis,opstr,indx)
		

else:
	if (type(kblock) is int) and (type(pblock) is int) and (type(zblock) is int):
		m = np.zeros((Ns,),dtype=np.int16)
		Ns = basis_ops.make_m_t_p_z_basis(L,Nup,pblock,zblock,kblock,a,N,m,basis)
		m = m[:Ns]
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_p_z_op"](N,m,basis,opstr,indx,L,pblock,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 

	elif (type(kblock) is int) and (type(pblock) is int):
		m = np.zeros((Ns,),dtype=np.int8)
		Ns = basis_ops.make_m_t_p_basis(L,Nup,pblock,kblock,a,N,m,basis)
		m = m[:Ns]
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_p_op"](N,m,basis,opstr,indx,L,pblock,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 

	elif (type(kblock) is int) and (type(zblock) is int):
		m = np.zeros((Ns,),dtype=np.int8)
		Ns = basis_ops.make_m_t_z_basis(L,Nup,zblock,kblock,a,N,m,basis)
		m = m[:Ns]
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_z_op"](N,m,basis,opstr,indx,L,zblock,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 


	elif (type(kblock) is int) and (type(pzblock) is int):
		m = np.zeros((Ns,),dtype=np.int8)
		Ns = basis_ops.make_m_t_pz_basis(L,Nup,pzblock,kblock,a,N,m,basis)
		m = m[:Ns]
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_pz_op"](N,m,basis,opstr,indx,L,pzblock,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 


	elif (type(pblock) is int) and (type(zblock) is int):
		Ns = basis_ops.make_m_p_z_basis(L,Nup,pblock,zblock,N,basis)
		basis = basis[:Ns]
		N = N[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_p_z_op"](N,basis,opstr,indx,L,pblock,zblock)
		print b.basis-basis
		print 
		print b.N-N
		print 
		print b.m-m
		print 


	elif (type(kblock) is int):
		Ns = basis_ops.make_m_t_basis(L,Nup,kblock,a,N,basis)
		basis = basis[:Ns]
		N = N[:Ns]

		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_t_op"](N,basis,opstr,indx,L,kblock,a)
		print b.basis-basis
		print 
		print b.N-N
		print 

	
	elif (type(pzblock) is int):
		Ns = basis_ops.make_m_pz_basis(L,Nup,pzblock,N,basis)
		basis = basis[:Ns]
		N = N[:Ns]

		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_pz_op"](N,basis,opstr,indx,pzblock)
		print b.basis-basis
		print 
		print b.N-N
		print 


	elif (type(pblock) is int):
		Ns = basis_ops.make_m_p_basis(L,Nup,pblock,N,basis)
		basis = basis[:Ns]
		N = N[:Ns]

		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_p_op"](N,basis,opstr,indx,pblock)
		print b.basis-basis
		print 
		print b.N-N
		print 



	elif (type(zblock) is int):
		Ns = basis_ops.make_m_z_basis(L,Nup,zblock,basis)
		basis = basis[:Ns]

		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_z_op"](basis,opstr,indx,zblock)
		print b.basis-basis
		print 


	else:
		basis = basis_ops.make_m_basis(L,Nup,Ns)
		basis = basis[:Ns]
		col_test,ME_test,error = basis_ops.__dict__[typecode[dtype]+"_m_op"](basis,opstr,indx)
		print b.basis-basis
		print 


print Ns
ME,row,col = b.Op(opstr,indx,1.0,dtype,True)
mask = col_test > -1

col_test = col_test[mask]
ME_test = ME_test[mask]

mask = (ME != 0.0)
mask_test = ME_test != 0.0

#print ME[mask].real,"\n",ME_test[mask_test].real
#print np.linalg.norm(ME-ME_test)
#print ME-ME_test
print 
#print col[mask],"\n",col_test[mask_test]
#print np.linalg.norm(col-col_test)
#print col-col_test
#print 
#print b.basis[col],"\n",basis[col_test]
#print basis[col != col_test]


#for me,r,c,me1,r1,c1 in zip(ME,basis,basis[col],ME_test,basis,basis[col_test]):
#	print ("{0:0"+str(L)+"b}").format(r),("{0:0"+str(L)+"b}").format(c),("{0:0"+str(L)+"b}").format(c1)

"""



print basis.dtype


col,ME,error = basis_ops.z_spinop(basis,opstr,indx)

print ME.dtype
print error

for si,sf,me in zip(basis,col,ME):
	print ("{0:0"+str(L)+"b} {1:0"+str(L)+"b}").format(np.uint64(si),np.uint64(sf)),me

"""
