from exact_diag_py.spins1D import hamiltonian
from exact_diag_py.Basis import Basis1D
from numpy import *

L=10
Nup=None

def A(t):
	return t

J=[[-1.0,i,(i+1)%L] for i in xrange(L)]
h=[[-2.0,i] for i in xrange(L)]

static=[['zz',J],['xx',J],['yy',J]]


print "making H"




H=hamiltonian(static,[],L,dtype=float32,Nup=Nup)

"""
print "making H"

H=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup)
Ns=H.Ns


H_21=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,zblock=1)
H_22=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,zblock=-1)



H_11=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,pblock=1)
H_12=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,pblock=-1)



H_31=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,pzblock=1)
H_32=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,pzblock=-1)


H_41=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,pblock=1,zblock=1)
H_42=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,pblock=1,zblock=-1)
H_43=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,pblock=-1,zblock=1)
H_44=Hamiltonian1D(static,[],L,dtype=float32,Nup=Nup,pblock=-1,zblock=-1)


E=H.DenseEE()
E1=asarray(sorted(concatenate((H_11.DenseEE(),H_12.DenseEE()))))
E2=asarray(sorted(concatenate((H_21.DenseEE(),H_22.DenseEE()))))
E3=asarray(sorted(concatenate((H_31.DenseEE(),H_32.DenseEE()))))
E4=asarray(sorted(concatenate((H_41.DenseEE(),H_42.DenseEE(),H_43.DenseEE(),H_44.DenseEE()))))

print sum(E-E1)/Ns
print sum(E-E2)/Ns
print sum(E-E3)/Ns
print sum(E-E4)/Ns

"""




