from quspin.basis import spin_basis_general,spin_basis_1d
import numpy as np


L = 4

t = (np.arange(L)+1)%L
z = -(np.arange(L)+1)

b = spin_basis_1d(L,kblock=1)
b1 = spin_basis_1d(L,kblock=1,zblock=1)
b2 = spin_basis_1d(L,kblock=1,zblock=-1)
b3 = spin_basis_general(L,kblock=(t,1),zblock=(z,0))
b4 = spin_basis_general(L,kblock=(t,1),zblock=(z,1))
b5 = spin_basis_general(L,kblock=(t,1))

print b.Ns
print b1.Ns+b2.Ns
print b3.Ns+b4.Ns
print b5.Ns
