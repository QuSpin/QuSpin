from quspin.basis import spinless_fermion_basis_general,spinless_fermion_basis_1d
import numpy as np


L = 6
Nf=2

t = (np.arange(L)-1)%L

Ns = 0
Ns1 = 0

for i in range(L):
	print "kblock={}".format(i)
	b = spinless_fermion_basis_1d(L,Nf=Nf,kblock=i)
	b1 = spinless_fermion_basis_general(L,Nf=Nf,kblock=(t,i))
	print b
	print b1
	print "-------------------------------------------------------------------"
	Ns += b.Ns
	Ns1 += b1.Ns

print Ns,Ns1