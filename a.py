from quspin.basis import boson_basis_general,spin_basis_general

L=12
t = [(i+4)%L for i in range(L)]
p = [L-i-1 for i in range(L)]
b = spin_basis_general(L,S="1",Nup=L,pblock=(p,1))
print b