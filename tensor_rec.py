from quspin.basis import tensor_basis,boson_basis_1d



L=3
Li=2
b1 = boson_basis_1d(Li,sps=2)

basis1 = boson_basis_1d(Li*L,sps=2)
basis2 = tensor_basis(*(b1 for i in range(L)))

print basis1
print basis2




