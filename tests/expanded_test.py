import sys,os
quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_1d


L=10

J = [[1.0,i,(i+1)%L,(i+2)%L] for i in range(L)]

static = [["xyx",J],["yxy",J],["z+z",J]]
dynamic = []


b = spin_basis_1d(L)


static,dynamic = b.expanded_form(static,dynamic)

for opstr,indx in static:
	print opstr,indx



