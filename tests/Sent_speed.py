from quspin.basis import spin_basis_1d
from quspin.tools.measurements import ent_entropy
import numpy as np
import timeit


setup_pure="""
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import ent_entropy
import numpy as np
import timeit

np.random.seed(0)


L = 10

basis = spin_basis_1d(L,S="1/2")

Ns = basis.Ns
print Ns

psi = np.random.uniform(-1,1,size=Ns) + 1j*np.random.uniform(-1,1,size=Ns)
psi /= np.linalg.norm(psi)

#rho = np.outer(psi.conj(),psi)
psis = np.vstack((psi for i in range(100)))


subsys = [1]

"""



number = 1000
print "measurement: ",timeit.timeit("ent_entropy(psi,basis,chain_subsys = subsys)",setup=setup_pure, number=number)/number
print "basis:       ",timeit.timeit("basis.ent_entropy(psi,sparse=True)",setup=setup_pure, number=number)/number

print "measurement: ",timeit.timeit("ent_entropy({'V_states':psis.T},basis,chain_subsys = subsys)",setup=setup_pure, number=number)/number
print "basis:       ",timeit.timeit("basis.ent_entropy(psis)",setup=setup_pure, number=number)/number

#print "measurement: ",timeit.timeit("ent_entropy(rho,basis,chain_subsys = subsys)",setup=setup_pure, number=number)/number
#print "basis:       ",timeit.timeit("basis.ent_entropy(rho)",setup=setup_pure, number=number)/number

