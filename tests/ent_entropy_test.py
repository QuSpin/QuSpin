import sys,os
qspin_path = os.path.join(os.getcwd(),"../QuSpin_dev/")
sys.path.insert(0,qspin_path)

from quspin.operators import hamiltonian
from quspin.tools.measurements import ent_entropy
import numpy as np




static = [["zz",[[1.0,0,1]]],["yy",[[1.0,0,1]]],["xx",[[1.0,0,1]]]]
H = hamiltonian(static,[],N=2,dtype=np.float64)
basis =  H.basis

E,V = H.eigh()

D = []
for v in V.T[:]:
	D.append(np.outer(v,v))

rho = sum(D)/basis.Ns


rho_d = np.ones(basis.Ns)/basis.Ns

system_state = {"rho_d":rho_d,"V_rho":V}
print V
print '--------'

print ent_entropy(rho,basis,chain_subsys=[0])["Sent"]
#print 
print ent_entropy(system_state,basis,chain_subsys=[0])["Sent"]

exit()





print 
D_red = []
for d in D:
	Sent = ent_entropy(d,basis,chain_subsys=[0],DM="chain_subsys")
	d_red = Sent["DM_chain_subsys"]
	print Sent["Sent"]
	print d_red
	D_red.append(d_red)
	print

print
rho_red = sum(D_red)/basis.Ns

print -np.trace(rho_red*np.log(rho_red))

