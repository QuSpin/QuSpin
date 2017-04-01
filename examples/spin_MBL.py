from quspin.basis import boson_basis_1d,spin_basis_1d,fermion_basis_1d,tensor_basis
from quspin.operators import hamiltonian,exp_op
from quspin.tools.measurements import obs_vs_time
import numpy as np
import matplotlib.pyplot as plt


L=6
S="1/2"
sps=2
Nup=L*(sps-1)/2

w=0.5
basis = spin_basis_1d(L,S=S,Nup=Nup)
print basis.Ns


np.random.seed()

Jzz = [[1.0,i,i+1] for i in range(L-1)]
Jxy = [[0.5,i,i+1] for i in range(L-1)]

stag = [[(-1.0)**i,i] for i in range(L)]

static = [["+-",Jxy],["-+",Jxy],["zz",Jzz]]
dynamic = []


H0=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)
Ms = hamiltonian([["z",stag]],[],basis=basis,dtype=np.float64)

psi_i = np.zeros((basis.Ns),dtype=np.float64)
N0 = sum((sps-1)*(sps)**i for i in range(0,L,2))
i = basis.index(N0)

psi_i[i] = 1.0


start=0
stop=40
num=100
endpoint=True

times = np.linspace(start,stop,num=num,endpoint=endpoint)


def realization(H0,basis,Ms,psi_i,w,start,stop,num,endpoint,i):
	print i
	disorder = [[np.random.uniform(-w,w),i] for i in range(L)]
	Hl = hamiltonian([["z",disorder]],[],basis=H0.basis,dtype=np.float32,check_pcon=False,check_herm=False,check_symm=False)
	expH = exp_op(H0+Hl,a=-1j,start=start,stop=stop,num=num,endpoint=endpoint)

	times = expH.grid

	psi_t = expH.dot(psi_i)
	Sent = basis.ent_entropy(psi_t)/(basis.L/2.0)
	obs = obs_vs_time(psi_t,times,{"Ms":Ms})
	return obs["Ms"]/L,Sent

nbin=100
nbs=2000
data = [realization(H0,basis,Ms,psi_i,w,start,stop,num,endpoint,i) for i in range(nbin)]
n_data,S_data = zip(*data)

n_data = np.vstack(n_data)
S_data = np.vstack(S_data)

n = n_data.mean(axis=0)
bootstraps = np.random.choice(n_data.shape[0],size=(nbs,))
dn = np.sqrt(((n_data[bootstraps] - n)**2).sum(axis=0)/(nbs*nbin))

S = S_data.mean(axis=0)
bootstraps = np.random.choice(S_data.shape[0],size=(nbs,))
dS = np.sqrt(((S_data[bootstraps] - n)**2).sum(axis=0)/(nbs*nbin))


plt.errorbar(times,n,dn,marker=".")
plt.figure()

plt.errorbar(times,S,dS,marker=".")
plt.show()



