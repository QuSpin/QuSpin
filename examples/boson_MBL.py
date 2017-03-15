from quspin.basis import boson_basis_1d,spin_basis_1d
from quspin.operators import hamiltonian,exp_op
from quspin.tools.measurements import obs_vs_time
import numpy as np
import matplotlib.pyplot as plt


N=3
L=2*N

start=0
stop=10
num=100
endpoint=True



basis = boson_basis_1d(L,Nb=N)
print basis.Ns

np.random.seed()

U_1 = [[1.0,i,i] for i in range(L)]
U_2 = [[-1.0,i] for i in range(L)]
t = [[1.0,i,(i+1)] for i in range(L-1)]

sublat_A = [[1.0,i] for i in range(0,L,2)]
sublat_B = [[1.0,i] for i in range(1,L,2)]

static = [["+-",t],["-+",t],["nn",U_1],["n",U_2]]
dynamic = []
static,dynamic = basis.expanded_form(static,dynamic)

H0=hamiltonian(static,dynamic,basis=basis,dtype=np.float32)
nA = hamiltonian([["n",sublat_A]],[],basis=basis,dtype=np.float32)
nB = hamiltonian([["n",sublat_B]],[],basis=basis,dtype=np.float32)

psi_i = np.zeros((basis.Ns),dtype=np.float32)
N0 = sum((N+1)**i for i in range(0,L,2))
i = basis.index(N0)

psi_i[i] = 1.0



times = np.linspace(start,stop,num=num,endpoint=endpoint)


def realization(H0,vasis,psi_i,w,start,stop,num,endpoint,i):
	print i
	disorder = [[np.random.uniform(-w,w),i] for i in range(L)]
	Hl = hamiltonian([["n",disorder]],[],basis=H0.basis,dtype=np.float32,check_pcon=False,check_herm=False,check_symm=False)
	expH = exp_op(H0+Hl,a=-1j,start=start,stop=stop,num=num,endpoint=endpoint,iterate=True)

	times = expH.grid

	psi_t = expH.dot(psi_i)
	obs = obs_vs_time(psi_t,times,{"nA":nA,"nB":nB},basis=basis,Sent_args=dict(sparse=True,sub_sys_A=[0]))
	n = (obs["nA"]-obs["nB"])/N

	return n,obs["Sent_time"]["Sent"]

nbin=100
nbs=1000
data = [realization(H0,basis,psi_i,5.0,start,stop,num,endpoint,i) for i in range(nbin)]
n_data,S_data = zip(*data)

n_data = np.vstack(n_data)
S_data = np.vstack(S_data)

n = n_data.mean(axis=0)
bootstraps = np.random.choice(n_data.shape[0],size=(nbs,))
dn = np.sqrt(((n_data[bootstraps] - n)**2).sum(axis=0)/(nbs*nbin))

S = S_data.mean(axis=0)
bootstraps = np.random.choice(S_data.shape[0],size=(nbs,))
dS = np.sqrt(((S_data[bootstraps] - n)**2).sum(axis=0)/(nbs*nbin))


plt.errorbar(times,n,dn,marker=".",color="blue")
plt.figure()

plt.errorbar(times,S,dS,marker=".",color="blue")
plt.figure()


nbin=100
nbs=1000
data = [realization(H0,basis,psi_i,1.0,start,stop,num,endpoint,i) for i in range(nbin)]
n_data,S_data = zip(*data)

n_data = np.vstack(n_data)
S_data = np.vstack(S_data)

n = n_data.mean(axis=0)
bootstraps = np.random.choice(n_data.shape[0],size=(nbs,))
dn = np.sqrt(((n_data[bootstraps] - n)**2).sum(axis=0)/(nbs*nbin))

S = S_data.mean(axis=0)
bootstraps = np.random.choice(S_data.shape[0],size=(nbs,))
dS = np.sqrt(((S_data[bootstraps] - n)**2).sum(axis=0)/(nbs*nbin))


plt.errorbar(times,n,dn,marker=".",color="green")
plt.figure()

plt.errorbar(times,S,dS,marker=".",color="green")
plt.show()




