from quspin.basis import boson_basis_1d,spin_basis_1d
from quspin.operators import hamiltonian,exp_op
from quspin.tools.measurements import obs_vs_time
import numpy as np
import matplotlib.pyplot as plt


N=10
L=2*N

start=0
stop=10
num=20
endpoint=True



basis = boson_basis_1d(L,Nb=N)


np.random.seed()

U_1 = [[1.0,i,i] for i in range(L)]
U_2 = [[-1.0,i] for i in range(L)]
t = [[1.0,i,(i+1)] for i in range(L-1)]

I_list = [[(-1.0)**i,i] for i in range(L)]

static = [["+-",t],["-+",t],["nn",U_1],["n",U_2]]
dynamic = []
static,dynamic = basis.expanded_form(static,dynamic)

H0=hamiltonian(static,dynamic,basis=basis,dtype=np.float32)
I = hamiltonian([["n",I_list]],[],basis=basis,dtype=np.float32)/N

psi_i = np.zeros((basis.Ns),dtype=np.float32)
N0 = sum((N+1)**i for i in range(0,L,2))
i = basis.index(N0)

psi_i[i] = 1.0



times = np.linspace(start,stop,num=num,endpoint=endpoint)


def realization(H0,basis,psi_i,w,I,start,stop,num,endpoint,i):
	print i
	disorder = [[np.random.uniform(-w,w),i] for i in range(L)]
	Hl = hamiltonian([["n",disorder]],[],basis=H0.basis,dtype=np.float32,check_pcon=False,check_herm=False,check_symm=False)
	expH = exp_op(H0+Hl,a=-1j,start=start,stop=stop,num=num,endpoint=endpoint,iterate=True)

	times = expH.grid

	psi_t = expH.dot(psi_i)
	obs = obs_vs_time(psi_t,times,{"I":I},basis=basis)#,Sent_args=dict(sparse=True,sub_sys_A=[0]))


	try:
		return obs["I"],obs["Sent_time"]["Sent"]
	except KeyError:
		return obs["I"]

nbin=100
nbs=1000
n_data = [realization(H0,basis,psi_i,5.0,I,start,stop,num,endpoint,i) for i in range(nbin)]
#n_data,S_data = zip(*n_data)

n_data = np.vstack(n_data)
n_1 = n_data.mean(axis=0)
bootstraps = np.random.choice(n_data.shape[0],size=(nbs,))
dn_1 = np.sqrt(((n_data[bootstraps] - n_1)**2).sum(axis=0)/(nbs*nbin))

"""
S_data = np.vstack(S_data)
S_1 = S_data.mean(axis=0)
bootstraps = np.random.choice(S_data.shape[0],size=(nbs,))
dS_1 = np.sqrt(((S_data[bootstraps] - S_1)**2).sum(axis=0)/(nbs*nbin))
"""

nbin=100
nbs=1000
n_data = [realization(H0,basis,psi_i,1.0,I,start,stop,num,endpoint,i) for i in range(nbin)]
#n_data,S_data = zip(*n_data)

n_data = np.vstack(n_data)
n_2 = n_data.mean(axis=0)
bootstraps = np.random.choice(n_data.shape[0],size=(nbs,))
dn_2 = np.sqrt(((n_data[bootstraps] - n_2)**2).sum(axis=0)/(nbs*nbin))

"""
S_data = np.vstack(S_data)
S_2 = S_data.mean(axis=0)
bootstraps = np.random.choice(S_data.shape[0],size=(nbs,))
dS_2 = np.sqrt(((S_data[bootstraps] - S_2)**2).sum(axis=0)/(nbs*nbin))
"""

plt.errorbar(times,n_1,dn_1,marker=".",color="green")
plt.errorbar(times,n_2,dn_2,marker=".",color="blue")
#plt.figure()
#plt.errorbar(times,S_1,dS_1,marker=".",color="green")
#plt.errorbar(times,S_2,dS_2,marker=".",color="blue")
plt.show()




