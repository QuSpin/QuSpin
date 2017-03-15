import numpy as np
from quspin.basis import tensor_basis,fermion_basis_1d
from quspin.operators import hamiltonian,exp_op
from quspin.tools.measurements import obs_vs_time



L = 10
N = L//2
w = 1.0
J = 1.0
U = 10.0

start=0
stop=100
num=101

n_real = 10
n_boot = 10*n_real

N_up = N//2
N_down = N-N_up

 


basis_up = fermion_basis_1d(L,Nf=N_up)
basis_down = fermion_basis_1d(L,Nf=N_down)
basis = tensor_basis(basis_up,basis_down)



J_right = [[J,i,i+1] for i in range(L-1)]
J_left = [[-J,i,i+1] for i in range(L-1)]
U_list = [[U,i,i] for i in range(L)]


sublat_list = [[(-1)**i,i] for i in range(0,L)]

static = [	
			["+-|",J_left], # up hopping
			["-+|",J_right], 
			["|+-",J_left], # down hopping
			["|-+",J_right],
			["n|n",U_list], # onsite interaction
			["n|",V_list], # onsite potential
		 ]
dynamic = []


# set up hamiltonian, evolution operator, and observable
H0 = hamiltonian(static,dynamic,basis=basis,check_pcon=False,check_symm=False)
I = hamiltonian([["n|",sublat_list],["|n",sublat_list]],[],basis=basis,check_pcon=False,check_symm=False)/N

# setting up initial state
s_up = sum((1 << i) for i in range(2,L,2*L//N))
s_down = sum((1 << i) for i in range(0,L,2*L//N))
state = list((">{:0"+str(L)+"b},{:0"+str(L)+"b}|").format(s_up,s_down))
state.reverse()
print "".join(state)

i_0 = basis.index(s_up,s_down)
psi_0 = np.zeros(basis.Ns)
psi_0[i_0] = 1.0

def realization(H0,I,psi_0,w,start,stop,num,i):
	print i
	basis = H0.basis

	V_list = [[np.random.uniform(-w,w),i] for i in range(L)]
	H = H0 + hamiltonian([["n|",V_list]],[],basis=basis)
	expH = exp_op(H,start=start,stop=stop,num=num,iterate=True,endpoint=True)


	times = expH.grid

	psi_t = expH.dot(psi_0)

	obs_t = obs_vs_time(psi_t,times,dict(I=I))

	return obs_t["I"]



I_data = np.vstack((realization(H0,I,psi_0,w,start,stop,num) for i in range(n_real)))

bootstraps = np.choice(n_real,size=(n_real,n_boot))

I_boot = I[:,bootstraps]
