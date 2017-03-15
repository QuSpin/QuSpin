import numpy as np
from quspin.basis import tensor_basis,fermion_basis_1d
from quspin.operators import hamiltonian,exp_op
from quspin.tools.measurements import obs_vs_time



L = 16
N = L//2
w = 10.0
J = 1.0
U = 4.0
beta = 0.721
phi = np.e

#if L%4 != 0:
#	raise ValueError("L must be multiple of 4")


N_up = N//2
N_down = N-N_up

 


basis_up = fermion_basis_1d(L,Nf=N_up)
basis_down = fermion_basis_1d(L,Nf=N_down)
basis = tensor_basis(basis_up,basis_down)

print basis.Ns

J_right = [[J,i,i+1] for i in range(L-1)]
J_left = [[-J,i,i+1] for i in range(L-1)]
U_list = [[U,i,i] for i in range(L)]
V_list = [[w*np.sin(2*beta*np.pi*i+phi),i] for i in range(L)]

sublat_A_list = [[1,i] for i in range(0,L,2)]
sublat_B_list = [[1,i] for i in range(1,L,2)]

static = [	
			["+-|",J_left], # up hopping
			["-+|",J_right], 
			["|+-",J_left], # down hopping
			["|-+",J_right],
			["n|n",U_list], # onsite interaction
			["n|",V_list], # onsite potential
			["|n",V_list], 
		 ]
dynamic = []


# set up hamiltonian, evolution operator, and observable
H = hamiltonian(static,dynamic,basis=basis,check_pcon=False,check_symm=False)
expH = exp_op(H,a=-1j,start=0,stop=100,num=1001,endpoint=True,iterate=True)
n_A = hamiltonian([["n|",sublat_A_list],["|n",sublat_A_list]],[],basis=basis,check_pcon=False,check_symm=False)
n_B = hamiltonian([["n|",sublat_B_list],["|n",sublat_B_list]],[],basis=basis,check_pcon=False,check_symm=False)
I = (n_A - n_B)/N

times = expH.grid

# setting up initial state
s_up = sum((1 << i) for i in range(2,L,2*L//N))
s_down = sum((1 << i) for i in range(0,L,2*L//N))
state = list(">{:010b},{:010b}|".format(s_up,s_down))
state.reverse()
print "".join(state)

i_0 = basis.index(s_up,s_down)

psi_0 = np.zeros(basis.Ns)
psi_0[i_0] = 1.0

psi_t = expH.dot(psi_0)

obs_t = obs_vs_time(psi_t,times,dict(I=I,n_A=n_A,n_B=n_B),disp=True)


import matplotlib.pyplot as plt
print obs_t["I"][0],obs_t["n_A"][0],obs_t["n_B"][0]
plt.plot(times,obs_t["I"],marker=".")
plt.show()

