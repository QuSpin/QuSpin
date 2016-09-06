from qspin.operators import hamiltonian,exp_op # Hamiltonian and matrix exp
from qspin.basis import spin_basis_1d # Hilbert space spin basis
from qspin.tools.measurements import obs_vs_time, diag_ensemble # t_dep measurements
from qspin.tools.Floquet import Floquet, Floquet_t_vec # Floquet Hamiltonian
import numpy as np # generic math functions
#
##### define model parameters #####
L=2 # system size
J=1.0 # spin interaction
g=0.809 # transverse field
h=0.9045 # parallel field
Omega=4.5 # drive frequency
#
##### set up alternating Hamiltonians #####
# define time-reversal symmetric periodic step drive
def drive(t,Omega):
	return np.sign(np.cos(Omega*t))
drive_args=[Omega]
# compute basis in the 0-total momentum and +1-parity sector
basis=spin_basis_1d(L=L) #,kblock=0,pblock=1)
# define operators with PBC
x_field_pos=[[+g,i]	for i in range(L)]
x_field_neg=[[-g,i]	for i in range(L)]
z_field=[[h,i]		for i in range(L)]
J_nn=[[J,i,(i+1)%L] for i in range(L)] # PBC


static=[["++",J_nn],["y",z_field]]
dynamic=[["-y",J_nn,drive,drive_args],["y",x_field_neg,drive,drive_args]]
Obs = hamiltonian(static,dynamic,basis=basis,check_herm=False)



# static and dynamic parts
static=[["zz",J_nn],["z",z_field],["x",x_field_pos]]
dynamic=[["zz",J_nn,drive,drive_args],
		 ["z",z_field,drive,drive_args],["x",x_field_neg,drive,drive_args]]
# compute Hamiltonians
H=0.5*hamiltonian(static,dynamic,dtype=np.float64,basis=basis)


#
##### set up second-order van Vleck Floquet Hamiltonian #####
# zeroth-order term
Heff_0=0.5*hamiltonian(static,[],dtype=np.float64,basis=basis)
# second-order term
Heff2_term_1=[[J**2*g,i,(i+1)%L,(i+2)%L] for i in range(L)] # PBC
Heff2_term_2=[[J*g*h,i,(i+1)%L] for i in range(L)] # PBC
Heff2_term_3=[[-J*g**2,i,(i+1)%L] for i in range(L)] # PBC
Heff2_term_4=[[J**2*g+0.5*h**2*g,i] for i in range(L)]
Heff2_term_5=[[0.5*h*g**2,i] for i in range(L)]
# define static part
Heff_static=[["zxz",Heff2_term_1],
			 ["xz",Heff2_term_2],["zx",Heff2_term_2],
			 ["yy",Heff2_term_3],["zz",Heff2_term_2],
			 ["x",Heff2_term_4],
			 ["z",Heff2_term_5]							] 
# compute van Vleck Hamiltonian
Heff_2=hamiltonian(Heff_static,[],dtype=np.float64,basis=basis)
Heff_2*=-np.pi**2/(12.0*Omega**2)
# zeroth + second order van Vleck Floquet Hamiltonian
Heff_02=Heff_0+Heff_2
#
##### set up second-order van Vleck Kick operator #####
Keff2_term_1=[[J*g,i,(i+1)%L] for i in range(L)] # PBC
Keff2_term_2=[[h*g,i] for i in range(L)]
# define static part
Keff_static=[["zy",Keff2_term_1],["yz",Keff2_term_1],["y",Keff2_term_2]]
Keff_02=hamiltonian(Keff_static,[],dtype=np.complex128,basis=basis)
Keff_02*=-np.pi**2/(8.0*Omega**2)
#
##### rotate Heff to stroboscopic basis #####
import scipy.sparse.linalg as _sla
V_K = _sla.expm(-1j*Keff_02.tocsr())
FH_02 = hamiltonian( [(V_K.dot(Heff_02)).dot(V_K.T.conj())],[] )

HF_02 = exp_op(-1j*Keff_02,iterate=False,start=1.0,stop=1.0,num=1).sandwich(Heff_02) # e^{-1j*Keff_02} Heff_02 e^{+1j*Keff_02}
HF_02_2 = Heff_02.rotate( Keff_02, a=-1j,iterate=True,start=1.0,stop=1.0,num=1)


for i in HF_02:
	print np.linalg.norm( FH_02.todense() - i.todense() )
	for j in HF_02_2:
		print  np.linalg.norm( j.todense() - i.todense() )

#print  np.linalg.norm( HF_02_2.todense() - HF_02.todense() )

#print exp_op(Keff_02,a=-1j,time=0).__class__.__name__


"""
print np.linalg.norm( (V_K.dot(Heff_02)).todense() -  exp_op(-1j*Keff_02).dot(Heff_02).todense() )
print np.linalg.norm( Heff_02.dot(exp_op(1j*Keff_02)).todense() -  exp_op(1j*Keff_02).rdot(Heff_02).todense() )
print np.linalg.norm( Heff_02.dot(V_K.T.conj()).todense() -  exp_op(1j*Keff_02).rdot(Heff_02).todense() )
print np.linalg.norm( FH_02.todense() -  exp_op(-1j*Keff_02).dot(Heff_02).dot(exp_op(1j*Keff_02)).todense() )
print np.linalg.norm( FH_02.todense() - HF_02.todense() )
"""



"""
h1 = exp_op(Obs) #.dot(FH_02)
print h1.O.todense()
print "------------"
print np.around(h1.expm().todense(),1)
print "------------"
print np.around(h1.H.expm().todense(),1)
"""

exit()

hdot = h1.dot(Keff_02)
#hrdot = Keff_02.dot(h1)
#print type(hrdot)

exit()

print type(exp_op(1j*Keff_02))
print isinstance(exp_op(1j*Keff_02), exp_op)

#print np.linalg.norm( (V_K.dot(Heff_02)).todense() -  exp_op(-1j*Keff_02).dot(Heff_02).todense() )
print np.linalg.norm( Heff_02.dot(exp_op(1j*Keff_02)).todense() -  exp_op(1j*Keff_02).rdot(Heff_02).todense() )
#print np.linalg.norm( Heff_02.dot(V_K.T.conj()).todense() -  exp_op(1j*Keff_02).rdot(Heff_02).todense() )
#print np.linalg.norm( FH_02.todense() -  exp_op(1j*Keff_02).rdot(exp_op(-1j*Keff_02).dot(Heff_02)).todense() )



#EF_02, psi_i  = HF_02.eigsh(k=1,sigma=-100)

