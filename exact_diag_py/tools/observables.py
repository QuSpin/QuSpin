# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import scipy.sparse as _sp
import numpy as _np

# needed for isinstance only
from exact_diag_py.hamiltonian import hamiltonian
from exact_diag_py.basis import spin_basis_1d,photon_basis

import warnings

#__all__ = ["Entanglement_entropy", "Diag_Ens_Observables", "Kullback_Leibler_div", "Observable_vs_time", "Mean_Level_Spacing"]

def Entanglement_entropy_photon(L,Nph,psi,chain_subsys=None,basis=None,alpha=1.0,DM=False,chain_symm=False):
	# psi: pure quantum state
	# subsys: a list of integers modelling the site numbers of the subsystem
	# basis: the basis of the Hamiltonian: needed only when symmetries are used
	# alpha: Renyi parameter
	# DM: if on returns the reduced density matrix corresponding to psi

	if not(type(L) is int):
		raise TypeError("System size 'L' must be a positive integer!")

	if isinstance(alpha,complex) or alpha < 0.0:
		raise TypeError("Renyi entropy parameter 'alpha' must be real-valued and non-negative!")


	if chain_subsys is None: 
		chain_subsys=[i for i in xrange( int(L) )]
		warnings.warn("subsystem automatically set to the entire chain.")
	elif not isinstance(chain_subsys,list):
		raise TypeError("'subsys' must be a list of integers to lable the lattice site numbers of the subsystem!")
	elif min(chain_subsys) < 0:
		raise TypeError("'subsys' must be a list of nonnegative numbers!")
	elif max(chain_subsys) > L-1:
		raise TypeError("'subsys' contains sites exceeding the total lattice site number!")


	# initiate variables
	variables = ["Sent"]
	
	if DM=='DM_photon':
		variables.append("DM_photon")
		print "Density matrix calculation is enabled. The reduced DM is produced in the full photon basis."
	elif DM=='DM_chain':
		variables.append("DM_chain")
		print "Density matrix calculation is enabled. The reduced DM is produced in the chain sigma^z basis."
	elif DM==True:
		variables.append("DM_photon")
		variables.append("DM_chain")
		print "Density matrix calculation is enabled. The reduced DM_chain (DM_photon) is produced in the full chain (HO) basis."


	#calculate H-space dimensions of the subsystem and the system
	L_A = len(chain_subsys)
	Ns_A = 2**L_A

	# define lattice indices putting the subsystem to the left
	system = chain_subsys[:]
	[system.append(i) for i in xrange(L) if not i in chain_subsys]
	
	# re-write the state in the initial basis
	Ns_spin = 2**L
	if len(psi)<2**L*(Nph+1):
		if basis:
			if chain_symm: #basis must be the chain basis except when subsys length < L
				if isinstance(basis,spin_basis_1d):
					Ns_spin = basis.Ns
					if L_A < L:
						raise TypeError("Chain subsystem size < L: please parse the non-symmetrised (full) photon basis and set 'chain_symm=False'!")
				else:
					raise TypeError("'chain_symm' is 'True': basis parsed must be the symmetrised chain basis!")
			else: #basis must be the particle conserving photon basis
				if isinstance(basis,photon_basis):
					psi = _np.asarray( basis.get_vec(psi,sparse=False) )#[:,0]
					Ns_spin = 2**L
				else:
					raise TypeError("'chain_symm' is 'False': basis parsed must be the non-symmetrised (full) photon basis!")
		else:
			raise TypeError("Basis contains symmetries; Please parse the basis variable!")
	del basis


	if L_A==L:
		# reshape state vector psi
		v = _np.reshape(psi, (Ns_spin,Nph+1) ).T
		del psi
	else:
		# performs 2) and 3)
		
		chain_dim_per_site = [2 for i in xrange(L)]
		chain_dim_per_site.append(Nph+1) 
		
		v = _np.reshape(psi, tuple(chain_dim_per_site) )
		del psi, chain_dim_per_site
		# performs 4)
		system.append(len(system))
		v = _np.transpose(v, axes=system) 
		# performs 5)
		v = _np.reshape(v, ( Ns_A, 2**(L-L_A)*(Nph+1) ) )
	
	del system, chain_subsys
	
	
	# apply singular value decomposition
	if DM==False:
		gamma = _la.svd(v, compute_uv=False, overwrite_a=True, check_finite=True)
	elif DM=='DM_photon':
		U, gamma, _ = _la.svd(v, full_matrices=False, overwrite_a=True, check_finite=True)
		DM_photon = _np.einsum('ij,j,kj->ik',U,gamma**2,U.conjugate() )
		del U
	elif DM=='DM_chain':
		_, gamma, V = _la.svd(v, full_matrices=False, overwrite_a=True, check_finite=True)
		DM_chain = _np.einsum('ji,j,jk->ik',V.conjugate(),gamma**2,V ) 
		del V
	else:
		U, gamma, V = _la.svd(v, full_matrices=False, overwrite_a=True, check_finite=True)
		DM_photon = _np.einsum('ij,j,kj->ik',U,gamma**2,U.conjugate() )
		DM_chain = _np.einsum('ji,j,jk->ik',V.conjugate(),gamma**2,V ) 
		
		del U,V	
	del v

	
	# calculate Renyi entropy
	if any(gamma == 1.0):
		Sent = 0.0
	else:
		if any(gamma == 0.0):
			# remove all zero entries to prevent the log from giving an error
			gamma = gamma[gamma!=0.0]

		if alpha == 1.0:
			Sent = -( abs(gamma)**2).dot( 2*_np.log( abs(gamma)  ) ).sum()
		else:
			Sent =  1./(1-alpha)*_np.log( (gamma**alpha).sum() )

	# define dictionary with outputs
	return_dict = {}
	for i in variables:
		return_dict[i] = vars()[i]

	return return_dict

def Entanglement_entropy(L,psi,chain_subsys=None,basis=None,alpha=1.0, DM=False):
	# psi: pure quantum state
	# subsys: a list of integers modelling the site numbers of the subsystem
	# basis: the basis of the Hamiltonian: needed only when symmetries are used
	# alpha: Renyi parameter
	# DM: if on returns the reduced density matrix corresponding to psi

	if not(type(L) is int):
		raise TypeError("System size 'L' must be a positive integer!")

	if isinstance(alpha,complex) or alpha < 0.0:
		raise TypeError("Renyi entropy parameter 'alpha' must be real-valued and non-negative!")

	if chain_subsys is None: 
		chain_subsys=[i for i in xrange( int(L/2) )]
		warnings.warn("subsystem automatically set to contain sites {}.".format(chain_subsys))
	elif not isinstance(chain_subsys,list):
		raise TypeError("'subsys' must be a list of integers to lable the lattice site numbers of the subsystem!")
	elif min(chain_subsys) < 0:
		raise TypeError("'subsys' must be a list of nonnegative numbers!")
	elif max(chain_subsys) > L-1:
		raise TypeError("'subsys' contains sites exceeding the total lattice site number!")
	


	variables = ["Sent"]
	
	if _np.any(DM):
		variables.append("DM")
		print "Density matrix calculation is enabled. The reduced DM is produced in the full basis containing all states of the subsystem."


	
	# re-write the state in the initial basis
	if len(psi)<2**L:
		if basis:
			psi = _np.asarray( basis.get_vec(psi,sparse=False) )
		else:
			raise TypeError("Spin basis contains symmetries; Please parse the basis variable!")
	del basis

	#calculate H-space dimensions of the subsystem and the system
	L_A = len(chain_subsys)
	Ns_A = 2**L_A
	Ns = len(psi)

	# define lattice indices putting the subsystem to the left
	system = chain_subsys[:]
	[system.append(i) for i in xrange(L) if not i in chain_subsys]


	'''
	the algorithm for the entanglement entropy of an arbitrary subsystem goes as follows:

	1) the initial state psi has 2^L entries corresponding to the spin-z configs
	2) reshape psi into a 2x2x2x2x...x2 dimensional array (L products in total). Call this array v.
	3) v should satisfy the property that v[0,1,0,0,0,1,...,1,0], total of L entries, should give the entry of psi 
	   along the the spin-z basis vector direction (0,1,0,0,0,1,...,1,0). This ensures a correspondence of the v-indices
	   (and thus the psi-entries) to the L lattice sites.
	4) fix the lattice sites that define the subsystem L_A, and reshuffle the array v according to this: e.g. if the 
 	   subsystem consistes of sites (k,l) then v should be reshuffled such that v[(k,l), (all other sites)]
 	5) reshape v[(k,l), (all other sites)] into a 2D array of dimension ( L_A x L/L_A ) and proceed with the SVD as below  
	'''

	# performs 2) and 3)
	v = _np.reshape(psi, tuple([2 for i in xrange(L)] ) )
	del psi
	# performs 4)
	v = _np.transpose(v, axes=system) 
	# performs 5)
	v = _np.reshape(v, ( Ns_A, Ns/Ns_A) )
	
	del system, chain_subsys
	
	# apply singular value decomposition
	if DM==False:
		gamma = _la.svd(v, compute_uv=False, overwrite_a=True, check_finite=True)
	else:
		U, gamma, _ = _la.svd(v, full_matrices=False, overwrite_a=True, check_finite=True)
		# calculate reduced density matrix DM	
		#DM = reduce( _np.dot, [U, _np.diag(gamma**2), U.T.conjugate() ] ) 
		#----  
		#DM = sum( _np.outer(_np.einsum('ij,j->ij',U,gamma**2)[:,_j], U.conjugate()[:,_j] ) for _j in xrange(len(gamma)) )
		#U=None # cannot delete U due to sum()
		#----
		DM = _np.einsum('ij,j,kj->ik',U,gamma**2,U.conjugate() ) 
		
		#print _np.linalg.norm( DM - DM2   )
		
		del U	
		print "NEED TO TEST this reduced DM against something known!!!"
	del v

	# calculate Renyi entropy
	if any(gamma == 1.0):
		Sent = 0.0
	else:
		if any(gamma == 0.0):
			# remove all zeros to prevent the log from giving an error
			gamma = gamma[gamma!=0.0]
			#gamma = _np.array([_np.finfo(_np.float64).eps if _i==0 else _i for _i in gamma])

		if alpha == 1.0:
			Sent = -( abs(gamma)**2).dot( 2*_np.log( abs(gamma)  ) ).sum()
		else:
			Sent =  1./(1-alpha)*_np.log( (gamma**alpha).sum() ) 

	# define dictionary with outputs
	return_dict = {}
	for i in range(len(variables)):
		return_dict[variables[i]] = vars()[variables[i]]

	return return_dict


def Project_Operator(Obs,reduced_basis,dtype=_np.complex128,Proj=False):

	variables = ["Proj_Obs"]

	if _np.any(Proj):
		variables.append("Proj")

	Proj = reduced_basis.get_proj(dtype=dtype)

	Proj_Obs = Proj.T.conj()*Obs*Proj

	# define dictionary with outputs
	return_dict = {}
	for i in range(len(variables)):
		return_dict[variables[i]] = vars()[variables[i]]

	return return_dict


def Diag_Ens_Observables(L,V1,E1,V2,betavec=[],alpha=1.0,Obs=False,Ed=False,S_double_quench=False,Sd_Renyi=False,deltaE=False):
	# V1, V2:  matrices with pre and post quench eigenbases
	# E1: vector of energies of pre-quench
	# Obs: any hermitian observable
	# betavec: vector of inverse temperatures
	# alpha: Renyi entropy parameter

	if not(type(L) is int):
		raise TypeError("System size 'L' must be a positive integer!")

	if isinstance(alpha,complex) or alpha < 0.0:
		raise TypeError("Renyi entropy parameter 'alpha' must be real-valued and non-negative!")

	if V1.shape != V2.shape:
		raise TypeError("Unitary eigenstate matrices 'V1' and 'V2' must have the same shape!")
	elif len(V1[0,:]) != len(E1):
		raise TypeError("Number of eigenstates in 'V1' must equal number of eigenvalues in 'E1'!")

	variables_GS = []
	variables_T = []

	if Obs is not False:
		if _la.norm(Obs.todense().T.conj() - Obs.todense()) > 1E4*_np.finfo(eval('_np.'+Obs[0,0].dtype.name)).eps:
			raise ValueError("'Obs' is not hermitian!")
		variables_GS.append("Obs_GS")
		variables_T.append("Obs_T")
	if Ed:
		warnings.warn("The value of E_Tinf depends on the symmetries used!",UserWarning)
		variables_GS.append("Ed_GS")
		variables_GS.append("E_Tinf")
		variables_T.append("Ed_T")
		variables_T.append("E_Tave")
	if S_double_quench:
		variables_GS.append("S_double_quench_GS")
		variables_T.append("S_double_quench_T")
	if Sd_Renyi:
		variables_GS.append("Sd_Renyi_GS")
		variables_T.append("Sd_Renyi_T")
	if S_double_quench or Sd_Renyi:
		variables_GS.append("S_Tinf")
	if deltaE:
		variables_GS.append("deltaE_GS")
		variables_T.append("deltaE_T")

	if not variables_GS:
		warnings.warn("No observables were requested: ..exiting", UserWarning)
		return None

	
	Ns = len(E1) # Hilbert space dimension

	if betavec:
		warnings.warn("All thermal expectation values depend statistically on the symmetry used via the available number of states as part of the system-size dependence!",UserWarning)
		#define thermal density matrix w.r.t. the basis V1	
		rho = _np.zeros((Ns,len(betavec)),dtype=type(betavec[0]) )
		for i in xrange(len(betavec)):
			rho[:,i] =_np.exp(-betavec[i]*(E1-E1[0]))/sum(_np.exp(-betavec[i]*(E1-E1[0]) ) )

	# diagonal matrix elements of Obs in the basis V2
	if Obs is not False:
		O_mm = _np.real( _np.einsum( 'ij,ji->i', V2.transpose().conj(), Obs.dot(V2) ) )
	#probability amplitudes
	a_n = V1.conjugate().transpose().dot(V2);
	del V1
	del V2
	# transition rates matrix (H1->H2)
	T_nm = _np.real( _np.multiply(a_n, a_n.conjugate()) )
	T_nm[T_nm<=1E-16] = _np.finfo(float).eps	
	# probability rates matrix (H1->H2->H1)
	if Ed or S_double_quench:
		pn = T_nm.dot(T_nm.transpose() )


	# diagonal ens expectation value of Obs in post-quench basis
	if Obs is not False:
		Obs_GS = T_nm[0,:].dot(O_mm)/L # GS
		if betavec:
			Obs_T = (_np.einsum( 'ij,j->i', T_nm, O_mm )/L ).dot(rho) # finite-temperature


	#calculate diagonal energy <H1> in long time limit
	if Ed:
		Ed_GS = pn[0,:].dot(E1)/L  # GS
		if betavec:
			Ed_T  = (pn.dot(E1)/L ).dot(rho) # finite-temperature
			E_Tave = E1.dot(rho)/L # average energy density
		E_Tinf = E1.sum()/Ns/L # infinite temperature

	#calculate double-quench entropy (H1->H2->H1)
	if S_double_quench:
		S_double_quench_GS = -pn[0,:].dot(_np.log(pn[0,:]))/L # GS
		if betavec:
			S_double_quench_T  = (_np.einsum( 'ij,ji->i', -pn,_np.log(pn) )/L ).dot(rho) # finite-temperature
	

	# free up memory
	if Ed or S_double_quench:
		del pn

	# calculate diagonal Renyi entropy for parameter alpha: equals (Shannon) entropy for alpha=1: (H1->H2)
	if Sd_Renyi:
		if alpha != 1.0:
			#calculate diagonal (Renyi) entropy for parameter alpha (H1->H2)
			Sd_Renyi_GS = 1/(1-alpha)*_np.log(_np.power( T_nm[0,:], alpha ).sum() )/L  # # GS
			if betavec:
				Sd_Renyi_T = 1/(1-alpha)*(_np.log(_np.power( T_nm, alpha ).sum(1)  )/L  ).dot(rho) # finite-temperature
		else:
			warnings.warn("Renyi entropy equals diagonal entropy.", UserWarning)
			Sd_Renyi_GS = -T_nm[0,:].dot(_np.log(T_nm[0,:]) ) /L # GS
			if betavec:
				Sd_Renyi_T = (np.einsum( 'ij,ji->i', -T_nm,_np.log(T_nm.transpose()) )/L ).dot(rho) # finite-temperature

	# infinite temperature entropy
	if S_double_quench or Sd_Renyi:
		S_Tinf = _np.log(2); 

	# calculate long-time energy fluctuations
	if deltaE:
		# calculate <H1^2>
		H1_mn2 = (a_n.conjugate().transpose().dot(_np.einsum('i,ij->ij',E1,a_n)) )**2
		del a_n
		_np.fill_diagonal(H1_mn2,0.0) 

		deltaE_GS = _np.real( reduce( _np.dot,[T_nm[0,:], H1_mn2, T_nm[0,:] ])  )/L**2  # GS
		if betavec:
			deltaE_T  = _np.real(_np.einsum( 'ij,ji->i', T_nm, H1_mn2.dot(T_nm.transpose()) )/(L**2) ).dot(rho) # finite-temperature
		# free up memory
		del T_nm
		del H1_mn2

	return_dict = {}
	for i in range(len(variables_GS)):
		return_dict[variables_GS[i]] = vars()[variables_GS[i]]
	if betavec:
		for i in range(len(variables_T)):
			return_dict[variables_T[i]] = vars()[variables_T[i]]
			

	
	return return_dict
		

def Kullback_Leibler_div(p1,p2):
	# p1,p2: probability distributions
	if len(p1) != len(p2):
		raise TypeError("The probability distributions 'p1' and 'p2' must have same size!")
	if len(p1.shape)!=1 or len(p2.shape)!=1:
		raise TypeError("The probability distributions 'p1' and 'p2' must have linear dimension!")

	p1 = _np.asarray(p1)
	p2 = _np.asarray(p2)

	if any(i<0.0 for i in p1) or any(i<=0.0 for i in p2):
		raise TypeError("All entries of the probability distributions 'p1' and 'p2' must be non-negative!")
	if any(p1==0.0):

		inds = _np.where(p1 == 0)[0]

		p1 = [_np.delete(p1,i) for i in inds][0]	
		p2 = [_np.delete(p2,i) for i in inds][0]


	return _np.multiply( p1, _np.log( _np.divide(p1,p2) ) ).sum()


def Observable_vs_time(psi,V,E,Obs,times,return_state=False):
	# psi: initial state
	# V2, E2: matrix w/ eigenbasis and vector of eogenvalues of post-quench Hamiltonian H2
	# Obs: observable of interest
	# times: vector with time values

	if len(V[0,:]) != len(E):
		raise TypeError("Number of eigenstates in 'V' must equal number of eigenvalues in 'E'!")
	elif len(psi) != len(E):
		raise TypeError("Variables 'psi' and 'E' must have the same dimension!")
	elif V.shape!=Obs.shape:
		raise TypeError("Sizes of 'V1' and 'Obs' must be equal!")

	if not isinstance(times,list):
		TypeError("Variable 'times' must be a list!")

	variables = ['Expt_time']

	if return_state==True:
		variables.append("psi_time")

	# project initial state onto basis V2
	c_n = V.conjugate().transpose().dot(psi)


	# define time-evolved state in basis V2
	def psit(a_n,t):
		# a_n: probability amplitudes
		# t: time vector

		return V.dot(_np.multiply(_np.exp(-1j*E*t), a_n ) )

	
	Lt = len(times)

	# preallocate state
	if return_state==True:
		psi_time = _np.zeros((len(E),Lt),dtype=_np.complex128)

	# preallocate expectation value
	Expt_time = _np.zeros((Lt),dtype=_np.float64)

	# loop over time vector
	for m in xrange(Lt):
		if return_state==True:
			psi_time[:,m] = psit(c_n,times[m])
		psi_t = psit(c_n,times[m])
		#print _np.real( reduce( _np.dot, [psi_t.conjugate().T, Obs, psi_t ]  )  )
		#print _np.real( _np.einsum('i,ij,j->',psi_t.conjugate().T, Obs, psi_t ) )
		Expt_time[m] = _np.real( _np.einsum('i,ij,j->',psi_t.conjugate().T, Obs.todense(), psi_t ) )

	return_dict = {}
	for i in range(len(variables)):
		return_dict[variables[i]] = vars()[variables[i]]

	return return_dict


def Mean_Level_Spacing(E):

	# compute consecutive E-differences
	sn = _np.diff(E)
	# check for degeneracies
	if len(_np.unique(E)) != len(E):
		raise ValueError("Degeneracies found in spectrum 'E'!")
	# calculate the ratios of consecutive spacings
	aux = _np.zeros((len(E)-1,2),dtype=_np.float64)

	aux[:,0] = sn
	aux[:,1] = _np.roll(sn,-1)

	return _np.mean(_np.divide( aux.min(1), aux.max(1) )[0:-1] )



