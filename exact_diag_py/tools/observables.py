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

def Entanglement_entropy_photon(L,Nph,Ntot,psi,chain_subsys=None,basis=None,alpha=1.0,DM=False,chain_symm=False):
	"""
	Entanglement_entropy_photon(L,Nph,psi,chain_subsys=None,basis=None,alpha=1.0,DM=False,chain_symm=False) 

	This routine calculates the entanglement (Renyi) entropy of a pure chain-photon quantum state 'psi' 
	in a chain subsystem of arbitraty choice. It returns a dictionary in which the entanglement (Renyi) 
	entropy has the key 'Sent'. The arguments are:

	L: (compulsory) chain length. Always the first argument.

	Nph: (compulsory) number of photon states. 

	Ntot: (compulsory) number of total possible photons. 

	psi: (compulsory) a pure quantum state, to calculate the entanglement entropy of.

	chain_subsys: (optional) a list of site numbers defining uniquely the CHAIN subsystem of which 
			the entanglement entropy (reduced and density matrix) are calculated. Notice that the site 
			labelling of the chain goes as [0,1,....,L-1]. If not specified, the default subsystem 
			chosen is the entire chain [0,...,L-1]. If in addition symmetries as present, it is required 
			that 'chain_symm=False' whenever L_A = len(subsys) < L, and the density matrix (if on) is 
			returned in the full spin-z basis of the chain subsystem containing 2^L_A sites.

	basis: (semi-compulsory) basis of 'psi'. If no symmetry is invoked and if the basis of 'psi' contains 
			all 2^L states, one can ommit the basis argument. If the state 'psi' is written in a 
			symmetry-reduced basis, then one must also parse the basis in which 'psi' is given. 

	chain_symm: (semi-compulsory) if the Hamiltonian of the chain part of the photon-chain model has 
			symmetries used in the construction of the state 'psi', and if the 'chain_subsys' is the entire
			spin chain, then the chain symmetries are inherited by the density matrices whenever 
			'chain_symm = True'. Requires that 'basis' is parsed. 

	alpha: (optional) Renyi parameter alpha. The default is 'alpha=1.0', corresponding to von Neumann's entropy.

	DM: (optional) when set to 'True', the returned dictionary contains the reduced density matrices of the
			chain subsystems under the key 'DM_chain', and the photon + rest-of-chain subsystem under 
			the key 'DM_photon'. If 'chain_subsys' is not specified then 'DM_photon' is the density matrix 
			of the photon mode and 'DM_chain' is the chain DM. If only one DM is required, one can used
			'DM = 'DM_photon' or 'DM = DM_chain', respectively. The basis for the DM depends on the symmetries
			symmetries of psi.
	"""

	if not(type(L) is int):
		raise TypeError("System size 'L' must be a positive integer!")

	if (Ntot is not None) and (Nph is not None):
		raise TypeError("Only one of the parameters 'Ntot' or 'Nph' is allowed!")

	if isinstance(alpha,complex) or alpha < 0.0:
		raise TypeError("Renyi entropy parameter 'alpha' must be real-valued and non-negative!")


	if chain_subsys is None: 
		chain_subsys=[i for i in xrange( int(L) )]
		warnings.warn("subsystem automatically set to the entire chain.")
	elif not isinstance(chain_subsys,list):
		raise TypeError("'subsys' must be a list of integers to label the lattice site numbers of the subsystem!")
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
		if chain_symm:
			print "Density matrix calculation is enabled. If symmetries are on, reduced DM_chain (DM_photon) is produced in the symmetry-reduced chain (photon) basis."
		else:
			print "Density matrix calculation is enabled. The reduced DM_chain (DM_photon) is produced in the full chain (photon) basis."


	#calculate H-space dimensions of the subsystem and the system
	L_A = len(chain_subsys)
	Ns_A = 2**L_A

	# define lattice indices putting the subsystem to the left
	system = chain_subsys[:]
	[system.append(i) for i in xrange(L) if not i in chain_subsys]
	
	# re-write the state in the initial basis

	if ( (Nph is not None) and len(psi)<2**L*(Nph+1) ) or (Ntot is not None): #basis required
		if not isinstance(basis,photon_basis):
			raise TypeError("Basis contains symmetries; Please parse the basis variable!")


	if Nph is not None: # no total particle conservation
		if len(psi) < 2**L*(Nph+1): #chain symmetries present
			if chain_symm:
				if L_A < L:
					raise TypeError("'chain_symm' set to 'True': subsystem size must be < L!")
				else:
					psi = _np.asarray( basis.get_vec(psi,sparse=False,full_part=False) )
					Ns_spin = basis.chain_Ns
			else:
				psi = _np.asarray( basis.get_vec(psi,sparse=False,full_part=True) )
				Ns_spin = 2**L
		else:
			Ns_spin = 2**L

	elif Ntot is not None: # total particle-conservation
		Nph = Ntot
		if len(psi) < 2**L: #chain symemtries present
			if chain_symm:
				raise TypeError("'chain_symm' is incompatible with Ntot symmetry!")
			else:
				psi = _np.asarray( basis.get_vec(psi,sparse=False,full_part=True) )
				Ns_spin = 2**L
		else: # no chain symmetries present
			psi = _np.asarray( basis.get_vec(psi,sparse=False,full_part=True) )
			Ns_spin = basis.chain_Ns

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

def Entanglement_entropy(L,psi,chain_subsys=None,basis=None,alpha=1.0,DM=False):
	"""
	Entanglement_entropy(L,psi,chain_subsys=None,basis=None,alpha=1.0,DM=False) 

	This routine calculates the entanglement (Renyi) entropy of a pure quantum state 'psi' in a subsystem 
	of arbitraty choice. It returns a dictionary in which the entanglement (Renyi) entropy has the key 
	'Sent'. The arguments are:

	L: (compulsory) chain length. Always the first argument.

	psi: (compulsory) a pure quantum state, to calculate the entanglement entropy of. Always the second argument.

	basis: (semi-compulsory) basis of psi. If the state 'psi' is written in a symmetry-reduced basis, 
			then one must also parse the basis in which 'psi' is given. However, if no symmetry is invoked 
			and if the basis of 'psi' contains all 2^L states, one can ommit the basis argument.

	chain_subsys: (optional) a list of site numbers defining uniquely the subsystem of which 
			the entanglement entropy (reduced and density matrix) are calculated. Notice that the site 
			labelling of the chain goes as [0,1,....,L-1]. If not specified, the default subsystem 
			chosen is [0,...,floor(L/2)].

	alpha: (optional) Renyi parameter alpha. The default is 'alpha=1.0', corresponding to von Neumann's entropy.

	DM: (optional) when set to 'True', the returned dictionary contains the reduced density matrix under 
			the key 'DM'. Note that the reduced DM is written in the full basis over all 2^L_A states of 
			the subchain in question.
	"""

	if not(type(L) is int):
		raise TypeError("System size 'L' must be a positive integer!")

	if isinstance(alpha,complex) or alpha < 0.0:
		raise TypeError("Renyi entropy parameter 'alpha' must be real-valued and non-negative!")

	if chain_subsys is None: 
		chain_subsys=[i for i in xrange( int(L/2) )]
		warnings.warn("subsystem automatically set to contain sites {}.".format(chain_subsys))
	elif not isinstance(chain_subsys,list):
		raise TypeError("'subsys' must be a list of integers to label the lattice site numbers of the subsystem!")
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
	"""
	Project_Operator(Obs,reduced_basis,dtype=_np.complex128,Proj=False)

	This function takes an observable 'Obs' and a reduced basis 'reduced_basis' and projects 'Obs'
	onto the reduced basis. It returns a dictionary with keys 'Proj_Obs' and 'Proj' (optional).

	Obs: (compulsory) operator to be projected.

	reduced_basis: (compulsory) basis of the final space after the projection.

	dtype: (optional) data type. Default is np.complex128.

	Proj: (optional) Projector operator. Default is 'None'. If 'Proj = True' is used, the projector isinstance
			calculated and returned as a key. If 'Proj = operator' is put in, the input 'operator' is used to as 
			a projector; it is not returned as a key.

	"""

	variables = ["Proj_Obs"]

	if _np.any(Proj):
		if Proj == True:
			variables.append("Proj")
			Proj = reduced_basis.get_proj(dtype=dtype)

	Proj_Obs = Proj.T.conj()*Obs*Proj

	# define dictionary with outputs
	return_dict = {}
	for i in range(len(variables)):
		return_dict[variables[i]] = vars()[variables[i]]

	return return_dict


def Diag_Ens_Observables(L,V1,E1,V2,betavec=[],alpha=1.0,Obs=False,Ed=False,S_double_quench=False,Sd_Renyi=False,deltaE=False):
	"""
	This is routine calculates the expectation values of physical quantities in the Diagonal ensemble 
	(see eg. arXiv:1509.06411), and returns a dictionary. Equivalently, these are the infinite-time 
	expectation values after a sudden quench at time t=0 from a Hamiltonian H1 to a Hamiltonian H2. 

	L: (compulsory) chain length.

	V1: (compulsory) unitary square matrix. Contains the eigenvectors of H1 in the columns. 
			The initial state is the first column of V1.

	E1: (compulsory) vector of real numbers. Contains the eigenenergies of H1. The order of the 
			eigenvalues must correspond to the order of the columns of V1.

	V2: (compulsory) unitary square matrix. Contains the eigenvectors of H2 in the columns. Must have 
			the same size as V1.

	Obs: (optional) hermitian matrix of the same size as V1. Infinite-time expectation value of the 
			observable Obs in the state V1[:,0]. Has the key 'Obs' in the returned dictionary.

	Ed: (optional) infinite-time expectation value of the Hamiltonian H1 in the state V1[:,0]. 
			Has the key 'Ed' in the returned dictionary.

	deltaE: (optional) infinite-time fluctuations around the energy expectation Ed. 
			Has the key 'deltaE' in the returned dictionary.

	Sd_Renyi: (optional) diagonal Renyi entropy after a quench H1->H2. The default Renyi parameter is 
			'alpha=1.0'. Has the key 'Sd_Renyi' in the returned dictionary.

	alpha: (optional) diagonal Renyi entropy parameter. Default value is 'alpha=1.0'.

	S_double_quench: (optional) diagonal entropy after a double quench H1->H2->H1. 
			Has the key 'S_double_quench' in the returned dictionary.

	betavec: (optional) a list of INVERSE temperatures to specify the distribution of an initial 
			thermal state. When passed the routine returns the corresponding finite-temperature 
			expectation of each specified quantity defined above. The corresponding keys in the r
			eturned dictionary are 'Obs_T', 'Ed_T', 'deltaE_T', 'Sd_Renyi_T', 'S_double_quench_T'. 
	"""

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
	"""
	This routine returns the Kullback-Leibler divergence of the discrete probability distrobutions p1 and p2.
	"""



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
	"""
	Observable_vs_time(psi,V2,E2,Obs,times,return_state=False)

	This routine calculates the expectation value as a function of time of an observable Obs. The initial 
	state is 'psi' and the time evolution is carried out under the Hamiltonian H2. Returns a dictionary 
	in which the time-dependent expectation value has the key 'Expt_time'.

	psi: (compulsory) initial state.

	V2: (compulsory) unitary matrix containing in its columns all eigenstates of the Hamiltonian H2. 

	E2: (compulsory) real vector containing the eigenvalues of the Hamiltonian H2. 
			The order of the eigenvalues must correspond to the order of the columns of V2. 

	Obs: (compulsory) hermitian matrix to calculate its time-dependent expectation value. 

	times: (compulsory) a vector of times to evaluate the expectation value at. 

	return_state: (optional) when set to 'True', returns a matrix whose columns give the state vector 
			at the times specified by the row index. The return dictonary key is 'psi_time'.
	"""

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
	"""
	Mean_Level_Spacing(E)

	This routine returns the mean-level spacing 'r_ave' of the energy distribution E, see arXiv:1212.5611. 

	E: (compulsory) ordered list of ascending, nondegenerate eigenenergies.
	"""

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



