# need linear algebra packages
import scipy.sparse.linalg as _sla
import scipy.linalg as _la
import numpy.linalg as _npla
import scipy.sparse as _sp
from scipy.special import binom, hyp2f1
import numpy as _np
from inspect import isgenerator as _isgenerator 

# needed for isinstance only
from ..hamiltonian import ishamiltonian as _ishamiltonian
from ..hamiltonian import hamiltonian as _hamiltonian
from ..basis import spin_basis_1d,photon_basis,isbasis,photon_Hspace_dim

import warnings

### TO DO
# finish examples 2, 3


__all__ = ["ent_entropy", "diag_ensemble", "KL_div", "obs_vs_time", "ED_state_vs_time", "mean_level_spacing","coherent_state","project_operator","commutator","anti_commutator"]


def commutator(H1,H2):
	return H1.dot(H2) - H2.dot(H1)

def anti_commutator(H1,H2):
	return H1.dot(H2) + H2.dot(H1)

def coherent_state(a,n,dtype=_np.float64):
	"""
	This function creates a harmonic oscillator coherent state.

	RETURNS: numpy array with harmonic oscilaltor coherent state.

	--- arguments ---

	a: (required) expectation value of annihilation operator, i.e. sqrt(mean particle number)

	n: (required) cut-off on the number of states kept in the definition of the coherent state

	dtype: (optional) data type. Default is set to numpy.float64
	"""

	s1 = _np.full((n,),-_np.abs(a)**2/2.0,dtype=dtype)
	s2 = _np.arange(n,dtype=_np.float64)
	s3 = _np.array(s2)
	s3[0] = 1
	_np.log(s3,out=s3)
	s3[1:] = 0.5*_np.cumsum(s3[1:])
	state = s1+_np.log(a)*s2-s3
	return _np.exp(state)

def ent_entropy(system_state,basis,chain_subsys=None,densities=True,subsys_ordering=True,alpha=1.0,DM=False,svd_return_vec=[False,False,False]):
	"""
	This function calculates the entanglement entropy of a lattice quantum subsystem based on the Singular
	Value Decomposition (svd).

	RETURNS:	dictionary with keys:

	'Sent': entanglement entropy.

	'DM_chain_subsys': (optional) reduced density matrix of chain subsystem.

	'DM_other_subsys': (optional) reduced density matrix of the complement subsystem.

	'U': (optional) svd U matrix

	'V': (optional) svd V matrix

	'lmbda': (optional) svd singular values

	--- arguments ---

	system_state: (required) the state of the quantum system. Can be a:

				-- pure state [numpy array of shape (Ns,)].

				-- density matrix (DM) [numpy array of shape (Ns,Ns)].

				-- diagonal DM [dictionary {'V_rho': V_rho, 'rho_d': rho_d} containing the diagonal DM
					rho_d [numpy array of shape (Ns,)] and its eigenbasis in the columns of V_rho
					[numpy arary of shape (Ns,Ns)]. The keys are CANNOT be chosen arbitrarily.].

				-- a collection of states [dictionary {'V_states':V_states}] containing the states
					in the columns of V_states [shape (Ns,Nvecs)]

	basis: (required) the basis used to build 'system_state'. Must be an instance of 'photon_basis',
				'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'. 

	chain_subsys: (optional) a list of lattice sites to specify the chain subsystem. Default is

				-- [0,1,...,N/2-1,N/2] for 'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'.

				-- [0,1,...,N-1,N] for 'photon_basis'.

	DM: (optional) String to enable the calculation of the reduced density matrix. Available options are

				-- 'chain_subsys': calculates the reduced DM of the subsystem 'chain_subsys' and
					returns it under the key 'DM_chain_subsys'.

				-- 'other_subsys': calculates the reduced DM of the complement of 'chain_subsys' and
					returns it under the key 'DM_other_subsys'.

				-- 'both': calculates and returns both density matrices as defined above.

				Default is 'False'. 	

	alpha: (optional) Renyi alpha parameter. Default is '1.0'. When alpha is different from unity,
				the entropy keys have attached '_Renyi' to their label.

	densities: (optional) if set to 'True', the entanglement entropy is normalised by the size of the
				subsystem [i.e., by the length of 'chain_subsys']. Detault is 'True'.

	subsys_ordering: (optional) if set to 'True', 'chain_subsys' is being ordered. Default is 'True'.

	svd_return_vec: (optional) list of three booleans to return Singular Value Decomposition (svd) 
				parameters:

				-- [True, . , . ] returns the svd matrix 'U'.

				-- [ . ,True, . ] returns the singular values 'lmbda'.

				-- [ . , . ,True] returns the svd matrix 'V'.

				Any combination of the above is possible. Default is [False,False,False].
	"""

	# initiate variables
	variables = ["Sent"]
	
	if DM=='chain_subsys':
		variables.append("DM_chain_subsys")
		if svd_return_vec[0]:
			variables.append('U')
	elif DM=='other_subsys':
		variables.append("DM_other_subsys")
		if svd_return_vec[2]:
			variables.append('V')
	elif DM=='both':
		variables.append("DM_chain_subsys")
		variables.append("DM_other_subsys")
		if svd_return_vec[0]:
			variables.append('U')
		if svd_return_vec[2]:
			variables.append('V')

	if svd_return_vec[1]:
		variables.append('lmbda')
	
	

	# calculate reshaped system_state
	v, rho_d, N_A = _reshape_as_subsys(system_state,basis,chain_subsys=chain_subsys,subsys_ordering=subsys_ordering)
	del system_state

	if DM == False:
		lmbda = _npla.svd(v, compute_uv=False)
	elif DM == 'chain_subsys':
		U, lmbda, _ = _npla.svd(v, full_matrices=False)
		if rho_d is not None:
			DM_chain_subsys = _np.einsum('n,nij,nj,nkj->ik',rho_d,U,lmbda**2,U.conj() )
		else:
			DM_chain_subsys = _np.einsum('nij,nj,nkj->nik',U,lmbda**2,U.conj() )
	elif DM == 'other_subsys':
		_, lmbda, V = _npla.svd(v, full_matrices=False)
		if rho_d is not None:
			DM_other_subsys = _np.einsum('n,nji,nj,nkj->ki',rho_d,V.conj(),lmbda**2,V )
		else:
			DM_other_subsys = _np.einsum('nji,nj,nkj->nki',V.conj(),lmbda**2,V )
	elif DM == 'both':
		U, lmbda, V = _npla.svd(v, full_matrices=False)
		if rho_d is not None:
			DM_chain_subsys = _np.einsum('n,nij,nj,nkj->ik',rho_d,U,lmbda**2,U.conj() )
			DM_other_subsys = _np.einsum('n,nji,nj,nkj->ki',rho_d,V.conj(),lmbda**2,V )
		else:
			DM_chain_subsys = _np.einsum('nij,nj,nkj->nik',U,lmbda**2,U.conj() )
			DM_other_subsys = _np.einsum('nji,nj,nkj->nki',V.conj(),lmbda**2,V )

	del v
	# add floating point number to zero elements
	lmbda[lmbda<=1E-16] = _np.finfo(lmbda.dtype).eps

	# calculate singular values of reduced DM and the corresponding probabilities
	if rho_d is not None:
		p = rho_d.dot(lmbda**2)
		lmbda = _np.sqrt(p)
	else:# calculate probabilities
		p = (lmbda**2).T

	# calculate entanglement entropy of 'system_state'
	if alpha == 1.0:
		Sent = -_np.sum( p*_np.log(p),axis=0)
	else:
		Sent =  1./(1-alpha)*_np.log(_np.sum(p**alpha, axis=0))


	if densities:
		Sent *= 1.0/N_A


	# store variables to dictionar
	return_dict = {}
	for i in variables:
		return_dict[i] = vars()[i]

	return return_dict


def _reshape_as_subsys(system_state,basis,chain_subsys=None,subsys_ordering=True):
	"""
	This function reshapes an input state (or matrix with 'Nstates' initial states) into an array of
	the shape (Nstates,Ns_subsys,Ns_other) with 'Ns_subsys' and 'Ns_other' the Hilbert space dimensions
	of the subsystem and its complement, respectively.

	RETURNS:	reshaped state, 
				vector with eigenvalues of the DM associated with the initial state, 
				subsystem size

	--- arguments ---

	system_state: (required) the state of the quantum system. Can be a:

				-- pure state [numpy array of shape (1,) or (,1)].

				-- density matrix (DM) [numpy array of shape (1,1)].

				-- diagonal DM [dictionary {'V_rho': V_rho, 'rho_d': rho_d} containing the diagonal DM
					rho_d [numpy array of shape (1,) or (,1)] and its eigenbasis in the columns of V_rho
					[numpy arary of shape (1,1)]. The keys are CANNOT be chosen arbitrarily. 'rho_d'
					can be 'None', but needs to always be passed.

				-- a collection of states [dictionary {'V_states':V_states}] containing the states
					in the columns of V_states [shape (Ns,Nvecs)]

	basis: (required) the basis used to build 'system_state'. Must be an instance of 'photon_basis',
				'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'. 

	chain_subsys: (optional) a list of lattice sites to specify the chain subsystem. Default is

				-- [0,1,...,N/2-1,N/2] for 'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'.

				-- [0,1,...,N-1,N] for 'photon_basis'. 

	subsys_ordering: (optional) if set to 'True', 'chain_subsys' is being ordered. Default is 'True'. 
	"""

	try:
		N = basis.N
	except AttributeError:
		N = basis.chain_N



	if chain_subsys:
		if not isinstance(chain_subsys,list):
			raise TypeError("'subsys' must be a list of integers to label the lattice site numbers of the subsystem!")
		elif min(chain_subsys) < 0:
			raise TypeError("'subsys' must be a list of nonnegative numbers!")
		elif max(chain_subsys) > N-1:
			raise TypeError("'subsys' contains sites exceeding the total lattice site number!")
		elif len(set(chain_subsys)) < len(chain_subsys):
			raise TypeError("'subsys' cannot contain repeating site indices!")
		elif subsys_ordering:
			if len(set(chain_subsys))==len(chain_subsys) and sorted(chain_subsys)!=chain_subsys:
				# if chain subsys is def with unordered sites, order them
				warnings.warn("'subsys' {} contains non-ordered sites. 'subsys' re-ordered! To change default set 'subsys_ordering = False'.".format(chain_subsys),stacklevel=4)
				chain_subsys = sorted(chain_subsys)

	

	if isinstance(system_state,dict):
		keys = set(system_state.keys())
		if keys == set(['V_rho','rho_d']):
			istate = 'DM'
			# define initial state
			rho_d = system_state['rho_d']
			psi = system_state['V_rho']
		elif keys == set(['V_states']):
			istate = 'pure'
			rho_d = None
			psi = system_state['V_states']
		else:
			raise ValueError("expecting dictionary with keys ['V_rho','rho_d'] or ['V_states']")


		if _sp.issparse(system_state):
			warnings.warn("Entanglement_Entropy function only handles numpy.ndarrays, sparse matrix will be comverted to dense matrix.",UserWarning,stacklevel=4)
			system_state = system_state.todense()
			if system_state.shape[1] == 1:
				system_state = system_state.ravel()

		elif system_state.__class__ not in  [_np.ndarray,_np.matrix]:
			system_state = _np.asanyarray(system_state)


		if psi.ndim != 2:
			raise ValueError("Expecting ndim == 2 for V_states.")

		if psi.shape[0] != basis.Ns:
			raise ValueError("V_states shape {0} not compatible with basis size: {1}.".format(psi.shape,basis.Ns))
	else:
		if _sp.issparse(system_state):
			warnings.warn("Entanglement_Entropy function only handles numpy.ndarrays, sparse matrix will be comverted to dense matrix.",UserWarning,stacklevel=4)
			system_state = system_state.todense()
			if system_state.shape[1] == 1:
				system_state = system_state.ravel()
		elif system_state.__class__ not in  [_np.ndarray,_np.matrix]:
			system_state = _np.asanyarray(system_state)

			


		if system_state.ndim == 1: # pure state
			istate = 'pure'
			# define initial state
			psi = system_state
			rho_d = _np.reshape(1.0,(1,))
		elif system_state.ndim == 2: # DM
			if system_state.shape[0] != system_state.shape[1]:
				raise ValueError("Expecting square array for Density Matrix.")
			istate = 'DM'
			# diagonalise DM
			rho_d, psi = _la.eigh(system_state)	

		if psi.shape[0] != basis.Ns:
			raise ValueError("V_states shape {0} not compatible with basis size: {1}.".format(psi.shape,basis.Ns))			
			


	# clear up memory
	del system_state



	# define number of participating states in 'system_state'
	Ns = psi[0,].size



	
	if basis.__class__.__name__[:-9] in ['spin','boson','fermion']:

		# set chain subsys if not defined
		if chain_subsys is None: 
			chain_subsys=[i for i in xrange( int(N/2) )]
			warnings.warn("Subsystem set to contain sites {}.".format(chain_subsys),stacklevel=4)
		
	
		# re-write the state in the initial basis
		if basis.Ns<2**N:
			psi = basis.get_vec(psi,sparse=False)
		
		#calculate H-space dimensions of the subsystem and the system
		N_A = len(chain_subsys)
		Ns_A = 2**N_A

		# define lattice indices putting the subsystem to the left
		system = chain_subsys[:]
		[system.append(i) for i in xrange(N) if not i in chain_subsys]


		'''
		the algorithm for the entanglement entropy of an arbitrary subsystem goes as follows:

		1) the initial state psi has 2^N entries corresponding to the spin-z configs
		2) reshape psi into a 2x2x2x2x...x2 dimensional array (N products in total). Call this array v.
		3) v should satisfy the property that v[0,1,0,0,0,1,...,1,0], total of N entries, should give the entry of psi 
		   along the the spin-z basis vector direction (0,1,0,0,0,1,...,1,0). This ensures a correspondence of the v-indices
		   (and thus the psi-entries) to the N lattice sites.
		4) fix the lattice sites that define the subsystem N_A, and reshuffle the array v according to this: e.g. if the 
	 	   subsystem consistes of sites (k,l) then v should be reshuffled such that v[(k,l), (all other sites)]
	 	5) reshape v[(k,l), (all other sites)] into a 2D array of dimension ( N_A x N/N_A ) and proceed with the SVD as below  
		'''

		if chain_subsys==range(min(chain_subsys), max(chain_subsys)+1): 
			# chain_subsys sites come in consecutive order
			# define reshape tuple
			reshape_tuple2 = (Ns, Ns_A, 2**N/Ns_A)
			# reshape states
			v = _np.reshape(psi.T, reshape_tuple2)
			del psi
		else: # if chain_subsys not consecutive
			# performs 2) and 3)
			# update reshape tuple
			reshape_tuple1 = (Ns,) + tuple([2 for i in xrange(N)])
			# upadte axes dimensions
			system = [s+1 for s in system]
			system.insert(0,0)
			# reshape states
			v = _np.reshape(psi.T,reshape_tuple1)
			del psi
			# performs 4)
			v=v.transpose(system) 
			# performs 5)
			reshape_tuple2 = (Ns, Ns_A, 2**N/Ns_A)
			v = _np.reshape(v,reshape_tuple2)
			

	elif basis.__class__.__name__[:-6] == 'photon':


		'''
		def photon_Hspace_dim(N,Ntot,Nph):

			"""
			This function calculates the dimension of the total spin-photon Hilbert space.
			"""
			if Ntot is None and Nph is not None: # no total particle # conservation
				return 2**N*(Nph+1)
			elif Ntot is not None:
				return 2**N - binom(N,Ntot+1)*hyp2f1(1,1-N+Ntot,2+Ntot,-1)
			else:
				raise TypeError("Either 'Ntot' or 'Nph' must be defined!")
		'''

		# set chain subsys if not defined; 
		if chain_subsys is None: 
			chain_subsys=[i for i in xrange( int(N) )]
			warnings.warn("subsystem automatically set to the entire chain.",stacklevel=4)


		#calculate H-space dimensions of the subsystem and the system
		N_A = len(chain_subsys)
		Ns_A = 2**N_A

		# define lattice indices putting the subsystem to the left
		system = chain_subsys[:]
		[system.append(i) for i in xrange(N) if not i in chain_subsys]
		
		# re-write the state in the initial basis
		if basis.Nph is not None: # no total particle conservation
			Nph = basis.Nph
			if basis.Ns < photon_Hspace_dim(N,basis.Ntot,basis.Nph): #chain symmetries present
				if N_A!=N: # doesn't make use of chain symmetries
					psi = basis.get_vec(psi,sparse=False,full_part=True)
				Ns_spin = basis.chain_Ns
			else:
				Ns_spin = 2**N

		elif basis.Ntot is not None: # total particle-conservation
			Nph = basis.Ntot
			if basis.Ns < photon_Hspace_dim(N,basis.Ntot,basis.Nph): #chain symmetries present
				#psi = _np.asarray( basis.get_vec(psi,sparse=False,full_part=True) )
				if N_A==N:
					psi = basis.get_vec(psi,sparse=False,full_part=False)
					Ns_spin = basis.chain_Ns
				else:
					psi = basis.get_vec(psi,sparse=False,full_part=True)
					Ns_spin = 2**N
			else: # no chain symmetries present
				if N_A==N:
					psi = basis.get_vec(psi,sparse=False,full_part=False)
				else:
					psi = basis.get_vec(psi,sparse=False,full_part=True)
				Ns_spin = basis.chain_Ns

		#del basis
		if sorted(chain_subsys)==range(min(chain_subsys), max(chain_subsys)+1): 
			# chain_subsys sites come in consecutive order
			# define reshape tuple
			if N_A==N: # chain_subsys equals entire lattice
				reshape_tuple2 = (Ns, Ns_spin,Nph+1)
			else: #chain_subsys is smaller than entire lattice
				reshape_tuple2 = (Ns, Ns_A, 2**(N-N_A)*(Nph+1) )
			v = _np.reshape(psi.T,reshape_tuple2)
			del psi
		else: # if chain_subsys not consecutive
			# performs 2) and 3)	
			reshape_tuple1 = (Ns,) + tuple([2 for i in xrange(N)]) + (Nph+1,)
			# upadte axes dimensions
			system = [s+1 for s in system]
			system.insert(0,0)
			# reshape states
			v = _np.reshape(psi.T, reshape_tuple1)
			del psi
			# performs 4)
			system.append(len(system))
			v=v.transpose(system)
			# performs 5)
			reshape_tuple2 = (Ns, Ns_A, 2**(N-N_A)*(Nph+1) )
			v = _np.reshape(v,reshape_tuple2)
				
	else:
		raise ValueError("'basis' class {} not supported!".format(basis.__class__.__name__))

	return v, rho_d, N_A


def _inf_time_obs(rho,istate,Obs=False,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Sent_Renyi=False,alpha=1.0):
	"""
	This function calculates various quantities (observables, fluctuations, entropies) written in the
	diagonal basis of a density matrix 'rho'. See also documentation of 'Diagonal_Ensemble'. The 
	fuction is vectorised, meaning that 'rho' can be an array containing the diagonal density matrices
	in the columns.

	RETURNS:	dictionary with keys corresponding to the observables

	--- variables --- 

	istate: (required) type of initial state. Allowed strings are 'pure', 'DM', 'mixed', 'thermal'.

	Obs: (optional) array of shape (,1) with the diagonal matrix elements of an observable in the basis
			where the density matrix 'rho' is diagonal.

	delta_t_Obs: (optional) array of shape (1,1) containing the off-diagonal matrix elements of the 
			square of an observable, to evaluate the infinite-time temporal fluctuations

	delta_q_Obs: (optional) array containing the diagonal elements (Obs^2)_{nn} - (Obs_{nn})^2 in the 
			basis where the DM 'rho' is diagonal. Evaluates the infinite-time quantum fluctuations.

	Sd_Renyi: (optional) when set to 'True', returns the key with diagonal density matrix of 'rho'.

	Sent_Renyi: (optional) (i,n) array containing the singular values of the i-th state of the eigenbasis
			of 'rho'. Returns the key with the entanglement entropy of 'rho' reduced to a subsystem of
			given choice at infinite times.

	alpha: (optional) Renyi entropy parameter. 
	""" 

	# if Obs or deltaObs: parse V2

	if isinstance(alpha,complex) or alpha < 0.0:
		raise TypeError("Renyi parameter 'alpha' must be real-valued and non-negative!")

	istates = ['pure', 'DM','mixed','thermal']
	if istate not in istates:
		raise TypeError("Uknown type 'istate' encountered! Try {}!".format(istates))

	# initiate observables dict
	variables = []


	if Obs is not False:
		variables.append("Obs_"+istate)
	if delta_t_Obs is not False:
		variables.append("delta_t_Obs_"+istate)
	if delta_q_Obs is not False:
		variables.append("delta_q_Obs_"+istate)
	if Sd_Renyi:
		if alpha == 1.0:
			variables.append("Sd_"+istate)
		else:
			variables.append("Sd_Renyi_"+istate)
	if Sent_Renyi is not False:
		if alpha == 1.0:
			variables.append("Sent_"+istate)
		else:
			variables.append("Sent_Renyi_"+istate)


	#################################################################

	# def einsum string
	def es_str(s):
		'''
		This function uses the np.einsum string to calculate the diagonal of a matrix product (d=1) in d=0.
		'''
		if istate in ['pure','DM']:
			return s.replace(s[-1],'')
		else:
			return s


	# calculate diag ens value of Obs
	if Obs is not False:
		Obs_d = Obs.dot(rho)

	# calculate diag ens value of Obs fluctuations
	if delta_t_Obs is not False:

		delta_t_Obs_d = _np.einsum(es_str('ji,jk,ki->i'),rho,delta_t_Obs,rho).real

		# calculate diag ens value of Obs fluctuations
		if delta_q_Obs is not False:
			delta_q_Obs_d = _np.sqrt( _np.einsum(es_str('ji,j->i'),rho,delta_q_Obs).real - delta_t_Obs_d - Obs_d**2 )

		delta_t_Obs_d = _np.sqrt( delta_t_Obs_d )

		
	# calculate Shannon entropy for the distribution p
	def Entropy(p,alpha):
		""" 
		This function calculates the Renyi entropy of the distribution p with parameter alpha.
		"""
		if alpha == 1.0:
			warnings.warn("Renyi entropy equals von Neumann entropy.", UserWarning,stacklevel=4)
			S = - _np.einsum(es_str('ji,ji->i'),p,_np.log(p))
		else:
			S = 1.0/(1.0-alpha)*_np.log(_np.sum(p**alpha,axis=0) )
			
		return S

	# calculate diag ens ent entropy in post-quench basis
	if Sent_Renyi is not False:
		# calculate effective diagonal singular values, \lambda_i^{(n)} = Sent_Renyi
		rho_ent = (Sent_Renyi**2).dot(rho) # has components (i,psi)
		Sent_Renyi_d = Entropy(rho_ent,alpha)

		
	# calculate diag ens entropy in post-quench basis
	if Sd_Renyi:
		Sd_Renyi_d = Entropy(rho,alpha)
		

	# define return dict
	return_dict = {}
	for i in variables:

		j=i
		if alpha == 1.0 and ("Sent" in i or 'Sd' in i):
			i=i.replace(istate,'Renyi_{}'.format(istate))

		return_dict[j] = locals()[i[:-len(istate)]+'d']
	

	return return_dict
		

def diag_ensemble(N,system_state,V2,densities=True,alpha=1.0,rho_d=False,Obs=False,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Sent_Renyi=False,Sent_args=()):
	"""
	This function calculates the expectation values of physical quantities in the Diagonal ensemble 
	set by the initial state (see eg. arXiv:1509.06411). Equivalently, these are the infinite-time 
	expectation values after a sudden quench at time t=0 from a Hamiltonian H1 to a Hamiltonian H2.


	RETURNS: 	dictionary with keys depending on the passed optional arguments:


	replace "..." below by 'pure', 'thermal' or 'mixed' depending on input params.

	'Obs_...': infinite time expectation of observable 'Obs'.

	'delta_t_Obs_...': infinite time temporal fluctuations of 'Obs'.

	'delta_q_Obs_...': infinite time quantum fluctuations of 'Obs'.

	'Sd_Renyi_...': Renyi entropy of density matrix of Diagonal Ensemble with parameter 'alpha'.

	'Sent_Renyi_...': Renyi entanglement entropy of reduced density matrix of Diagonal Ensemble 
			with parameter 'alpha'.

	'rho_d': density matrix of diagonal ensemble


	--- arguments ---


	N: (required) system size N.

	system_state: (required) the state of the quantum system. Can be a:

				-- pure state [numpy array of shape (1,) or (,1)].

				-- density matrix (DM) [numpy array of shape (1,1)].

				-- mixed DM [dictionary] {'V1':V1,'E1':E1,'f':f,'f_args':f_args,'V1_state':int,'f_norm':False} to 
					define a diagonal DM in the basis 'V1' of the Hamiltonian H1. The keys are

					== 'V1': (required) array with the eigenbasis of H1 in the columns.

					== 'E1': (required) eigenenergies of H1.

					== 'f': (optional) the distribution used to define the mixed DM. Default is
						'f = lambda E,beta: numpy.exp(-beta*(E - E[0]) )'. 

					== 'f_args': (required) list of arguments of function 'f'. If 'f' is not defined, 
						it specifies the inverse temeprature list [beta].

					== 'V1_state' (optional) : list of integers to specify the states of 'V1' wholse pure 
						expectations are also returned.

					== 'f_norm': (optional) if set to 'False' the mixed DM built from 'f' is NOT normalised
						and the norm is returned under the key 'f_norm'. 

					The keys are CANNOT be chosen arbitrarily.

	V2: (required) numpy array containing the basis of the Hamiltonian H2 in the columns.

	rho_d: (optional) When set to 'True', returns the Diagonal ensemble DM under the key 'rho_d'. 

	Obs: (optional) hermitian matrix of the same size as V2, to calculate the Diagonal ensemble 
			expectation value of. Appears under the key 'Obs'.

	delta_t_Obs: (optional) TIME fluctuations around infinite-time expectation of 'Obs'. Requires 'Obs'. 
			Appears under the key 'delta_t_Obs'.

	delta_q_Obs: (optional) QUANTUM fluctuations of the expectation of 'Obs' at infinite-times. 
			Requires 'Obs'. Appears under the key 'delta_q_Obs'. Returns temporal fluctuations 
			'delta_t_Obs' for free.

	Sd_Renyi: (optional) diagonal Renyi entropy in the basis of H2. The default Renyi parameter is 
			'alpha=1.0' (see below). Appears under the key Sd_Renyi'.

	Sent_Renyi: (optional) entanglement Renyi entropy of a subsystem of a choice. The default Renyi 
			parameter is 'alpha=1.0' (see below). Appears under the key Sent_Renyi'. Requires 
			'Sent_args'. To specify the subsystem, see documentation of '_reshape_as_subsys'.

	Sent_args: (optional) tuple of Entanglement_Entropy arguments, required when 'Sent_Renyi = True'.
			At least 'Sent_args=(basis)' is required. If not passed, assumes the default 'chain_subsys', 
			see documentation of '_reshape_as_subsys'.

	densities: (optional) if set to 'True', all observables are normalised by the system size N, except
				for the entanglement entropy which is normalised by the subsystem size 
				[i.e., by the length of 'chain_subsys']. Detault is 'True'.

	alpha: (optional) Renyi alpha parameter. Default is '1.0'.
	"""


	if N and not(type(N) is int):
		raise TypeError("System size 'N' must be a positive integer!")


	# various checks
	if delta_t_Obs or delta_q_Obs:
		if not Obs:
			raise TypeError("Expecting to parse the observable 'Obs' whenever 'delta_t_Obs = True' or 'delta_q_Obs = True'!")
	
	# calculate diagonal ensemble DM

	if isinstance(system_state,(list, tuple, _np.ndarray)): # initial state either pure or DM

		if len(system_state.shape)==1: # pure state
			istate = 'pure'
			# calculate diag ensemble DM
			rho = abs( system_state.conj().dot(V2) )**2;
		elif len(system_state.shape)==2: # DM
			istate = 'DM'
			# calculate diag ensemble DM
			rho = _np.einsum( 'ij,ji->i', V2.T.conj(), system_state.dot(V2) ).real

	
	elif isinstance(system_state,dict): # initial state is defined by diag distr
		# define allowed keys
		key_strings = ['V1','E1','f','f_args','V1_state','f_norm']

		if 'V1' in system_state.keys():
			V1 = system_state['V1']
		else:
			raise TypeError("Dictionary 'system_state' must contain states matrix 'V1'!")
		
		if 'E1' in system_state.keys():
			E1 = system_state['E1']
		else:
			raise TypeError("Dictionary 'system_state' must contain eigenvalues vector 'E1'!")
		
		if 'f_args' in system_state.keys():
			f_args = system_state['f_args']
		else:
			raise TypeError("Dictionary 'system_state' must contain function arguments list 'f_args'!")

		if 'V1_state' in system_state.keys():
			V1_state = system_state['V1_state']

		# check if user has passed the distribution 'f'
		if 'f' in system_state.keys():
			f = system_state['f']
			istate = 'mixed'
		else:
			istate = 'thermal'
			# define Gibbs distribution (up to normalisation)
			f = lambda E1,beta: _np.exp(-beta*(E1 - E1[0]))

		if 'f_norm' in system_state.keys():
			f_norm = system_state['f_norm']
			f_norms = _np.zeros((len(f_args[0])),dtype=type(f_args[0][0]) )
		else:
			f_norm = True


		'''
		# import array to be able to assign V1 from the keys below
		from numpy import array
		# turn dict into variables
		for key,value in system_state.iteritems():
			# check if key is allowed
			if key not in key_strings:
				raise TypeError("Key '{}' not allowed for use in dictionary 'system_state'!".format(key))
			# display full strings
			_np.set_printoptions(threshold='nan')
			# turn key to variable and assign its value
			exec("{} = {}".format(key,repr(value)) ) in locals()
		'''

		if 'V1_state' in locals():
			if not(type(V1_state) is int):
				raise TypeError("Expecting an integer value for variable 'V1_state'!")
			if V1_state < 0 or V1_state > len(E1)-1:
				raise TypeError("Value 'V1_state' violates '0 <= V1_state <= len(E1)-1'!")

		# define diagonal (in V1) mixed DM
		warnings.warn("All expectation values depend statistically on the symmetry block via the available number of states as part of the system-size dependence!",UserWarning,stacklevel=4)
		
		rho_mixed = _np.zeros((len(E1),len(f_args[0])),dtype=type(f_args[0][0]) )
		for i, arg in enumerate(f_args[0]):
			if f_norm:
				rho_mixed[:,i] = f(E1,arg) / sum(f(E1,arg))
			else:
				rho_mixed[:,i] = f(E1,arg)
				# calculate normalisation
				f_norms[i] = sum(f(E1,arg))


		# calculate diag ensemble DM for each state in V1
		rho = abs( V2.conj().T.dot(V1) )**2 # components are (n,psi)

		del V1, E1
	else:
		raise TypeError("Wrong variable type for 'system_state'! E.g., use np.ndarray.")


	# clear up memory
	del system_state

	# add floating point number to zero elements
	rho[rho<=1E-16] = _np.finfo(rho.dtype).eps



	# prepare observables
	if Obs is not False or delta_t_Obs is not False or delta_q_Obs is not False:
		# check if Obs is hermitian
		print "these lines need to be revised; Need also a flag to disable the hermiticity check."
		try: 
			if _la.norm(Obs.todense().T.conj() - Obs.todense()) > 1E4*_np.finfo(Obs.dtype).eps:
				raise ValueError("'Obs' is not hermitian!")
		except AttributeError:
			if _la.norm(Obs.T.conj() - Obs) > 1E4*_np.finfo(Obs.dtype).eps:
				raise ValueError("'Obs' is not hermitian!")

		if (delta_t_Obs or delta_q_Obs) and Obs is not False:
			# diagonal matrix elements of Obs^2 in the basis V2
			print "revisit dot product in deltaObs"
			#delta_t_Obs =  _np.einsum( 'ij,ji->i', V2.T.conj(), Obs.dot(Obs).dot(V2) ).real
			Obs = reduce(_np.dot,[V2.T.conj(),_np.asarray(Obs.todense()),V2])
			delta_t_Obs = _np.square(Obs)
			_np.fill_diagonal(delta_t_Obs,0.0)
			if delta_q_Obs is not False:
				delta_q_Obs = _np.diag(Obs.dot(Obs)).real
			Obs = _np.diag(Obs).real
			#delta_q_Obs = delta_q_Obs # - Obs**2

		elif Obs is not False:
			# diagonal matrix elements of Obs in the basis V2
			Obs = _np.einsum('ij,ji->i', V2.transpose().conj(), Obs.dot(V2) ).real

		
	if Sent_Renyi:
		# calculate singular values of columns of V2
		v, _, N_A = _reshape_as_subsys({'V_rho':V2,'rho_d':None},**Sent_args)
		Sent_Renyi = _npla.svd(v,compute_uv=False).T # components (i,n)
		
	# clear up memory
	del V2

	# calculate diag expectation values
	Expt_Diag = _inf_time_obs(rho,istate,alpha=alpha,Obs=Obs,delta_t_Obs=delta_t_Obs,delta_q_Obs=delta_q_Obs,Sent_Renyi=Sent_Renyi,Sd_Renyi=Sd_Renyi)
	

	# compute densities
	for key,value in Expt_Diag.iteritems():
		if densities:
			if 'ent' in key:
				value *= 1.0/N_A
			else:
				value *= 1.0/N

		Expt_Diag[key] = value
		# calculate thermal expectations
		if istate in ['mixed','thermal']:
			Expt_Diag_state = {}
			Expt_Diag[key] = value.dot(rho_mixed)
			# if 'GS' option is passed save GS value
			if 'V1_state' in locals():
				state_key = key[:-len(istate)]+'{}'.format(V1_state)
				Expt_Diag_state[state_key] = value[V1_state]
			# merge state and mixed dicts
			Expt_Diag.update(Expt_Diag_state)

	if istate in ['mixed','thermal']:
		if f_norm==False:
			Expt_Diag['f_norm'] = f_norms

	# return diag ensemble density matrix if requested
	if rho_d:
		if 'V1_state' in locals():
			Expt_Diag['rho_d'] = rho[:,V1_state]
		else:
			Expt_Diag['rho_d'] = rho


	return Expt_Diag


def project_op(Obs,proj,dtype=_np.complex128):
	"""
	This function takes an observable 'Obs' and a reduced basis or a projector and projects 'Obs'
	onto the reduced basis.

	RETURNS: 	dictionary with keys 

	'Proj_Obs': projected observable 'Obs'. and value the projected observable.

	--- arguments ---

	Obs: (required) operator to be projected.

	proj: (required) basis of the final space after the projection or a matrix which contains the projector.

	dtype: (optional) data type. Default is np.complex128.

	"""

	variables = ["Proj_Obs"]
	if proj.__class__ in [_np.ndarray,_np.matrix] or _sp.issparse(proj):
		if _ishamiltonian(Obs):
			proj_Obs = Obs.project_to(proj)

		else:
			if Obs.ndim != 2:
				raise ValueError("Expecting Obs to be a 2 dimensional array.")

			if Obs.shape[0] != Obs.shape[1]:
				raise ValueError("Expecting OBs to be a square array.")

			if Obs.shape[0] != proj.shape[1]:
				raise ValueError("Dimension mismatch Obs:{0} proj{1}".format(Obs.shape,proj.shape))

			proj_Obs = proj.T.conj().dot(Obs.dot(proj))
	elif isbasis(proj):
		proj = proj.get_proj(dtype)
		if _ishamiltonian(Obs):
			proj_Obs = Obs.project_to(proj)		
		else:
			if Obs.ndim != 2:
				raise ValueError("Expecting Obs to be a 2 dimensional array.")

			if Obs.shape[0] != Obs.shape[1]:
				raise ValueError("Expecting Obs to be a square array.")

			if Obs.shape[0] != proj.shape[1]:
				raise ValueError("Dimension mismatch Obs:{0} proj{1}".format(Obs.shape,proj.shape))
		
			proj_Obs = proj.T.conj().dot(Obs.dot(proj))
	else:
		raise ValueError("Expecting either matrix/array or basis object for proj argument.")

	# define dictionary with outputs
	return_dict = {}
	for i in range(len(variables)):
		return_dict[variables[i]] = locals()[variables[i]]

	return return_dict



def KL_div(p1,p2):
	"""
	This routine returns the Kullback-Leibler divergence of the discrete probability distributions 
	p1 and p2.
	"""
	p1 = _np.asarray(p1)
	p2 = _np.asarray(p2)


	if len(p1) != len(p2):
		raise TypeError("Expecting the probability distributions 'p1' and 'p2' to have same size!")
	if p1.ndim != 1 or p2.ndim != 1:
		raise TypeError("Expecting the probability distributions 'p1' and 'p2' to have linear dimension!")


	if _np.any(p1<=0.0) or _np.any(p2<=0.0):
		raise TypeError("Expecting all entries of the probability distributions 'p1' and 'p2' to be non-negative!")
	if _np.any(p1==0.0):

		inds = _np.where(p1 == 0)

		p1 = _np.delete(p1,inds)
		p2 = _np.delete(p2,inds)

	return _np.multiply( p1, _np.log( _np.divide(p1,p2) ) ).sum()



def ED_state_vs_time(psi,V,E,times,iterate=False):
	"""
	This routine calculates the time evolved initial state as a function of time. The initial 
	state is 'psi' and the time evolution is carried out under the Hamiltonian H with eigenenergies 
	'E' and eigensystem 'V'. 

	RETURNS:	either a matrix with the time evolved states as rows, 
				or an iterator which generates the states one by one.

	--- arguments --- 

	psi: (required) initial state.

	V: (required) unitary matrix containing in its columns all eigenstates of the Hamiltonian H. 

	E: (required) array containing the eigenvalues of the Hamiltonian H2. 
			The order of the eigenvalues must correspond to the order of the columns of V2. 

	times: (required) a vector of times to evaluate the time evolved state at. 

	iterate: (optional) if True this function returns the generator of the time evolved state. 
	"""

	if V.ndim != 2 or V.shape[0] != V.shape[1]:
		raise ValueError("'V' must be a square matrix")

	if V.shape[0] != len(E):
		raise TypeError("Number of eigenstates in 'V' must equal number of eigenvalues in 'E'!")
	if len(psi) != len(E):
		raise TypeError("Variables 'psi' and 'E' must have the same dimension!")

	if _np.isscalar(times):
		TypeError("Variable 'times' must be a array or iter like object!")

	times = _np.asarray(times)
	times = _np.array(-1j*times)


	# define generator of time-evolved state in basis V2
	def psi_t_iter(V,psi,times):
		# a_n: probability amplitudes
		# times: time vector
		a_n = V.T.conj().dot(psi)
		for t in times:
			yield V.dot( _np.exp(-1j*E*t)*a_n )

	if iterate:
		return psi_t_iter(V,psi,times)
	else:
		c_n = V.T.conj().dot(psi)
		Ntime = len(times)
		Ns = len(E)

		psi_t = _np.broadcast_to(times,(Ns,Ntime)).T # generate [[-1j*times[0], ..., -1j*times[0]], ..., [-1j*times[-1], ..., -1j*times[01]]
		psi_t = psi_t*E # [[-1j*E[0]*times[0], ..., -1j*E[-1]*times[0]], ..., [-1j*E[0]*times[-1], ..., -1j*E[-1]*times[-1]]
		_np.exp(psi_t,psi_t) # [[exp(-1j*E[0]*times[0]), ..., exp(-1j*E[-1]*times[0])], ..., [exp(-1j*E[0]*times[-1]), ..., exp(-1j*E[01]*times[01])]


		psi_t *= c_n # [[c_n[0]exp(-1j*E[0]*times[0]), ..., c_n[-1]*exp(-1j*E[-1]*times[0])], ..., [c_n[0]*exp(-1j*E[0]*times[-1]), ...,c_n[o]*exp(-1j*E[01]*times[01])]


		psi_t = psi_t.T 
		# for each vector trasform back to original basis
		psi_t = V.dot(psi_t) 

		return psi_t.T # [ psi(times[0]), ...,psi(times[-1]) ]



def obs_vs_time(psi_t,Obs_list,return_state=False,times=None):
	
	"""
	This routine calculates the expectation value of (a list of) observable(s) as a function of time 
	in the time-dependent state 'psi_t'.

	RETURNS:	dictionary with keys:

	'Expt_time': 2D array, contains the time-dependent expectation (in the rows) of the i'th 
				observable from 'Obs_list' in the i'th column.

	'psi_t': (optional) returns a 2D array the columns of which give the state at the associated time.

	--- arguments ---

	psi_t: (required) three different inputs:
		i) psi_t tuple(psi,E,V,times) 
			psi: initial state
	
			V: unitary matrix containing in its columns all eigenstates of the Hamiltonian H2. 

			E: real vector containing the eigenvalues of the Hamiltonian H2. 
			   The order of the eigenvalues must correspond to the order of the columns of V2.
	
			times: list or array of times to evolve to.

		ii) numpy array or matrix with states in the columns.

		iii) generator which generates the states

	Obs: (required) tuple of hermitian matrices to calculate its time-dependent expectation value. 

	times: (required) a vector of times to evaluate the expectation value at. 

	return_state: (optional) when set to 'True', returns a matrix whose columns give the state vector 
			at the times specified by the column index. The return dictonary key is 'psi_time'.
	"""

	variables = ['Expt_time']
	
	if type(Obs_list) not in [list,tuple]:
		raise ValueError("Obs_list must be tuple or list.")

	num_Obs = len(Obs_list)
	Obs_list = list(Obs_list)
	ham_list = []

	i=0
	while (i < num_Obs):
		if _ishamiltonian(Obs_list[i]):
			Obs = Obs_list.pop(i)
			num_Obs -= 1
			ham_list.append(Obs)
		else:
			if not(_sp.issparse(Obs_list[i])) and not(Obs_list[i].__class__ in [_np.ndarray,_np.matrix]):
				Obs_list[i] = _np.asanyarray(Obs_list[i])
			
			i += 1

	Obs_list = tuple(Obs_list)
	ham_list = tuple(ham_list)


	if type(psi_t) is tuple:

		psi,E,V,times = psi_t

		if V.ndim != 2 or V.shape[0] != V.shape[1]:
			raise ValueError("'V' must be a square matrix")

		if V.shape[0] != len(E):
			raise TypeError("Number of eigenstates in 'V' must equal number of eigenvalues in 'E'!")
		if len(psi) != len(E):
			raise TypeError("Variables 'psi' and 'E' must have the same dimension!")
		for Obs in Obs_list:
			if V.shape != Obs.shape:
				raise TypeError("shapes of 'V1' and 'Obs' must be equal!")
		for ham in ham_list:
			if V.shape != ham.get_shape:
				raise TypeError("shapes of 'V1' and 'Obs' must be equal!")
			

		if _np.isscalar(times):
			TypeError("Variable 'times' must be a array or iter like object!")

		if return_state:
			variables.append("psi_t")

		
		# get iterator over time dependent state (see function above)
		psi_t = ED_state_vs_time(psi,V,E,times,iterate = not(return_state) ).T

	elif psi_t.__class__ in [_np.ndarray,_np.matrix]:

		if psi_t.ndim != 2:
			raise ValueError("states must come in two dimensional array.")
		for Obs in Obs_list:
			if psi_t.shape[0] != Obs.shape[1]:
				raise ValueError("states must be in columns of input matrix.")

		for ham in ham_list:
			if psi_t.shape[0] != ham.get_shape[1]:
				raise ValueError("states must be in columns of input matrix.")

		if return_state:
			variables.append("psi_t")
		else:
			return_state=True # set to True to use einsum but do not return state

	elif _isgenerator(psi_t):
		if return_state:
			variables.append("psi_t")
			psi_t_list = []
			for psi in psi_t:
				psi_t_list.append(psi)

			psi_t = _np.vstack(psi_t_list).T

			for Obs in Obs_list:
				if psi_t.shape[0] != Obs.shape[1]:
					raise ValueError("states must be in columns of input matrix.")

			for ham in ham_list:
				if psi_t.shape[0] != ham.get_shape[1]:
					raise ValueError("states must be in columns of input matrix.")


	else:
		raise ValueError("input not recognized")
	
		

	Expt_time = []

	if return_state:
		for Obs in Obs_list:
			psi_l = Obs.dot(psi_t)
			Expt_time.append(_np.einsum("ji,ji->i",psi_t.conj(),psi_l).real)

		for ham in ham_list:
			if times is not None:
				psi_l = ham.dot(psi_t,time=times,check=False)
			else:
				warnings.warn("argument 'times' needed for time dependent Obs.")
				psi_l = ham.dot(psi_t)
		
			Expt_time.append(_np.einsum("ji,ji->i",psi_t.conj(),psi_l).real)
		Expt_time = _np.vstack(Expt_time).T

	else:

		# loop over psi generator
		for m,psi in enumerate(psi_t):
			if psi.ndim == 2:
				psi = psi.ravel()

			if times is not None:
				time = times[m]
			else:
				time = 0

			Expt = []
			for Obs in Obs_list:
				psi_l = Obs.dot(psi)
				Expt.append(_np.vdot(psi,psi_l).real)

			for ham in ham_list:
				Expt.append(ham.matrix_ele(psi,psi,time=time).real)

			Expt_time.append(_np.asarray(Expt))

		Expt_time = _np.vstack(Expt_time)

	return_dict = {}
	for i in variables:
		return_dict[i] = locals()[i]

	return return_dict



def mean_level_spacing(E):
	"""
	This routine calculates the mean-level spacing 'r_ave' of the energy distribution E, see arXiv:1212.5611.

	RETURNS: float with mean-level spacing 'r_ave'

	--- arguments ---

	E: (required) ordered list of ascending, nondegenerate eigenenergies.
	"""

	if not isinstance(E,_np.ndarray):
		E = _np.asarray(E)

	if _np.any(_np.sort(E)!=E):
		raise TypeError("Expecting a sorted list of ascending, nondegenerate eigenenergies 'E'.")

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


