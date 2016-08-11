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
from ..basis import spin_basis_1d,photon_basis

import warnings


__all__ = ["Entanglement_Entropy", "Diag_Ens_Observables", "Kullback_Leibler_div", "Observable_vs_time", "ED_state_vs_time", "Mean_Level_Spacing"]

# coherent state function
def coherent_state(a,n,dtype=_np.float64):
	"""
	This function creates a harmonic oscillator (ho) coherent state.

	RETURNS: numpy array with ho coherent state.

	--- arguments ---

	a: (compulsory) expectation value of annihilation operator, i.e. sqrt(mean particle number)

	n: (compulsory) cut-off on the number of states kept in the definition of the coherent state

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

def Entanglement_Entropy(system_state,basis,chain_subsys=None,densities=True,subsys_ordering=True,alpha=1.0,DM=False,svd_return_vec=[False,False,False]):
	"""
	This function calculates the entanglement entropy of a lattice quantum subsystem based on the Singular
	Value Decomposition (svd).

	RETURNS:	dictionary in which the entanglement entropy has the key 'Sent'.

	--- arguments ---

	system_state: (compulsory) the state of the quantum system. Can be a:

				-- pure state [numpy array of shape (1,) or (,1)].

				-- density matrix (DM) [numpy array of shape (1,1)].

				-- diagonal DM [dictionary {'V_rho': V_rho, 'rho_d': rho_d} containing the diagonal DM
					rho_d [numpy array of shape (1,) or (,1)] and its eigenbasis in the columns of V_rho
					[numpy arary of shape (1,1)]. The keys are CANNOT be chosen arbitrarily.].

	basis: (compulsory) the basis used to build 'system_state'. Must be an instance of 'photon_basis',
				'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'. 

	chain_subsys: (optional) a list of lattice sites to specify the chain subsystem. Default is

				-- [0,1,...,L/2-1,L/2] for 'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'.

				-- [0,1,...,L-1,L] for 'photon_basis'.

	DM: (optional) String to enable the calculation of the reduced density matrix. Available options are

				-- 'chain_subsys': calculates the reduced DM of the subsystem 'chain_subsys' and
					returns it under the key 'DM_chain_subsys'.

				-- 'other_subsys': calculates the reduced DM of the complement of 'chain_subsys' and
					returns it under the key 'DM_other_subsys'.

				-- 'both': calculates and returns both density matrices as defined above.

				Default is 'False'. 	

	alpha: (optional) Renyi alpha parameter. Default is '1.0'.

	densities: (optional) if set to 'True', the entanglement entropy is normalised by the size of the
				subsystem [i.e., by the length of 'chain_subsys']. Detault is 'True'.

	subsys_ordering: (optional) if set to 'True', 'chain_subsys' is being ordered. Default is 'True'.

	svd_return_vec: (optional) list of three booleans to return Singular Value Decomposition (svd) 
				parameters:

				-- [True, . , . ] returns the svd matrix 'U'.

				-- [ . ,True, . ] returns the (effective) singular values 'lmbda'.

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
	v, rho_d, L_A = reshape_as_subsys(system_state,basis,chain_subsys=chain_subsys,subsys_ordering=subsys_ordering)
	del system_state

	if len(v.shape) != 3:
		v = _np.reshape(v,(1,)+v.shape)
		rho_d = _np.reshape(rho_d,(1,))

	if DM == False:
		lmbda = _npla.svd(v, compute_uv=False)
	elif DM == 'chain_subsys':
		U, lmbda, _ = _npla.svd(v, full_matrices=False)
		DM_chain_subsys = _np.einsum('n,nij,nj,nkj->ik',rho_d,U,lmbda**2,U.conj() )
	elif DM == 'other_subsys':
		_, lmbda, V = _npla.svd(v, full_matrices=False)
		DM_other_subsys = _np.einsum('n,nji,nj,nkj->ki',rho_d,V.conj(),lmbda**2,V )
	elif DM == 'both':
		U, lmbda, V = _npla.svd(v, full_matrices=False)
		DM_chain_subsys = _np.einsum('n,nij,nj,nkj->ik',rho_d,U,lmbda**2,U.conj() )
		DM_other_subsys = _np.einsum('n,nji,nj,nkj->ki',rho_d,V.conj(),lmbda**2,V )


	# add floating point number to zero elements
	lmbda[lmbda<=1E-16] = _np.finfo(lmbda.dtype).eps

	# calculate singular values of reduced DM
	lmbda = _np.sqrt( rho_d.dot(lmbda**2) )

	
	# calculate entanglement entropy of 'system_state'
	if alpha == 1.0:
		Sent = -( lmbda**2).dot( _np.log( lmbda**2  ) ).sum()
	else:
		Sent =  1./(1-alpha)*_np.log( (lmbda**alpha).sum() )


	if densities:
		Sent *= 1.0/L_A


	# store variables to dictionar
	return_dict = {}
	for i in variables:
		return_dict[i] = vars()[i]

	return return_dict



def reshape_as_subsys(system_state,basis,chain_subsys=None,subsys_ordering=True):
	"""
	This function reshapes an input state (or matrix with 'Nstates' initial states) into an array of
	the shape (Nstates,Ns_subsys,Ns_other) with 'Ns_subsys' and 'Ns_other' the Hilbert space dimensions
	of the subsystem and its complement, respectively.

	RETURNS:	reshaped state, vector with eigenvalues of the DM associated with the initial state, 
				subsystem size

	--- arguments ---

	system_state: (compulsory) the state of the quantum system. Can be a:

				-- pure state [numpy array of shape (1,) or (,1)].

				-- density matrix (DM) [numpy array of shape (1,1)].

				-- diagonal DM [dictionary {'V_rho': V_rho, 'rho_d': rho_d} containing the diagonal DM
					rho_d [numpy array of shape (1,) or (,1)] and its eigenbasis in the columns of V_rho
					[numpy arary of shape (1,1)]. The keys are CANNOT be chosen arbitrarily.].

	basis: (compulsory) the basis used to build 'system_state'. Must be an instance of 'photon_basis',
				'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'. 

	chain_subsys: (optional) a list of lattice sites to specify the chain subsystem. Default is

				-- [0,1,...,L/2-1,L/2] for 'spin_basis_1d', 'fermion_basis_1d', 'boson_basis_1d'.

				-- [0,1,...,L-1,L] for 'photon_basis'. 

	subsys_ordering: (optional) if set to 'True', 'chain_subsys' is being ordered. Default is 'True'. 
	"""

	try:
		L = basis.L
	except AttributeError:
		L = basis.chain_L



	if chain_subsys:
		if not isinstance(chain_subsys,list):
			raise TypeError("'subsys' must be a list of integers to label the lattice site numbers of the subsystem!")
		elif min(chain_subsys) < 0:
			raise TypeError("'subsys' must be a list of nonnegative numbers!")
		elif max(chain_subsys) > L-1:
			raise TypeError("'subsys' contains sites exceeding the total lattice site number!")
		elif len(set(chain_subsys)) < len(chain_subsys):
			raise TypeError("'subsys' cannot contain repeating site indices!")
		elif subsys_ordering:
			if len(set(chain_subsys))==len(chain_subsys) and sorted(chain_subsys)!=chain_subsys:
				# if chain subsys is def with unordered sites, order them
				warnings.warn("'subsys' {} contains non-ordered sites. 'subsys' re-ordered! To change default set 'subsys_ordering = False'.".format(chain_subsys),stacklevel=4)
				chain_subsys = sorted(chain_subsys)

	

	# read off initial input
	if isinstance(system_state,(list, tuple, _np.ndarray)): # initial state either pure or DM
		if len(system_state.shape)==1: # pure state
			istate = 'pure'
			# define initial state
			psi = system_state
			rho_d = _np.array(1.0)
		elif len(system_state.shape)==2: # DM
			istate = 'DM'
			# diagonalise DM
			if _sp.issparse(system_state):
				rho_d, psi = _sla.eigsh(system_state)
			else:
				rho_d, psi = _la.eigh(system_state)
	elif isinstance(system_state,dict): # initial DM is diagonal in basis Vrho
		key_strings = ['V_rho','rho_d']
		if 'V_rho' not in system_state.keys():
			raise TypeError("Dictionary 'system_state' must contain eigenstates matrix 'V_rho'!")
		elif 'rho_d' not in system_state.keys():
			raise TypeError("Dictionary 'system_state' must contain diagonal DM 'rho_d'!")
		istate = 'DM'
		# define initial state
		rho_d = system_state['rho_d']
		psi = system_state['V_rho']
	else:
		raise TypeError("Wrong variable type for 'system_state'! E.g., use np.ndarray.")


	# clear up memory
	del system_state



	if basis.__class__.__name__[:-9] in ['spin','boson','fermion']:

		# set chain subsys if not defined
		if chain_subsys is None: 
			chain_subsys=[i for i in xrange( int(L/2) )]
			warnings.warn("Subsystem set to contain sites {}.".format(chain_subsys),stacklevel=4)
		
	
		# re-write the state in the initial basis
		if basis.Ns<2**L:
			psi = basis.get_vec(psi,sparse=False)
		
		#calculate H-space dimensions of the subsystem and the system
		L_A = len(chain_subsys)
		Ns_A = 2**L_A

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

		if chain_subsys==range(min(chain_subsys), max(chain_subsys)+1): 
			# chain_subsys sites come in consecutive order
			# define reshape tuple
			reshape_tuple2 = (Ns_A, 2**L/Ns_A)
			if istate == 'DM':
				reshape_tuple2 = (basis.Ns,) + reshape_tuple2
			# reshape states
			v = _np.reshape(psi.T, reshape_tuple2)
			del psi
		else: # if chain_subsys not consecutive
			# performs 2) and 3)
			reshape_tuple1 = tuple([2 for i in xrange(L)] )
			reshape_tuple2 = (Ns_A, 2**L/Ns_A)
			if istate == 'DM':
				# update reshape tuples
				reshape_tuple1 = (psi.shape[1],) + reshape_tuple1
				reshape_tuple2 = (basis.Ns,) + reshape_tuple2
				# upadte axes dimensions
				system = [s+1 for s in system]
				system.insert(0,0)
			# reshape states
			v = _np.reshape(psi.T, reshape_tuple1)
			del psi
			# performs 4)
			v.transpose(system) 
			# performs 5)
			v = _np.reshape(v,reshape_tuple2)
			

	elif basis.__class__.__name__[:-6] == 'photon':



		def photon_Hspace_dim(L,Ntot,Nph):

			"""
			This function calculates the dimension of the total spin-photon Hilbert space.
			"""
			if Ntot is None and Nph is not None: # no total particle # conservation
				return 2**L*(Nph+1)
			elif Ntot is not None:
				return 2**L - binom(L,Ntot+1)*hyp2f1(1,1-L+Ntot,2+Ntot,-1)
			else:
				raise TypeError("Either 'Ntot' or 'Nph' must be defined!")


		# set chain subsys if not defined; 
		if chain_subsys is None: 
			chain_subsys=[i for i in xrange( int(L) )]
			warnings.warn("subsystem automatically set to the entire chain.",stacklevel=4)


		#calculate H-space dimensions of the subsystem and the system
		L_A = len(chain_subsys)
		Ns_A = 2**L_A

		# define lattice indices putting the subsystem to the left
		system = chain_subsys[:]
		[system.append(i) for i in xrange(L) if not i in chain_subsys]
		
		# re-write the state in the initial basis
		if basis.Nph is not None: # no total particle conservation
			Nph = basis.Nph
			if basis.Ns < photon_Hspace_dim(L,basis.Ntot,basis.Nph): #chain symmetries present
				if L_A!=L: # doesn't make use of chain symmetries
					psi = basis.get_vec(psi,sparse=False,full_part=True)
				Ns_spin = basis.chain_Ns
			else:
				Ns_spin = 2**L

		elif basis.Ntot is not None: # total particle-conservation
			Nph = basis.Ntot
			if basis.Ns < photon_Hspace_dim(L,basis.Ntot,basis.Nph): #chain symmetries present
				#psi = _np.asarray( basis.get_vec(psi,sparse=False,full_part=True) )
				if L_A==L:
					psi = basis.get_vec(psi,sparse=False,full_part=False)
					Ns_spin = basis.chain_Ns
				else:
					psi = basis.get_vec(psi,sparse=False,full_part=True)
					Ns_spin = 2**L
			else: # no chain symmetries present
				if L_A==L:
					psi = basis.get_vec(psi,sparse=False,full_part=False)
				else:
					psi = basis.get_vec(psi,sparse=False,full_part=True)
				Ns_spin = basis.chain_Ns

		#del basis
		if sorted(chain_subsys)==range(min(chain_subsys), max(chain_subsys)+1): 
			# chain_subsys sites come in consecutive order
			# define reshape tuple
			if L_A==L: # chain_subsys equals entire lattice
				reshape_tuple2 = (Ns_spin,Nph+1)
			else: #chain_subsys is smaller than entire lattice
				reshape_tuple2 = ( Ns_A, 2**(L-L_A)*(Nph+1) )
			# check if user parsed a DM
			if istate == 'DM':
				# update reshape tuples
				reshape_tuple2 = (basis.Ns,) + reshape_tuple2
			# reshape states
			v = _np.reshape(psi.T, reshape_tuple2 )
			del psi
		else: # if chain_subsys not consecutive
			# performs 2) and 3)	
			reshape_tuple1 = tuple([2 for i in xrange(L)]) + (Nph+1,)
			reshape_tuple2 = ( Ns_A, 2**(L-L_A)*(Nph+1) )
			if istate == 'DM':
				# update reshape tuples
				reshape_tuple1 = (psi.shape[1],) + reshape_tuple1
				reshape_tuple2 = (basis.Ns,) + reshape_tuple2
				# upadte axes dimensions
				system = [s+1 for s in system]
				system.insert(0,0)
			# reshape states
			v = _np.reshape(psi.T, reshape_tuple1)
			del psi
			# performs 4)
			system.append(len(system))
			v.transpose(system)
			# performs 5)
			v = _np.reshape(v, reshape_tuple2)
				
	else:
		raise ValueError("'basis' class {} not supported!".format(basis.__class__.__name__))

	return v, rho_d, L_A


def inf_time_obs(rho,istate,alpha=1.0,Obs=False,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Sent_Renyi=False):
	"""
	This function calculates various quantities (observables, fluctuations, entropies) written in the
	diagonal basis of a density matrix 'rho'. See also documentation of 'Diag_Ens_Observables'. The 
	fuction is vectorised, meaning that 'rho' can be an array containing the diagonal density matrices
	in the columns.

	RETURNS:	dictionary with keys corresponding to the observables

	--- variables --- 

	istate: (compulsory) type of initial state. Allowed strings are 'pure', 'DM', 'mixed', 'thermal'.

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

	def Fluctuations(delta_Obs,rho):
		return _np.sqrt( _np.einsum(es_str('ji,jk,ki->i'),rho,delta_Obs,rho).real )

	# calculate diag ens value of Obs fluctuations
	if delta_t_Obs is not False:
		delta_t_Obs_d = Fluctuations(delta_t_Obs,rho)

	# calculate diag ens value of Obs fluctuations
	if delta_q_Obs is not False:
		delta_q_Obs_d = Fluctuations(delta_q_Obs,rho)

	# calculate Shannon entropy for the distribution p
	def Entropy(p,alpha):
		""" 
		This function calculates the Renyi entropy of the distribution p with parameter alpha.
		"""
		if alpha == 1.0:
			warnings.warn("Renyi entropy equals von Neumann entropy.", UserWarning,stacklevel=4)
			S = - _np.einsum(es_str('ji,ji->i'),p,_np.log(p))
		else:
			S = 1.0/(1.0-alpha)*_np.log(_np.sum(_np.power(p,alpha),axis=0) )
			
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
		

def Diag_Ens_Observables(L,system_state,V2,densities=True,alpha=1.0,rho_d=False,Obs=False,delta_t_Obs=False,delta_q_Obs=False,Sd_Renyi=False,Sent_Renyi=False,Sent_args=()):
	"""
	This function calculates the expectation values of physical quantities in the Diagonal ensemble 
	set by the initial state (see eg. arXiv:1509.06411). Equivalently, these are the infinite-time 
	expectation values after a sudden quench at time t=0 from a Hamiltonian H1 to a Hamiltonian H2.

	RETURNS: 	dictionary

	--- arguments ---

	L: (compulsory) system size L.

	system_state: (compulsory) the state of the quantum system. Can be a:

				-- pure state [numpy array of shape (1,) or (,1)].

				-- density matrix (DM) [numpy array of shape (1,1)].

				-- mixed DM [dictionary] {'V1':V1,'E1':E1,'f':f,'f_args':f_args,'V1_state':int,'f_norm':False} to 
					define a diagonal DM in the basis 'V1' of the Hamiltonian H1. The keys are

					== 'V1': (compulsory) array with the eigenbasis of H1 in the columns.

					== 'E1': (compulsory) eigenenergies of H1.

					== 'f': (optional) the distribution used to define the mixed DM. Default istate
						'f = lambda E,beta: numpy.exp(-beta*(E - E[0]) )'. 

					== 'f_args': (compulsory) list of arguments of function 'f'. If 'f' is not defined, 
						it specifies the inverse temeprature list [beta].

					== 'V1_state' (optional) : list of integers to specify the states of 'V1' wholse pure 
						expectations are also returned.

					== 'f_norm': (optional) if set to 'False' the mixed DM built from 'f' is NOT normalised
						and the norm is returned under the key 'f_norm'. 

					The keys are CANNOT be chosen arbitrarily.

	V2: (compulsory) numpy array containing the basis of the Hamiltonian H2 in the columns.

	rho_d: (optional) When set to 'True', returns the Diagonal ensemble DM under the key 'rho_d'. 

	Obs: (optional) hermitian matrix of the same size as V2, to calculate the Diagonal ensemble 
			expectation value of. Appears under the key 'Obs'.

	delta_t_Obs: (optional) TIME fluctuations around infinite-time expectation of 'Obs'. Requires 'Obs'. 
			Appears under the key 'delta_t_Obs'.

	delta_q_Obs: (optional) QUANTUM fluctuations of the expectation of 'Obs' at infinite-times. 
			Requires 'Obs'. Appears under the key 'delta_q_Obs'.

	Sd_Renyi: (optional) diagonal Renyi entropy in the basis of H2. The default Renyi parameter is 
			'alpha=1.0' (see below). Appears under the key Sd_Renyi'.

	Sent_Renyi: (optional) entanglement Renyi entropy of a subsystem of a choice. The default Renyi 
			parameter is 'alpha=1.0' (see below). Appears under the key Sent_Renyi'. Requires 
			'Sent_args'. To specify the subsystem, see documentation of 'reshape_as_subsys'.

	Sent_args: (optional) tuple of Entanglement_Entropy arguments, required when 'Sent_Renyi = True'.
			At least 'Sent_args=(basis)' is required. If not passed, assumes the default 'chain_subsys', 
			see documentation of 'reshape_as_subsys'.

	densities: (optional) if set to 'True', all observables are normalised by the system size L, except
				for the entanglement entropy which is normalised by the subsystem size 
				[i.e., by the length of 'chain_subsys']. Detault is 'True'.

	alpha: (optional) Renyi alpha parameter. Default is '1.0'.
	"""


	if L and not(type(L) is int):
		raise TypeError("System size 'L' must be a positive integer!")


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

		if delta_t_Obs and delta_q_Obs and Obs is not False:
			# diagonal matrix elements of Obs^2 in the basis V2
			print "revisit dot product in deltaObs"
			#delta_t_Obs =  _np.einsum( 'ij,ji->i', V2.T.conj(), Obs.dot(Obs).dot(V2) ).real
			Obs = reduce(_np.dot,[V2.T.conj(),_np.asarray(Obs.todense()),V2])
			delta_t_Obs = _np.square(Obs)
			delta_q_Obs = _np.diag(delta_t_Obs)
			_np.fill_diagonal(delta_t_Obs,0.0)
			Obs = _np.diag(Obs).real
			delta_q_Obs -= Obs**2

		elif delta_t_Obs and Obs is not False:
			# diagonal matrix elements of Obs^2 in the basis V2
			print "revisit dot product in deltaObs"
			#delta_t_Obs =  _np.einsum( 'ij,ji->i', V2.T.conj(), Obs.dot(Obs).dot(V2) ).real
			Obs = reduce(_np.dot,[V2.T.conj(),_np.asarray(Obs.todense()),V2])
			delta_t_Obs = _np.square(Obs)
			_np.fill_diagonal(delta_t_Obs,0.0)
			Obs = _np.diag(Obs).real

		elif delta_q_Obs and Obs is not False:
			Obs = reduce(_np.dot,[V2.T.conj(),_np.asarray(Obs.todense()),V2])
			delta_q_Obs = _np.diag(_np.square(Obs)).real
			Obs = _np.diag(Obs).real
			delta_q_Obs -= Obs**2

		elif Obs is not False:
			# diagonal matrix elements of Obs in the basis V2
			Obs = _np.einsum('ij,ji->i', V2.transpose().conj(), Obs.dot(V2) ).real

		
	if Sent_Renyi:
		# calculate singular values of columns of V2
		v, _, L_A = reshape_as_subsys({'V_rho':V2,'rho_d':rho},**Sent_args)
		Sent_Renyi = _npla.svd(v, compute_uv=False).T # components (i,n) 

	# clear up memory
	del V2


	# calculate diag expectation values
	Expt_Diag = inf_time_obs(rho,istate,alpha=alpha,Obs=Obs,delta_t_Obs=delta_t_Obs,delta_q_Obs=delta_t_Obs,Sent_Renyi=Sent_Renyi,Sd_Renyi=Sd_Renyi)
	

	# compute densities
	for key,value in Expt_Diag.iteritems():
		if densities:
			if 'ent' in key:
				value *= 1.0/L_A
			else:
				value *= 1.0/L

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

def Project_Operator(Obs,reduced_basis,dtype=_np.complex128,Proj=False):
	"""
	This function takes an observable 'Obs' and a reduced basis 'reduced_basis' and projects 'Obs'
	onto the reduced basis.

	RETURNS: 	dictionary with keys 'Proj_Obs' and value the projected observable.

	--- arguments ---

	Obs: (compulsory) operator to be projected.

	reduced_basis: (compulsory) basis of the final space after the projection.

	dtype: (optional) data type. Default is np.complex128.

	Proj: (optional) Projector operator. Default is 'None'. If 'Proj = True' is used, the projector is
			calculated and returned under the key 'Proj'. If 'Proj = operator' is put in, the input array
			'operator' is used as the projector but it is not returned.
	"""

	variables = ["Proj_Obs"]

	if _np.any(Proj):
		if Proj == True:
			variables.append("Proj")
			Proj = reduced_basis.get_proj(dtype=dtype)
	else:
		Proj = reduced_basis.get_proj(dtype=dtype)

	Proj_Obs = Proj.T.conj()*Obs*Proj

	# define dictionary with outputs
	return_dict = {}
	for i in range(len(variables)):
		return_dict[variables[i]] = vars()[variables[i]]

	return return_dict



def Kullback_Leibler_div(p1,p2):
	"""
	This routine returns the Kullback-Leibler divergence of the discrete probability distrobutions 
	p1 and p2.
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



def ED_state_vs_time(psi,V,E,times,iterate=False):
	"""
	This routine calculates the time evolved initial state as a function of time. The initial 
	state is 'psi' and the time evolution is carried out under the Hamiltonian H. 

	RETURNS:	either a matrix with the time evolved states as rows, 
				or an iterator which generates the states one by one.

	--- arguments --- 

	psi: (compulsory) initial state.

	V: (compulsory) unitary matrix containing in its columns all eigenstates of the Hamiltonian H. 

	E: (compulsory) array containing the eigenvalues of the Hamiltonian H2. 
			The order of the eigenvalues must correspond to the order of the columns of V2. 

	times: (compulsory) a vector of times to evaluate the time evolved state at. 

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
		psi_t = psi_t * E # [[-1j*E[0]*times[0], ..., -1j*E[-1]*times[0]], ..., [-1j*E[0]*times[-1], ..., -1j*E[-1]*times[-1]]
		_np.exp(psi_t,psi_t) # [[exp(-1j*E[0]*times[0]), ..., exp(-1j*E[-1]*times[0])], ..., [exp(-1j*E[0]*times[-1]), ..., exp(-1j*E[01]*times[01])]


		psi_t *= c_n # [[c_n[0]exp(-1j*E[0]*times[0]), ..., c_n[-1]*exp(-1j*E[-1]*times[0])], ..., [c_n[0]*exp(-1j*E[0]*times[-1]), ...,c_n[o]*exp(-1j*E[01]*times[01])]


		psi_t = psi_t.T 
		# for each vector trasform back to original basis
		psi_t = V.dot(psi_t) 

		return psi_t.T # [ psi(times[0]), ...,psi(times[-1]) ]



def Observable_vs_time(psi_t,Obs_list,return_state=False,times=None):
	
	"""
	This routine calculates the expectation value as a function of time of an observable Obs. The initial 
	state is 'psi' and the time evolution is carried out under the Hamiltonian H. 

	RETURNS:	dictionary in which the time-dependent expectation value has the key 'Expt_time'.

	--- arguments ---

	psi_t: (compulsory) three different inputs:
		i) psi_t tuple(psi,E,V,times) 
			psi: initial state
	
			V: unitary matrix containing in its columns all eigenstates of the Hamiltonian H2. 

			E: real vector containing the eigenvalues of the Hamiltonian H2. 
			   The order of the eigenvalues must correspond to the order of the columns of V2.
	
			times: list or array of times to evolve to.

		ii) ndarray with states in the columns.

		iii) generator which generates the time dependent states

	Obs: (compulsory) hermitian matrix to calculate its time-dependent expectation value. 

	times: (compulsory) a vector of times to evaluate the expectation value at. 

	return_state: (optional) when set to 'True', returns a matrix whose columns give the state vector 
			at the times specified by the row index. The return dictonary key is 'psi_time'.
	"""

	variables = ['Expt_time']

	if type(Obs_list) is not tuple:
		raise ValueError

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

	elif psi_t.__class__ is _np.ndarray:

		if psi_t.ndim != 2:
			raise ValueError("states must come in two dimensional array.")
		for Obs in Obs_list:
			if psi_t.shape[0] != Obs.shape[1]:
				raise ValueError("states must be in columns of input matrix.")

		for ham in ham_list:
			if psi_t.shape[0] != ham.get_shape[1]:
				raise ValueError("states must be in columns of input matrix.")

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
		raise ValueError
	




		
	Expt_time = []

	if return_state:
		warnings.warn("MAINT:hamiltonian classes need to be extended such that it evaluates a time dependent dot with h at different times.",UserWarning)
		if times is not None:
			Expt_time.append(times)

		for Obs in Obs_list:
			psi_l = Obs.dot(psi_t)
			Expt_time.append(_np.einsum("ji,ji->i",psi_t.conj(),psi_l).real)

		for ham in ham_list:
			if times is not None:
				psi_l = ham.dot(psi_t,time=times,check=False)
			else:
				psi_l = ham.dot(psi_t)
			print psi_l.shape,psi_t.shape
		
			Expt_time.append(_np.einsum("ji,ji->i",psi_t.conj(),psi_l).real)
		Expt_time = _np.vstack(Expt_time).T

	else:

		# loop over psi generator
		for m,psi in enumerate(psi_t):
			if psi.ndim == 2:
				psi = psi.ravel()
			if times is not None:
				Expt = [times[m]]
				time = times[m]
			else:
				Expt = []
				time = 0

			for Obs in Obs_list:
				psi_l = Obs.dot(psi)
				Expt.append(_np.vdot(psi,psi_l).real)

			for ham in ham_list:
#				psi_l = ham.dot(psi,time=time,check=False)
#				Expt.append(_np.vdot(psi,psi_l).real)
				Expt.append(ham.matrix_ele(psi,psi,time=time).real)

			Expt_time.append(_np.asarray(Expt))

		Expt_time = _np.vstack(Expt_time)

	return_dict = {}
	for i in variables:
		return_dict[i] = locals()[i]

	return return_dict



def Mean_Level_Spacing(E):
	"""
	This routine calculates the mean-level spacing 'r_ave' of the energy distribution E, see arXiv:1212.5611.

	RETURNS: mean-level spacing 'r_ave'

	--- arguments ---

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



### old functions
'''
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
		if len(psi) < 2**L - binom(L,basis.Ntot+1)*hyp2f1(1,1-L+basis.Ntot,2+basis.Ntot,-1): #chain symemtries present
			if chain_symm:
				raise TypeError("'chain_symm' is incompatible with Ntot symmetry!")
			else:
				psi = _np.asarray( basis.get_vec(psi,sparse=False,full_part=True) )
				Ns_spin = 2**L
		else: # no chain symmetries present
			#print 'real', psi
			psi = _np.asarray( basis.get_vec(psi,sparse=False,full_part=True) )
			Ns_spin = 2**L

	del basis


	if L_A==L:
		# reshape state vector psi
		#print 'real', psi
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

	return v
	
	
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


def Entanglement_entropy2(L,psi,chain_subsys=None,basis=None,alpha=1.0,DM=False):
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
		warnings.warn("subsystem automatically set to contain sites {}.".format(chain_subsys),stacklevel=4)
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


	"""
	the algorithm for the entanglement entropy of an arbitrary subsystem goes as follows:

	1) the initial state psi has 2^L entries corresponding to the spin-z configs
	2) reshape psi into a 2x2x2x2x...x2 dimensional array (L products in total). Call this array v.
	3) v should satisfy the property that v[0,1,0,0,0,1,...,1,0], total of L entries, should give the entry of psi 
	   along the the spin-z basis vector direction (0,1,0,0,0,1,...,1,0). This ensures a correspondence of the v-indices
	   (and thus the psi-entries) to the L lattice sites.
	4) fix the lattice sites that define the subsystem L_A, and reshuffle the array v according to this: e.g. if the 
 	   subsystem consistes of sites (k,l) then v should be reshuffled such that v[(k,l), (all other sites)]
 	5) reshape v[(k,l), (all other sites)] into a 2D array of dimension ( L_A x L/L_A ) and proceed with the SVD as below  
	"""

	#print "real", psi

	# performs 2) and 3)
	v = _np.reshape(psi, tuple([2 for i in xrange(L)] ) )
	del psi
	# performs 4)
	v = _np.transpose(v, axes=system) 
	# performs 5)
	v = _np.reshape(v, ( Ns_A, Ns/Ns_A) )
	
	del system, chain_subsys

	#return v
	
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


def Diag_Ens_Observables_old(L,V1,E1,V2,Obs=False,rho_d=False,Ed=False,S_double_quench=False,Sd_Renyi=False,deltaE=False,state=0,alpha=1.0,betavec=[],E_gs=None,Z=None):
	"""
	This routine calculates the expectation values of physical quantities in the Diagonal ensemble 
	(see eg. arXiv:1509.06411), and returns a dictionary. Equivalently, these are the infinite-time 
	expectation values after a sudden quench at time t=0 from a Hamiltonian H1 to a Hamiltonian H2.
	All quantities are INTENSIVE, i.e. divided by the system size L 
	L: (compulsory) chain length.
	V1: (compulsory) unitary square matrix. Contains the eigenvectors corresponding to the eigenvalues
			of H1 in the columns (must come in the right order). If 'state' is not specified, the initial 
			state is the first column of V1; otherwise the state is V1[:,state].
	E1: (compulsory) vector of ordered real numbers. Contains the eigenenergies of H1. The order of the 
			eigenvalues must correspond to the order of the columns of V1.
	V2: (compulsory) unitary square matrix. Contains the eigenvectors of H2 in the columns. Must have 
			the same size as V1.
	state: (optional) integer, determines which state the non-thermal (e.g. 'GS') quantities should be 
			computed in. The default is 'state = 0', corresponding to the GS (provided 'E1' are ordered).
	rho_d: Diagonal Ensemble for the state V1[:,state]
	Obs: (optional) hermitian matrix of the same size as V1. Infinite-time expectation value of the 
			observable Obs in the state V1[:,state]. Has the key 'Obs' in the returned dictionary.
	Ed: (optional) infinite-time expectation value of the Hamiltonian H1 in the state V1[:,state]. 
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
	E_gs: (optional) ground state energy used for the definition of the thermal density matrix.
			Requires 'betavec'.
	Z: (optional) normalisation constant (partition function) used for definition of thermal density 
			matrix. Requires 'betavec'.
	"""

	if not(type(L) is int):
		raise TypeError("System size 'L' must be a positive integer!")

	if isinstance(alpha,complex) or alpha < 0.0:
		raise TypeError("Renyi entropy parameter 'alpha' must be real-valued and non-negative!")

	if V1.shape != V2.shape:
		raise TypeError("Unitary eigenstate matrices 'V1' and 'V2' must have the same shape!")
	elif len(V1[0,:]) != len(E1):
		raise TypeError("Number of eigenstates in 'V1' must equal number of eigenvalues in 'E1'!")

	if state:
		if not(type(state) is int):
			raise TypeError("'state' must be ingeter to pick the state V[:,state]!")
		if state<0 or state>len(E1)-1:
			raise ValueError("'state' must satisfy: '0 <= state <= len(E1) - 1'!")

	if betavec:
		if E_gs:
			if not Z:
				raise TypeError("Please, parse the thermal desity matrix normalisation variable 'Z'!")
			if E_gs < min(E1):
				raise ValueError("'E_gs' must satisfy: 'E_gs < min(E1)'!")

		if Z:
			if not E_gs:
				raise TypeError("Please, parse the ground state energy variable 'E_gs'!")

	variables_state = []
	variables_T = []

	if Obs is not False:
		if _la.norm(Obs.todense().T.conj() - Obs.todense()) > 1E4*_np.finfo(eval('_np.'+Obs[0,0].dtype.name)).eps:
			raise ValueError("'Obs' is not hermitian!")
		variables_state.append("Obs_state")
		variables_T.append("Obs_T")
	if rho_d:
		variables_state.append("rho_d")
	if Ed:
		warnings.warn("The value of E_Tinf depends on the symmetries used!",UserWarning)
		variables_state.append("Ed_state")
		variables_state.append("E_Tinf")
		variables_T.append("Ed_T")
		variables_T.append("E_Tave")
	if S_double_quench:
		variables_state.append("S_double_quench_state")
		variables_T.append("S_double_quench_T")
	if Sd_Renyi:
		variables_state.append("Sd_Renyi_state")
		variables_T.append("Sd_Renyi_T")
	if S_double_quench or Sd_Renyi:
		variables_state.append("S_Tinf")
	if deltaE:
		variables_state.append("deltaE_state")
		variables_T.append("deltaE_T")

	if not variables_state:
		warnings.warn("No observables were requested: ..exiting", UserWarning)
		return {}

	
	Ns = len(E1) # Hilbert space dimension

	if betavec:
		warnings.warn("All thermal expectation values depend statistically on the symmetry used via the available number of states as part of the system-size dependence!",UserWarning)
		#define thermal density matrix w.r.t. the basis V1	
		rho = _np.zeros((Ns,len(betavec)),dtype=type(betavec[0]) )
		for i in xrange(len(betavec)):
			if E_gs:
				rho[:,i] =_np.exp(-betavec[i]*(E1-E_gs))/Z
			else:
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
	if Ed or S_double_quench or rho_d:
		rho_d = T_nm.dot(T_nm.transpose() )


	# diagonal ens expectation value of Obs in post-quench basis
	if Obs is not False:
		Obs_state = T_nm[state,:].dot(O_mm)/L # GS
		if betavec:
			Obs_T = (_np.einsum( 'ij,j->i', T_nm, O_mm )/L ).dot(rho) # finite-temperature


	#calculate diagonal energy <H1> in long time limit
	if Ed:
		Ed_state = rho_d[state,:].dot(E1)/L  # GS
		if betavec:
			Ed_T  = (rho_d.dot(E1)/L ).dot(rho) # finite-temperature
			E_Tave = E1.dot(rho)/L # average energy density
		E_Tinf = E1.sum()/Ns/L # infinite temperature

	#calculate double-quench entropy (H1->H2->H1)
	if S_double_quench:
		S_double_quench_state = -rho_d[state,:].dot(_np.log(rho_d[state,:]))/L # GS
		if betavec:
			S_double_quench_T  = (_np.einsum( 'ij,ji->i', -rho_d,_np.log(rho_d) )/L ).dot(rho) # finite-temperature
	
	# clear up memory
	if 'rho_d' not in variables_state:
		del rho_d
	else:
		rho_d = T_nm[state,:]

	# calculate diagonal Renyi entropy for parameter alpha: equals (Shannon) entropy for alpha=1: (H1->H2)
	if Sd_Renyi:
		if alpha != 1.0:
			#calculate diagonal (Renyi) entropy for parameter alpha (H1->H2)
			Sd_Renyi_state = 1/(1-alpha)*_np.log(_np.power( T_nm[state,:], alpha ).sum() )/L  # # GS
			if betavec:
				Sd_Renyi_T = 1/(1-alpha)*(_np.log(_np.power( T_nm, alpha ).sum(1)  )/L  ).dot(rho) # finite-temperature
		else:
			warnings.warn("Renyi entropy equals diagonal entropy.", UserWarning)
			Sd_Renyi_state = -T_nm[state,:].dot(_np.log(T_nm[state,:]) ) /L # GS
			if betavec:
				Sd_Renyi_T = (_np.einsum( 'ij,ji->i', -T_nm,_np.log(T_nm.transpose()) )/L ).dot(rho) # finite-temperature

	# infinite temperature entropy
	if S_double_quench or Sd_Renyi:
		S_Tinf = _np.log(2); 

	# calculate long-time energy fluctuations
	if deltaE:
		# calculate <H1^2>
		H1_mn2 = (a_n.conjugate().transpose().dot(_np.einsum('i,ij->ij',E1,a_n)) )**2
		del a_n
		_np.fill_diagonal(H1_mn2,0.0)
		deltaE_state = _np.real( reduce( _np.dot,[T_nm[state,:], H1_mn2, T_nm[state,:] ])  )/L**2  # GS
		if betavec:
			deltaE_T  = _np.real(_np.einsum( 'ij,ji->i', T_nm, H1_mn2.dot(T_nm.transpose()) )/(L**2) ).dot(rho) # finite-temperature
		# free up memory
		del T_nm
		del H1_mn2

	return_dict = {}
	for i in variables_state:
		return_dict[i] = vars()[i]
	if betavec:
		for i in variables_T:
			return_dict[i] = vars()[i]
			

	
	return return_dict

'''


