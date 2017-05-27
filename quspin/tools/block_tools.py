from __future__ import print_function, division
# QuSpin modules
from ..operators import hamiltonian as _hamiltonian
from ..operators import ishamiltonian as _ishamiltonian
# numpy modules
import numpy as _np # generic math functions
from numpy import hstack,vstack
# scipy modules
import scipy
import scipy.sparse as _sp
from scipy.sparse.linalg import expm_multiply
# multi-processing modules
from multiprocessing import Process,Queue,Event
from joblib import Parallel,delayed
# six for python 2.* and 3.* dictionary compatibility
from six import iteritems

try:
	from itertools import izip
except ImportError:
	izip = zip

__all__=["block_diag_hamiltonian","block_ops"]

def block_diag_hamiltonian(blocks,static,dynamic,basis_con,basis_args,dtype,basis_kwargs={},get_proj_kwargs={},get_proj=True,check_symm=True,check_herm=True,check_pcon=True):
	"""
	This function constructs a hamiltonian object which is block diagonal with the blocks being created by
	the list 'blocks'

	RETURNS:

	* P: sparse matrix which is the projector to the block space.

	* H: hamiltonian object in block diagonal form. 

	--- arguments ---
	* blocks: (required) list/tuple/iterator which contains the blocks the user would like to put into the hamiltonian as dictionaries or hamiltonian objects.

	* static: (required) the static operator list which is used to construct the block hamiltonians. follows hamiltonian format.

	* dynamic: (required) the dynamic operator list which is used to construct the block hamiltonians. follows hamiltonian format.

	* basis_con: (required) the basis constructor used to construct the basis objects which will create the block diagonal hamiltonians.

	* basis_args: (required) tuple which gets passed as the first argument for basis_con, contains required arguments. 

	* dtype: (required) the data type to construct the hamiltonian with.

	* get_proj: (optional) flag which tells the function to calculate and return the projector to the subpace requested.

	* basis_kwargs: (optional) dictionary of keyword arguments to add when calling basis constructor

	* get_proj_kwargs: (optional) dictionary of keyword arguments to add when calling basis.get_proj() & basis.get_get()

	* check_symm: (optional) flag which tells the function to check the symmetry of the operators for the first hamiltonian constructed.

	* check_herm: (optional) same for check_symm but for hermiticity.

	* check_pcon: (optional) same for check_symm but for particle conservation. 

	"""

	H_list = []
	P_list = []

	blocks = list(blocks)

	if all([isinstance(block,dict) for block in blocks]):
		[blocks[i].update(basis_kwargs) for i in range(len(blocks))]
		dynamic_list = [(tup[-2],tuple(tup[-1])) for tup in dynamic]
		dynamic_list = [([],f,f_args) for f,f_args in set(dynamic_list)]
		static_mats = []
		for block in blocks:
			b = basis_con(*basis_args,**block)
			if get_proj:
				P = b.get_proj(dtype,**get_proj_kwargs)
				P_list.append(P)

			H = _hamiltonian(static,dynamic,basis=b,dtype=dtype,check_symm=check_symm,check_herm=check_herm,check_pcon=check_pcon)
			check_symm = False
			check_herm = False
			check_pcon = False
			static_mats.append(H.static.tocoo())
			for i,(Hd,_,_) in enumerate(H.dynamic):
				dynamic_list[i][0].append(Hd.tocoo())

		static = [_sp.block_diag(static_mats,format="csr")]
		dynamic = []
		for mats,f,f_args in dynamic_list:
			mats = _sp.block_diag(mats,format="csr")
			dynamic.append([mats,f,f_args])

	elif all([_ishamiltonian(block) for block in blocks]):
		static_mats = []	
		dynamic_dict = {}
		for H in blocks:
			if get_proj:
				P = H.basis.get_proj(dtype,**get_proj_kwargs)
				P_list.append(P)

			static_mats.append(H.static.tocoo())
			for i,(Hd,f,f_args) in enumerate(H.dynamic):
				if (f,f_args) not in dynamic_dict:
					dynamic_dict[(f,f_args)] = [Hd.tocoo()]
				else:
					dynamic_dict[(f,f_args)].append(Hd.tocoo())

		static = [_sp.block_diag(static_mats,format="csr")]
		dynamic = []
		for (f,f_args),mats in dynamic_list.items():
			mats = _sp.block_diag(mats,format="csr")
			dynamic.append([mats,f,f_args])

	else:
		raise ValueError("blocks must be list of hamiltonians or list of dictionaries containing symmetry sectors.")



	if get_proj:
		P = _sp.hstack(P_list,format="csr")
		return P,_hamiltonian(static,dynamic,copy=False)
	else:
		return _hamiltonian(static,dynamic,copy=False)








# worker function which loops over one of more generators provided by gen_func and returns the result via queue 'q'.
# waits for signal from 'e' before continuing. 
def worker(gen_func,args_list,q,e):
	try:
		from itertools import izip
	except ImportError:
		izip = zip

	gens = []
	for arg in args_list:
		gens.append(gen_func(*arg))

	generator = izip(*gens)
	for s in generator:
		e.clear()
		q.put(s)
		e.wait()

	q.close()



# generator which spawns processes to run generators then uses a queue for each process to retrieve the results
# which it then yields. 
def generate_parallel(n_process,n_iter,gen_func,args_list):
	n_items = len(args_list)

	# calculate how to distribute generators over processes.
	if n_items <= n_process and n_process > 0:
		n_process = n_items
		n_pp = 1
		n_left = 1
	elif n_items > n_process and n_process > 0:
		n_pp = n_items//n_process
		n_left = n_pp + n_items%n_process		

	# if one process specified just do the generator without sub processes.
	if n_process <= 1:
		try:
			from itertools import izip
		except ImportError:
			izip = zip

		gens = []
		for arg in args_list:
			gens.append(gen_func(*arg))

		generator = izip(*gens)

		for s in generator:
			yield s

		return 
	# split up argument list 
	sub_lists = [args_list[0:n_left]]
	sub_lists.extend([ args_list[n_left + i*n_pp:n_left + (i+1)*n_pp] for i in range(n_process-1)])

	# create lists of queues, events, and processes.
	es = []
	qs = []
	ps = []
	for i in range(n_process):
		e = Event()
		q = Queue(1)
		p = Process(target=worker, args=(gen_func,sub_lists[i],q,e))
		p.daemon = True
		es.append(e)
		qs.append(q)
		ps.append(p)


	# start processes
	for p in ps:
		p.start()

	# for number of iterations
	for i in range(n_iter):
		s = []
		# retrieve results for each sub-process and let the process know to continue calculation.
		for q,e in zip(qs,es):
			s.extend(q.get())
			e.set() # free process to do next calculation

		# yield all results
		yield tuple(s)

	# end processes
	for p in ps:
		p.join()




# generating function for evolution with H.evolve
def _evolve_gen(psi,H,t0,times,H_real,imag_time,solver_name,solver_args):
	for psi in H.evolve(psi,t0,times,H_real=H_real,imag_time=imag_time,solver_name=solver_name,iterate=True,**solver_args):
		yield psi

# generating function for evolution via expm_multiply
def _expm_gen(psi,H,times,dt):
	if times[0] != 0:
		H *= times[0]
		psi = expm_multiply(H,psi)
		H /= times[0]

	yield psi

	H *= dt
	for t in times[1:]:
		psi = expm_multiply(H,psi)
		yield psi
	H /= dt

# using generate_parallel to get block evolution yields state in full H-space
def _block_evolve_iter(psi_blocks,H_list,P,t0,times,H_real,imag_time,solver_name,solver_args,n_jobs):
	args_list = [(psi_blocks[i],H_list[i],t0,times,H_real,imag_time,solver_name,solver_args) for i in range(len(H_list))]

	for psi_blocks in generate_parallel(n_jobs,len(times),_evolve_gen,args_list):
		psi_t = hstack(psi_blocks)
		yield P.dot(psi_t)

# using generate_parallel to get block evolution yields state in full H-space
def _block_expm_iter(psi_blocks,H_list,P,start,stop,num,endpoint,n_jobs):
	times,dt = _np.linspace(start,stop,num=num,endpoint=endpoint,retstep=True)
	args_list = [(psi_blocks[i],H_list[i],times,dt) for i in range(len(H_list))]
	for psi_blocks in generate_parallel(n_jobs,len(times),_expm_gen,args_list):
		psi_t = hstack(psi_blocks)
		yield P.dot(psi_t)	

# helper functions for doing evolution not with iterator
def _block_evolve_helper(H,psi,t0,times,H_real,imag_time,solver_name,solver_args):
	return H.evolve(psi,t0,times,H_real=H_real,imag_time=imag_time,solver_name=solver_name,**solver_args)


class block_ops(object):
	def __init__(self,blocks,static,dynamic,basis_con,basis_args,dtype,basis_kwargs={},get_proj_kwargs={},save_previous_data=True,compute_all_blocks=False,check_symm=True,check_herm=True,check_pcon=True):
		"""
		This class is used to split the dynamics of a state up over various symmetry sectors if the initial state does 
		not obey the symmetry but the hamiltonian does. Moreover we provide a multiprocessing option which allows the 
		user to split the dynamics up over multiple cores.

		---arguments---

		* blocks: (required) list/tuple/iterator which contains the blocks the user would like to put into the hamiltonian as dictionaries.

		* static: (required) the static operator list which is used to construct the block hamiltonians. follows hamiltonian format.

		* dynamic: (required) the dynamic operator list which is used to construct the block hamiltonians. follows hamiltonian format.

		* basis_con: (required) the basis constructor used to construct the basis objects which will create the block diagonal hamiltonians.

		* basis_args: (required) tuple which gets passed as the first argument for basis_con, contains required arguments. 

		* check_symm: (optional) flag to check symmetry 

		* dtype: (required) the data type to construct the hamiltonian with.

		* basis_kwargs: (optional) dictionary of keyword arguments to add when calling basis constructor

		* get_proj_kwargs: (optional) dictionary of keyword arguments to add when calling basis.get_vec()

		* save_previous_data: (optional) when doing the evolution this class has to construct the hamiltonians. this takes
		some time and so by setting this to true, the class will keep previously calculated hamiltonians so that next time
		it needs to do evolution in that block it doesn't have to calculate it again.

		* compute_all_blocks: (optional) flag which tells the class to just compute all hamiltonian blocks at initialization.
		This option also sets save_previous_data to True by default. 

		* check_symm: (optional) flag which tells the function to check the symmetry of the operators for the first hamiltonian constructed.

		* check_herm: (optional) same for check_symm but for hermiticity.

		* check_pcon: (optional) same for check_symm but for particle conservation. 

		--- block_ops attributes ---: '_. ' below stands for 'object. '

		_.dtype: the numpy data type the block hamiltonians are stored with

		_.save_previous_data: flag which tells the user if data is being saved. 

		_.H_dict: dictionary which contains the block hamiltonians under key str(block) wher block is the block dictionary.

		_.P_dict: dictionary which contains the block projectors under the same keys as H_dict.

		_.basis_dict: dictionary which contains the basis objects under the same keys ad H_dict. 

		_.static: list of the static operators used to construct block hamiltonians

		_.dynamic: list of dynamic operators use to construct block hamiltonians

		"""

		self._basis_dict = {}
		self._H_dict = {}
		self._P_dict = {}
		self._dtype=dtype
		self._save = save_previous_data
		self._static = static
		self._dynamic = dynamic
		self._checks = {"check_symm":check_symm,"check_herm":check_herm,"check_pcon":check_pcon}
		self._no_checks = {"check_symm":False,"check_herm":False,"check_pcon":False}
		self._checked = False
		self._get_proj_kwargs = get_proj_kwargs


		for block in blocks:
			block.update(basis_kwargs)
			b = basis_con(*basis_args,**block)
			if b.Ns >  0:
				self._basis_dict[str(block)]=b

		if compute_all_blocks:
			self._save=True
			self.compute_all_blocks()


	@property
	def dtype(self):
		return self._dtype

	@property
	def save_previous_data(self):
		return self._save

	@property
	def H_dict(self):
		return self._H_dict

	@property
	def P_dict(self):
		return self._P_dict

	@property
	def basis_dict(self):
		return self._basis_dict

	@property
	def static(self):
		return list(self._static)

	@property
	def dynamic(self):
		return list(self._dynamic)


	def update_blocks(self,blocks,basis_con,basis_args,compute_all_blocks=False):
		blocks = list(blocks)
		for block in blocks:
			if str(block) not in self._basis_dict.keys():
				b = basis_con(*basis_args,**block)

				if b.Ns >  0:
					self._basis_dict[str(block)]=b	

		if compute_all_blocks:
			self.compute_all_blocks()	


	def compute_all_blocks(self):
		for key,b in iteritems(self._basis_dict):
			if self._P_dict.get(key) is None:
				p = b.get_proj(self.dtype,**self._get_proj_kwargs)
				self._P_dict[key] = p

			if self._H_dict.get(key) is None:
				if not self._checked:
					H = _hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._checks)
					self._checked=True
				else:
					H = _hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._no_checks)
				self._H_dict[key] = H



	def evolve(self,psi_0,t0,times,iterate=False,n_jobs=1,block_diag=False,H_real=False,imag_time=False,solver_name="dop853",**solver_args):
		"""
		this function is the creates blocks and then uses them to run H.evole in parallel.
		
		RETURNS:
			1) iterate = True 
				* returns generator which generates the time dependent state in the full H-space basis.

			2) iterate = False
				* return numpy ndarray which has the time dependent states in the full H-space basis as rows.

		--- arguments ---

		* psi_0: (required) ndarray/list/tuple of state which lives in the full hilbert space of your problem. 
		Does not need to obey and sort of symmetry.

		* t0: (required) the inistial time the dynamics starts at.

		* times: (required) either list or numpy array containing the times you would like to have solution at.
		Must be some kind of iterable object.

		* iterate: (optional) tells the function to return generator or array of states.

		* n_jobs: (optional) number of processes to do dynamics with. NOTE: one of those processes is used to gather results.
		for best results all blocks should be approximately the same size and n_jobs-1 must be a common devisor of the number of
		blocks such that there are roughly equal workload for each process. Otherwise you will also be as slow as your
		slowest process.

		* block_diag: (optional) tells the function whether the blocks should be put into list and looped over or to put the blocks
						in a block diagonal matrix for the cumputation. This is useful if their are a lot of smaller blocks.

		The rest of these are just arguments which are used by H.evolve see Documentation for more detail. 

		"""
		if imag_time:
			raise ValueError("imaginary time not supported for block evolution.")
		P = []
		H_list = []
		psi_blocks = []
		for key,b in iteritems(self._basis_dict):
			if self._P_dict.get(key) is None:
				p = b.get_proj(self.dtype,**self._get_proj_kwargs)
				if self._save:
					self._P_dict[key] = p
			else:
				p = self._P_dict[key]


			if _sp.issparse(psi_0):
				psi = p.H.dot(psi_0).toarray()
			else:
				psi = p.H.dot(psi_0)

			psi = _np.asarray(psi).ravel()
			
			if _np.linalg.norm(psi) > 1000*_np.finfo(self.dtype).eps:
				psi_blocks.append(psi)
				P.append(p.tocoo())

				if self._H_dict.get(key) is None:
					if not self._checked:
						H = _hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._checks)
						self._checked=True
					else:
						H = _hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._no_checks)

					if self._save:
						self._H_dict[key] = H

					H_list.append(H)
				else:
					H_list.append(self._H_dict[key])


		if block_diag and H_list:
			N_H = len(H_list)
			n_pp = N_H//n_jobs
			n_left = n_pp + N_H%n_jobs	

			H_list_prime = []
			psi_blocks_prime = []
			if n_left != 0:
				H_list_prime.append(block_diag_hamiltonian(H_list[:n_left],None,None,None,None,self._dtype,get_proj=False,**self._no_checks))
				psi_list_prime.append(hstack(psi_blocks[:n_left]))

			for i in range(n_jobs-1):
				i1 = n_left + i*n_pp
				i2 = n_left + (i+1)*n_pp
				H_list_prime.append(block_diag_hamiltonian(H_list[i1:i2],None,None,None,None,self._dtype,get_proj=False,**self._no_checks))
				psi_list_prime.append(hstack(psi_blocks[i1:i2]))

			H_list = H_list_prime
			psi_blocks = psi_blocks_prime				


		if len(H_list) > 0:
			P = _sp.hstack(P,format="csr")

			if iterate:
				if _np.isscalar(times):
					raise ValueError("If iterate=True times must be a list/array.")
				return _block_evolve_iter(psi_blocks,H_list,P,t0,times,H_real,imag_time,solver_name,solver_args,n_jobs)
			else:
				psi_t = Parallel(n_jobs = n_jobs)(delayed(_block_evolve_helper)(H,psi,t0,times,H_real,imag_time,solver_name,solver_args) for psi,H in zip(psi_blocks,H_list))
				psi_t = vstack(psi_t)
				psi_t = P.dot(psi_t)
				return psi_t
		else:
			raise RuntimeError("initial state has no projection on to specified blocks.")



	def expm(self,psi_0,H_time_eval=0.0,iterate=False,n_jobs=1,block_diag=False,a=-1j,start=None,stop=None,endpoint=None,num=None,shift=None):
		"""
		this function creates blocks and then uses them to evolve state with expm_multiply in parallel.
		
		RETURNS:
			1) iterate = True 
				* returns generator which generates the time dependent state in the full H-space basis.

			2) iterate = False
				* return numpy ndarray which has the time dependent states in the full H-space basis as rows.

		--- arguments ---

		* psi_0: (required) ndarray/list/tuple of state which lives in the full hilbert space of your problem. 
		Does not need to obey and sort of symmetry.

		* H_time_eval: (optional) time to evaluate the hamiltonians at when doing the exponentiation. 

		* iterate: (optional) tells the function to return generator or array of states.

		* n_jobs: (optional) number of processes to do dynamics with. NOTE: one of those processes is used to gather results.
		for best results all blocks should be approximately the same size and n_jobs-1 must be a common devisor of the number of
		blocks such that there are roughly equal workload for each process. Otherwise you will also be as slow as your
		slowest process.

		* block_diag: (optional) tells the function whether the blocks should be put into list and looped over or to put the blocks
						in a block diagonal matrix for the cumputation.

		The rest of these are just arguments which are used by exp_op see Documentation for more detail. 

		"""

		if iterate:
			if start is None and  stop is None:
				raise ValueError("'iterate' can only be True with time discretization. must specify 'start' and 'stop' points.")

			if num is not None:
				if type(num) is not int:
					raise ValueError("expecting integer for 'num'.")
			else:
				num = 50

			if endpoint is not None:
				if type(endpoint) is not bool:
					raise ValueError("expecting bool for 'endpoint'.")
			else: 
				endpoint = True

		else:
			if start is None and  stop is None:
				if num != None:
					raise ValueError("unexpected argument 'num'.")
				if endpoint != None:
					raise ValueError("unexpected argument 'endpoint'.")
			else:
				if not (_np.isscalar(start)  and _np.isscalar(stop)):
					raise ValueError("expecting scalar values for 'start' and 'stop'")

				if not (_np.isreal(start) and _np.isreal(stop)):
					raise ValueError("expecting real values for 'start' and 'stop'")

				if num is not None:
					if type(num) is not int:
						raise ValueError("expecting integer for 'num'.")
				else:
					num = 50

				if endpoint is not None:
					if type(endpoint) is not bool:
						raise ValueError("expecting bool for 'endpoint'.")
				else: 
					endpoint = True
		
		P = []
		H_list = []
		psi_blocks = []
		for key,b in iteritems(self._basis_dict):
			if self._P_dict.get(key) is None:
				p = b.get_proj(self.dtype,**self._get_proj_kwargs)
				if self._save:
					self._P_dict[key] = p
			else:
				p = self._P_dict[key]


			if _sp.issparse(psi_0):
				psi = p.H.dot(psi_0).toarray()
			else:
				psi = p.H.dot(psi_0)

			psi = psi.ravel()
			if _np.linalg.norm(psi) > 1000*_np.finfo(self.dtype).eps:
				psi_blocks.append(psi)
				P.append(p.tocoo())
				if self._H_dict.get(key) is None:
					if not self._checked:
						H = _hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._checks)
						self._checked=True
					else:
						H = _hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._no_checks)

					if self._save:
						self._H_dict[key] = H
					H = H(H_time_eval)*a

				else:
					H = self._H_dict[key](H_time_eval)*a


				if shift is not None:
					H += a*shift*_sp.identity(b.Ns,dtype=self.dtype)

				H_list.append(H)

		if block_diag and H_list:
			N_H = len(H_list)
			n_pp = N_H//n_jobs
			n_left = n_pp + N_H%n_jobs	

			H_list_prime = []
			psi_blocks_prime = []

			psi_block = hstack(psi_blocks[:n_left])
			H_block = _sp.block_diag(H_list[:n_left],format="csr")

			H_list_prime.append(H_block)
			psi_blocks_prime.append(psi_block)


			for i in range(n_jobs-1):
				i1 = n_left + i*n_pp
				i2 = n_left + (i+1)*n_pp
				psi_block = hstack(psi_blocks[i1:i2])
				H_block = _sp.block_diag(H_list[i1:i2],format="csr")

				H_list_prime.append(H_block)
				psi_blocks_prime.append(psi_block)

			H_list = H_list_prime
			psi_blocks = psi_blocks_prime				


		H_is_complex = _np.iscomplexobj([_np.float32(1.0).astype(H.dtype) for H in H_list])

		if H_list:
			P = _sp.hstack(P,format="csr")
			if iterate:
				return _block_expm_iter(psi_blocks,H_list,P,start,stop,num,endpoint,n_jobs)
			else:
				ver = [int(v) for v in scipy.__version__.split(".")]
				if H_is_complex and (start,stop,num,endpoint) != (None,None,None,None) and ver[1] < 19:
					mats = _block_expm_iter(psi_blocks,H_list,P,start,stop,num,endpoint,n_jobs)
					return _np.array([mat for mat in mats]).T
				else:
					psi_t = Parallel(n_jobs = n_jobs)(delayed(expm_multiply)(H,psi,start=start,stop=stop,num=num,endpoint=endpoint) for psi,H in zip(psi_blocks,H_list))
					psi_t = hstack(psi_t).T
					psi_t = P.dot(psi_t)
					return psi_t
		else:
			raise RuntimeError("initial state has no projection on to specified blocks.")







class block_diag_ensemble(object):
	def __init__(self,blocks,static,dynamic,basis_con,basis_args,dtype,get_proj_kwargs={},save_previous_data=True,compute_all_blocks=False,check_symm=True,check_herm=True,check_pcon=True):
		"""
		This class is used to split the dynamics of a state up over various symmetry sectors if the initial state does 
		not obey the symmetry but the hamiltonian does. Moreover we provide a multiprocessing option which allows the 
		user to split the dynamics up over multiple cores.

		---arguments---

		* blocks: (required) list/tuple/iterator which contains the blocks the user would like to put into the hamiltonian as dictionaries.

		* static: (required) the static operator list which is used to construct the block hamiltonians. follows hamiltonian format.

		* dynamic: (required) the dynamic operator list which is used to construct the block hamiltonians. follows hamiltonian format.

		* basis_con: (required) the basis constructor used to construct the basis objects which will create the block diagonal hamiltonians.

		* basis_args: (required) tuple which gets passed as the first argument for basis_con, contains required arguments. 

		* check_symm: (optional) flag to check symmetry 

		* dtype: (required) the data type to construct the hamiltonian with.

		* save_previous_data: (optional) when doing the evolution this class has to construct the hamiltonians. this takes
		some time and so by setting this to true, the class will keep previously calculated hamiltonians so that next time
		it needs to do evolution in that block it doesn't have to calculate it again.

		* compute_all_blocks: (optional) flag which tells the class to just compute all hamiltonian blocks at initialization.
		This option also sets save_previous_data to True by default. 

		* check_symm: (optional) flag which tells the function to check the symmetry of the operators for the first hamiltonian constructed.

		* check_herm: (optional) same for check_symm but for hermiticity.

		* check_pcon: (optional) same for check_symm but for particle conservation. 

		--- block_ops attributes ---: '_. ' below stands for 'object. '

		_.dtype: the numpy data type the block hamiltonians are stored with

		_.save_previous_data: flag which tells the user if data is being saved. 

		_.H_dict: dictionary which contains the block hamiltonians under key str(block) wher block is the block dictionary.

		_.P_dict: dictionary which contains the block projectors under the same keys as H_dict.

		_.basis_dict: dictionary which contains the basis objects under the same keys ad H_dict. 

		_.static: list of the static operators used to construct block hamiltonians

		_.dynamic: list of dynamic operators use to construct block hamiltonians

		"""

		self._basis_dict = {}
		self._H_dict = {}
		self._P_dict = {}
		self._V_dict = {}
		self._E_dict = {}
		self._dtype=dtype
		self._save = save_previous_data
		self._static = static
		self._dynamic = dynamic
		self._checks = {"check_symm":check_symm,"check_herm":check_herm,"check_pcon":check_pcon}
		self._no_checks = {"check_symm":False,"check_herm":False,"check_pcon":False}
		self._checked = False
		self._get_proj_kwargs = get_proj_kwargs


		blocks = list(blocks)
		for block in blocks:
			b = basis_con(*basis_args,**block)
			if b.Ns >  0:
				self._basis_dict[str(block)]=b

		if compute_all_blocks:
			self._save=True
			self.compute_all_blocks()


	@property
	def dtype(self):
		return self._dtype

	@property
	def save_previous_data(self):
		return self._save

	@property
	def H_dict(self):
		return self._H_dict

	@property
	def P_dict(self):
		return self._P_dict

	@property
	def basis_dict(self):
		return self._basis_dict

	@property
	def static(self):
		return list(self._static)

	@property
	def dynamic(self):
		return list(self._dynamic)


	def update_blocks(self,blocks,basis_con,basis_args,compute_all_blocks=False):
		blocks = list(blocks)
		for block in blocks:
			if str(block) not in self._basis_dict.keys():
				b = basis_con(*basis_args,**block)

				if b.Ns >  0:
					self._basis_dict[str(block)]=b	

		if compute_all_blocks:
			self.compute_all_blocks()	


	def compute_all_blocks(self):
		for key,b in iteritems(self._basis_dict):
			if self._P_dict.get(key) is None:
				p = b.get_proj(self.dtype,**self._get_proj_kwargs)
				self._P_dict[key] = p

			if self._H_dict.get(key) is None:
				if not self._checked:
					H = _hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._checks)
					self._checked=True
				else:
					H = _hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._no_checks)
				self._H_dict[key] = H


	def diag_ensemble(istate,**diag_ensemble_kwargs):
		pass




