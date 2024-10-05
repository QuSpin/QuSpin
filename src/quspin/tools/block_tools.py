# -*- coding: utf-8 -*-

from __future__ import print_function, division
# QuSpin modules
# numpy modules
import numpy as _np # generic math functions
# _scipy modules
import scipy as _scipy
import scipy.sparse as _sp
from scipy.sparse.linalg import expm_multiply as _expm_multiply
# multi-processing modules
from multiprocessing import Process as _Process
from multiprocessing import Queue as _Queue
from multiprocessing import Event as _Event

from joblib import Parallel as _Parallel
from joblib import delayed as _delayed
# six for python 2.* and 3.* dictionary compatibility
from six import iteritems as _iteritems
from six import itervalues as _itervalues


__all__=["block_diag_hamiltonian","block_ops"]

def block_diag_hamiltonian(blocks,static,dynamic,basis_con,basis_args,dtype,basis_kwargs={},get_proj_kwargs={},get_proj=True,check_symm=True,check_herm=True,check_pcon=True):
	"""Block-diagonalises a Hamiltonian obeying a symmetry.

	The symmetry blocks are created via the argument 'blocks'.

	Examples
	--------

	The example below demonstrates how to to use the `block_diag_hamiltonian()` function to block-diagonalise
	the single-particle Hamiltonian
	
	.. math::
		H=\\sum_j (J+(-1)^j\\delta J)b^\\dagger_{j+1} b_j + \\mathrm{h.c.} + \\Delta(-1)^j b^\\dagger_j b_j

	with respect to translation symemtry. The Fourier transform is computed along the way.

	.. literalinclude:: ../../doc_examples/block_diag_hamiltonian-example.py
		:linenos:
		:language: python
		:lines: 7-

	Parameters
	-----------
	blocks : list/tuple/iterator
		Contains the symmetry blocks to construct the Hamiltonian with, as dictionaries.
	static : list
		Static operator list used to construct the block Hamiltonians. Follows `hamiltonian` format.
	dynamic : list
		Dynamic operator list used to construct the block Hamiltonians. Follows `hamiltonian` format.
	basis_con : :obj:`basis` 
		Basis constructor used to build the basis objects to create the block diagonal Hamiltonians.
	basis_args : tuple 
		This argument is passed as the first argument for `basis_con`.
		Contains all required arguments for the basis. 
	dtype : 'type'
		The data type (e.g. numpy.float64) to construct the Hamiltonian with.
	get_proj : bool, optional
		Flag which tells the function to calculate and return the projector to the 
		symmetry-block subpace requested. Default is 'True'.
	basis_kwargs : dict, optional
		Dictionary of keyword arguments to add when calling `basis` constructor.
	get_proj_kwargs : dict, optional 
		Dictionary of keyword arguments for `basis.get_proj()` and `basis.project_from()`.
	check_symm : bool, optional 
		Enable/Disable symmetry check of the operators for the first Hamiltonian constructed.
	check_herm : bool, optional
		Enable/Disable hermiticity check of the operators for the first Hamiltonian constructed.
	check_pcon : bool, optional
		Enable/Disable particle conservation check of the operators for the first Hamiltonian constructed.

	Returns
	--------
	tuple
		P : scipy.sparse.csr 
			Projector to the symmetr-block subspace (e.g. Fourier transform in case of momentum blocks).

		H : `obj`
			`hamiltonian` object in block diagonal form.

	Raises
	------
	ValueError
		If `blocks` is not a list of `hamiltonian` objects or a list of dictionaries containing 
		the symmetry sectors.

	"""
	from ..operators import hamiltonian

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

			H = hamiltonian(static,dynamic,basis=b,dtype=dtype,check_symm=check_symm,check_herm=check_herm,check_pcon=check_pcon)
			check_symm = False
			check_herm = False
			check_pcon = False
			static_mats.append(H.static.tocoo())
			for i,Hd in enumerate(_itervalues(H.dynamic)):
				dynamic_list[i][0].append(Hd.tocoo())

		static = [_sp.block_diag(static_mats,format="csr")]
		dynamic = []
		for mats,f,f_args in dynamic_list:
			mats = _sp.block_diag(mats,format="csr")
			dynamic.append([mats,f,f_args])

	else:
		raise ValueError("blocks must be list of dictionaries containing symmetry sectors.")



	if get_proj:
		P = _sp.hstack(P_list,format="csr")
		return P,hamiltonian(static,dynamic,copy=False)
	else:
		return hamiltonian(static,dynamic,copy=False)


def _worker(gen_func,args_list,q,e):
	"""
	Worker function which loops over one of more generators provided by `gen_func` and returns the result 
	via queue `q`. 

	Waits for signal from `e` before continuing. 

	"""

	gens = []
	for arg in args_list:
		gens.append(gen_func(*arg))

	generator = zip(*gens)
	for s in generator:
		e.clear()
		q.put(s)
		e.wait()

	q.close()
 
def _generate_parallel(n_process,n_iter,gen_func,args_list):
	"""
	Generator which spawns processes to run generators, then uses a queue for each process to retrieve 
	the results which it then yields.

	"""
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
		gens = []
		for arg in args_list:
			gens.append(gen_func(*arg))

		generator = zip(*gens)

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
		e = _Event()
		q = _Queue(1)
		p = _Process(target=_worker, args=(gen_func,sub_lists[i],q,e))
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

def _evolve_gen(psi,H,t0,times,stack_state,imag_time,solver_name,solver_args):
	"""Generating function for evolution with `H.evolve`."""
	for psi in H.evolve(psi,t0,times,stack_state=stack_state,imag_time=imag_time,solver_name=solver_name,iterate=True,**solver_args):
		yield psi

def _expm_gen(psi,H,times,dt):
	"""Generating function for evolution via `_expm_multiply`."""
	if times[0] != 0:
		H *= times[0]
		psi = _expm_multiply(H,psi)
		H /= times[0]

	yield psi

	H *= dt
	for t in times[1:]:
		psi = _expm_multiply(H,psi)
		yield psi
	H /= dt


def _block_evolve_iter(psi_blocks,H_list,P,t0,times,stack_state,imag_time,solver_name,solver_args,n_jobs):
	"""using `_generate_parallel` to get block evolution yields state in full H-space."""
	args_list = [(psi_blocks[i],H_list[i],t0,times,stack_state,imag_time,solver_name,solver_args) for i in range(len(H_list))]

	for psi_blocks in _generate_parallel(n_jobs,len(times),_evolve_gen,args_list):
		psi_t = _np.hstack(psi_blocks)
		yield P.dot(psi_t)

def _block_expm_iter(psi_blocks,H_list,P,start,stop,num,endpoint,n_jobs):
	"""using `_generate_parallel` to get block evolution yields state in full H-space."""
	times,dt = _np.linspace(start,stop,num=num,endpoint=endpoint,retstep=True)
	args_list = [(psi_blocks[i],H_list[i],times,dt) for i in range(len(H_list))]
	for psi_blocks in _generate_parallel(n_jobs,len(times),_expm_gen,args_list):
		psi_t = _np.hstack(psi_blocks)
		yield P.dot(psi_t)	

def _block_evolve_helper(H,psi,t0,times,stack_state,imag_time,solver_name,solver_args):
	"""helper functions for doing evolution not with iterator."""
	return H.evolve(psi,t0,times,stack_state=stack_state,imag_time=imag_time,solver_name=solver_name,**solver_args)


class block_ops(object):
	"""Splits up the dynamics of a state over various symmetry sectors.

	Particularly useful if the initial state does NOT obey a symmetry but the hamiltonian does. 
	Moreover, we provide a multiprocessing option which allows the user to split up the dynamics 
	over multiple processing cores.

	Can be used to calculate nonequal time correlators in symmetry-reduced sectors.

	Notes
	-----

	The `block_ops` object is initialised only after calling the function methods of the class to save memory.

	Examples
	--------

	The following sequence of examples uses the Bose-Hubbard model

	.. math::
		H=-J\\sum_j b^\\dagger_{j+1}b_j + \\mathrm{h.c.} + \\frac{U}{2}\\sum_j n_j(n_j-1)

	to show how to use the `block_ops` class to evolve a Fock state, which explicitly breaks
	translational invariance, by decomposing it in all momentum blocks, time-evolving the projections, and putting
	the state back together in the Fock basis in the end. We use the time-evolved state to measure the local density operator :math:`n_j`.

	The code snippets for the time evolution can be found in the examples for the function methods of the class.
	The code snippet below initiates the class, and is required to run the example codes for the function methods.

	.. literalinclude:: ../../doc_examples/block_ops-example.py
		:linenos:
		:language: python
		:lines: 7-55

	"""

	def __init__(self,blocks,static,dynamic,basis_con,basis_args,dtype,basis_kwargs={},get_proj_kwargs={},save_previous_data=True,compute_all_blocks=False,check_symm=True,check_herm=True,check_pcon=True):
		"""Instantiates the `block_ops` class.
		
		Parameters
		-----------
		blocks : list/tuple/iterator
			Contains the symmetry blocks to construct the Hamiltonian with, 
			as dictionaries or `hamiltonian` objects.	
		static : list
			Static operator list used to construct the block Hamiltonians. Follows `hamiltonian` format.
		dynamic : list
			Dynamic operator list used to construct the block Hamiltonians. Follows `hamiltonian` format.
		basis_con : :obj:`basis` 
			Basis constructor used to build the basis objects to create the block diagonal Hamiltonians.
		basis_args : tuple 
			This argument is passed as the first argument for `basis_con`.
			Contains all required arguments for the basis. 
		dtype : 'type'
			The data type (e.g. numpy.float64) to construct the Hamiltonian with.
		basis_kwargs : dict, optional
			Dictionary of keyword arguments to add when calling `basis` constructor.
		get_proj_kwargs : dict, optional 
			Dictionary of keyword arguments for `basis.get_proj()` and `basis.project_from()`.
		save_previous_data : bool, optional
			To do time evolution the `block_ops` class constructs Hamiltonians, which can take time. 
			Set this flag to `True`, and the class will save previously calculated Hamiltonians, so
			next time one needs to do evolution in that block, the code does NOT have to calculate it again.
			Default is `True`.
		compute_all_blocks : bool, optional 
			Flag which tells the `block_ops` class to compute all symmetry blocks at initialization.
			Default is `False`.

			This option sets `save_previous_data = True` automatically. 
		check_symm : bool, optional 
			Enable/Disable symmetry check of the operators for the first Hamiltonian constructed.
		check_herm : bool, optional
			Enable/Disable hermiticity check of the operators for the first Hamiltonian constructed.
		check_pcon : bool, optional
			Enable/Disable particle conservation check of the operators for the first Hamiltonian constructed.

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
		"""type: numpy data type to store the block hamiltonians in."""
		return self._dtype

	@property
	def save_previous_data(self):
		"""bool: reflects state of optimal argument `save_previous_data`."""
		return self._save

	@property
	def H_dict(self):
		"""dict: dictionary which contains the block Hamiltonians under keys labelled by the symmetry blocks,
		e.g. `str(block)` where `block` is a block dictionary variable.
		
		"""
		return self._H_dict

	@property
	def P_dict(self):
		"""dict: dictionary which contains the block projectors under keys labelled by the symmetry blocks,
		e.g. `str(block)` where `block` is a block dictionary variable.
		
		"""
		return self._P_dict

	@property
	def basis_dict(self):
		"""dict: dictionary which contains the `basis` objects under keys labelled by the symmetry blocks,
		e.g. `str(block)` where `block` is a block dictionary variable.
		
		"""
		return self._basis_dict

	@property
	def static(self):
		"""list: contains the static operators used to construct the symmetry-block Hamiltonians."""
		return list(self._static)

	@property
	def dynamic(self):
		"""list: contains the dynamic operators used to construct the symmetry-block Hamiltonians."""
		return list(self._dynamic)


	def update_blocks(self,blocks,basis_con,basis_args,compute_all_blocks=False):
		"""Allows to update the `blocks` variable of the class.

		Parameters
		-----------
		blocks : list/tuple/iterator
			Contains the new symmetry blocks to be added to the `basis_dict` attribute of the class, 
			as dictionaries or `hamiltonian` objects.
		basis_con : :obj:`basis` 
			Basis constructor used to build the basis objects to create the new block diagonal Hamiltonians.	
		basis_args : tuple 
			This argument is passed as the first argument for `basis_con`.
			Contains all required arguments for the basis.
		compute_all_blocks : bool, optional 
			Flag which tells the `block_ops` class to compute all symmetry blocks at initialization.
			Default is `False`.

		"""
		blocks = list(blocks)
		for block in blocks:
			if str(block) not in self._basis_dict.keys():
				b = basis_con(*basis_args,**block)

				if b.Ns >  0:
					self._basis_dict[str(block)]=b	

		if compute_all_blocks:
			self.compute_all_blocks()	


	def compute_all_blocks(self):
		"""Sets `compute_all_blocks = True`.

		Examples
		--------

		The example below builds on the code snippet shown in the description of the `block_ops` class.

		.. literalinclude:: ../../doc_examples/block_ops-example.py
			:linenos:
			:language: python
			:lines: 57-58

		"""
		from ..operators import hamiltonian

		for key,b in _iteritems(self._basis_dict):
			if self._P_dict.get(key) is None:
				p = b.get_proj(self.dtype,**self._get_proj_kwargs)
				self._P_dict[key] = p

			if self._H_dict.get(key) is None:
				if not self._checked:
					H = hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._checks)
					self._checked=True
				else:
					H = hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._no_checks)
				self._H_dict[key] = H


	def _get_P(self,key):
		if self._P_dict.get(key) is None:
			p = self._basis_dict[key].get_proj(self.dtype,**self._get_proj_kwargs)
			if self._save:
				self._P_dict[key] = p

			return p
		else:
			return self._P_dict[key]

	def _get_H(self,key):
		from ..operators import hamiltonian

		if self._H_dict.get(key) is None:
			if not self._checked:
				H = hamiltonian(self._static,self._dynamic,basis=self._basis_dict[key],dtype=self.dtype,**self._checks)
				self._checked=True
			else:
				H = hamiltonian(self._static,self._dynamic,basis=self._basis_dict[key],dtype=self.dtype,**self._no_checks)

			if self._save:
				self._H_dict[key] = H

			return H
		else:
			return self._H_dict[key]


	def evolve(self,psi_0,t0,times,iterate=False,n_jobs=1,block_diag=False,stack_state=False,imag_time=False,solver_name="dop853",**solver_args):
		"""Creates symmetry blocks of the Hamiltonian and then uses them to run `hamiltonian.evolve()` in parallel.
		
		**Arguments NOT described below can be found in the documentation for the `hamiltonian.evolve()` method.**

		Examples
		--------

		The example below builds on the code snippet shown in the description of the `block_ops` class.

		.. literalinclude:: ../../doc_examples/block_ops-example.py
			:linenos:
			:language: python
			:lines: 69-

		Parameters
		-----------
		psi_0 : numpy.ndarray, list, tuple
			Quantum state which defined on the full Hilbert space of the problem. 
			Does not need to obey and sort of symmetry.
		t0 : float
			Inistial time to start the evolution at.
		times : numpy.ndarray, list
			Contains the times to compute the solution at. Must be some an iterable object.
		iterate : bool, optional
			Flag to return generator when set to `True`. Otherwise the output is an array of states. 
			Default is 'False'.
		n_jobs : int, optional 
			Number of processes requested for the computation time evolution dynamics. 

			NOTE: one of those processes is used to gather results. For best performance, all blocks 
			should be approximately the same size and `n_jobs-1` must be a common devisor of the number of
			blocks, such that there is roughly an equal workload for each process. Otherwise the computation 
			will be as slow as the slowest process.
		block_diag : bool, optional 
			When set to `True`, this flag puts the Hamiltonian matrices for the separate symemtry blocks
			into a list and then loops over it to do time evolution. When set to `False`, it puts all
			blocks in a single giant sparse block diagonal matrix. Default is `False`.

			This flag is useful if there are a lot of smaller-sized blocks.

		Returns
		--------
		obj
			if `iterate = True`, returns generator which generates the time dependent state in the 
			full H-space basis.

			if `iterate = False`, returns `numpy.ndarray` which has the time-dependent states in the 
			full H-space basis in the rows.
		
		Raises
		------
		ValueError
			Variable `imag_time=True` option on `hamiltonian.evolve()` method not supported.
		ValueError
			`iterate=True` requires `times` to be an array or a list.
		RuntimeError
			Terminates when initial state has no projection onto the specified symmetry blocks.

		"""


		if imag_time:
			raise ValueError("imaginary time not supported for block evolution.")
		P = []
		H_list = []
		psi_blocks = []
		for key,b in _iteritems(self._basis_dict):
			p = self._get_P(key)

			if _sp.issparse(psi_0):
				psi = p.T.conj().dot(psi_0).toarray()
			else:
				psi = p.T.conj().dot(psi_0)

			psi = _np.asarray(psi).ravel()
			
			if _np.linalg.norm(psi) > 1000*_np.finfo(self.dtype).eps:
				psi_blocks.append(psi)
				P.append(p.tocoo())
				H_list.append(self._get_H(key))

		if block_diag and H_list:
			N_H = len(H_list)
			n_pp = N_H//n_jobs
			n_left = n_pp + N_H%n_jobs	

			H_list_prime = []
			psi_list_prime = []
			if n_left != 0:
				H_list_prime.append(block_diag_hamiltonian(H_list[:n_left],None,None,None,None,self._dtype,get_proj=False,**self._no_checks))
				psi_list_prime.append(_np.hstack(psi_blocks[:n_left]))

			for i in range(n_jobs-1):
				i1 = n_left + i*n_pp
				i2 = n_left + (i+1)*n_pp
				H_list_prime.append(block_diag_hamiltonian(H_list[i1:i2],None,None,None,None,self._dtype,get_proj=False,**self._no_checks))
				psi_list_prime.append(_np.hstack(psi_blocks[i1:i2]))

			H_list = H_list_prime
			psi_blocks = psi_list_prime				


		if len(H_list) > 0:
			P = _sp.hstack(P,format="csr")

			if iterate:
				if _np.isscalar(times):
					raise ValueError("If iterate=True times must be a list/array.")
				return _block_evolve_iter(psi_blocks,H_list,P,t0,times,stack_state,imag_time,solver_name,solver_args,n_jobs)
			else:
				psi_t = _Parallel(n_jobs = n_jobs)(_delayed(_block_evolve_helper)(H,psi,t0,times,stack_state,imag_time,solver_name,solver_args) for psi,H in zip(psi_blocks,H_list))
				psi_t = _np.vstack(psi_t)
				psi_t = P.dot(psi_t)
				return psi_t
		else:
			raise RuntimeError("initial state has no projection on to specified blocks.")


	def expm(self,psi_0,H_time_eval=0.0,iterate=False,n_jobs=1,block_diag=False,a=-1j,start=None,stop=None,endpoint=None,num=None,shift=None):
		"""Creates symmetry blocks of the Hamiltonian and then uses them to run `_expm_multiply()` in parallel.
		
		**Arguments NOT described below can be found in the documentation for the `exp_op` class.**

		Examples
		--------

		The example below builds on the code snippet shown in the description of the `block_ops` class.

		.. literalinclude:: ../../doc_examples/block_ops-example.py
			:linenos:
			:language: python
			:lines: 60-67

		Parameters
		-----------
		psi_0 : numpy.ndarray, list, tuple
			Quantum state which defined on the full Hilbert space of the problem. 
			Does not need to obey and sort of symmetry.
		t0 : float
			Inistial time to start the evolution at.
		H_time_eval : numpy.ndarray, list
			Times to evaluate the Hamiltonians at when doing the matrix exponentiation. 
		iterate : bool, optional
			Flag to return generator when set to `True`. Otherwise the output is an array of states. 
			Default is 'False'.
		n_jobs : int, optional 
			Number of processes requested for the computation time evolution dynamics. 

			NOTE: one of those processes is used to gather results. For best performance, all blocks 
			should be approximately the same size and `n_jobs-1` must be a common devisor of the number of
			blocks, such that there is roughly an equal workload for each process. Otherwise the computation 
			will be as slow as the slowest process.
		block_diag : bool, optional 
			When set to `True`, this flag puts the Hamiltonian matrices for the separate symemtri blocks
			into a list and then loops over it to do time evolution. When set to `False`, it puts all
			blocks in a single giant sparse block diagonal matrix. Default is `False`.

			This flag is useful if there are a lot of smaller-sized blocks.

		Returns
		--------
		obj
			if `iterate = True`, returns generator which generates the time dependent state in the 
			full H-space basis.

			if `iterate = False`, returns `numpy.ndarray` which has the time-dependent states in the 
			full H-space basis in the rows.

		Raises
		------
		ValueError
			Various `ValueError`s of `exp_op` class.
		RuntimeError
			Terminates when initial state has no projection onto the specified symmetry blocks.

		"""
		from ..operators import hamiltonian

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
				if num is not None:
					raise ValueError("unexpected argument 'num'.")
				if endpoint is not None:
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
		for key,b in _iteritems(self._basis_dict):
			p = self._get_P(key)

			if _sp.issparse(psi_0):
				psi = p.T.conj().dot(psi_0).toarray()
			else:
				psi = p.T.conj().dot(psi_0)

			psi = psi.ravel()
			if _np.linalg.norm(psi) > 1000*_np.finfo(self.dtype).eps:
				psi_blocks.append(psi)
				P.append(p.tocoo())
				H = self._get_H(key)
				H = H(H_time_eval)*a
				if shift is not None:
					H += a*shift*_sp.identity(b.Ns,dtype=self.dtype)

				H_list.append(H)

		if block_diag and H_list:
			N_H = len(H_list)
			n_pp = N_H//n_jobs
			n_left = n_pp + N_H%n_jobs	

			H_list_prime = []
			psi_blocks_prime = []

			psi_block = _np.hstack(psi_blocks[:n_left])
			H_block = _sp.block_diag(H_list[:n_left],format="csr")

			H_list_prime.append(H_block)
			psi_blocks_prime.append(psi_block)


			for i in range(n_jobs-1):
				i1 = n_left + i*n_pp
				i2 = n_left + (i+1)*n_pp
				psi_block = _np.hstack(psi_blocks[i1:i2])
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
				ver = [int(v) for v in _scipy.__version__.split(".")]
				if H_is_complex and (start,stop,num,endpoint) != (None,None,None,None) and ver[1] < 19:
					mats = _block_expm_iter(psi_blocks,H_list,P,start,stop,num,endpoint,n_jobs)
					return _np.array([mat for mat in mats]).T
				else:
					psi_t = _Parallel(n_jobs = n_jobs)(_delayed(_expm_multiply)(H,psi,start=start,stop=stop,num=num,endpoint=endpoint) for psi,H in zip(psi_blocks,H_list))
					psi_t = _np.hstack(psi_t).T
					psi_t = P.dot(psi_t)
					return psi_t
		else:
			raise RuntimeError("initial state has no projection on to specified blocks.")





'''
# TO DO

=======

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
		for key,b in _iteritems(self._basis_dict):
			if self._P_dict.get(key) is None:
				p = b.get_proj(self.dtype,**self._get_proj_kwargs)
				self._P_dict[key] = p

			if self._H_dict.get(key) is None:
				if not self._checked:
					H = hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._checks)
					self._checked=True
				else:
					H = hamiltonian(self._static,self._dynamic,basis=b,dtype=self.dtype,**self._no_checks)
				self._H_dict[key] = H


	def diag_ensemble(istate,**diag_ensemble_kwargs):
		pass

=======
'''
