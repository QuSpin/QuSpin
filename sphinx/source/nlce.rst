.. _nlce-label:


A tutorial on QuSpin's `nlce` module
====================================

Numerical linked-cluster expansions (NLCE) are a class a method which combines perturbative linked cluster expansion with exact diagonalization to perform calculations of lattice models directly in the thermodynamic limit. As such we have built a module into QuSpin that can generate the clusters used in these expansions. For an introduction to NLCE methods we suggest you read `A Short Introduction to Numerical Linked-Cluster Expansions <https://arxiv.org/abs/1207.3366>`_.


**TUTORIAL CONTENTS:**

* `Basics of NLCE`_
* `NLCE Tools in QuSpin`_
	* `


Basics of NLCE
++++++++++++++

The Inclusion Principle
````````````````````````

The basic idea behind linked cluster expansions is that quantities that are extensive :math:`P` in system size can be expressed as sums over linked clusters that are subgraphs of the full system. In the thermodynamic limit this series consists every possible linked cluster :math:`c` that is a subgraph of the infinite lattice:

.. math::
	\frac{P}{N}=\sum_c L(c)W_P(c)

Here, :math:`L(c)` is the number of times the cluster `c` can be embedded in the infinite lattice up to translations and :math:`W_P(c)` is the weight of quantity `P` of cluster `c` in the LCE. Here we assume that one can calculate an extensive property :math:`P(c)` using the physical system defined by the Hamiltonian :math:`H(c)` on the linked cluster. This can be the Free energy of the system, the entropy, the energy, correlation functions, anything that is an extensive in the system size. The weight, :math:`W_P(c)`, for property :math:`P` of a given cluster :math:`c` is not :math:`P(c)`, but follows, what is called, the inclusion principle:

.. math::
	W_P(c) = P(c) - \sum_{s\in c} W_P(s)

In words, this equation states that the weight of property :math:`P` for a given cluster contains only contributions of correlations between sites that are not contained in the subgraphs of that cluster. This equation also shows us the linked cluster expansion applied to the finite cluster :math:`c`.  In practice this series has to be truncated to a finite order. In some cases this can cause what appears to be a divergent series, however this is only indicating that the dominate weight in the series is shifting to clusters that are larger than what is accounted for in the truncated expansion. 

 Another thing to note is that for clusters that are topologically equivilant to one another, some quantities :math:`P(c)` have the same value, and hence, have the same weight in the series. In this case is it sufficient to only sum over clusters which are topologically unqiue and account for the the topological equivilance in the multiplicity :math:`L(c)`. As of version 0.4.0 the NLCE tools in QuSpin only calculates the series for topologically distinct clusters. 

In general it is arbitrary what building block are used to build the linked clusters, so long as they can be enbedded in the lattice. The simpliest building block would be a single site. Regardless of the building block used, he number of linked clusters that are topologically distinct grows exponentially for larger cluster sizes. As such, it can be advantagous to use LCEs that are built out of plaquets instead of individual sites as the number of sites per plaquet make it so that the total number of clusters in the expansion is smaller compared to the site expansion but the series will contain clusters that are much larger in terms of the number of individual sites. 

NLCE Tools in QuSpin
++++++++++++++++++++

General Work-flow
`````````````````

QuSpin's NLCE tools provides a simple work-flow for doing NLCE when combined with existing ED methods availible in QuSpin or any other python based methods one might be interested in using. The general work-flow consists of first constucting a class that stores all the topological clusters for the LCE which the user can pickle and reuse for many calculations. These classes provide the user a simple iterface to access cluster information like the graph and the position of the sites on the infinite lattice. Using this interface one can calculate :math:`P(c)` for all clusters in the expansion. The interface allows the user to access a single cluster at a time as well as a iterator functionality allowing one to loop over a series of clusters in a pythonic manner. One can combine this with `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ to distribute each calculation over many cores. Once all :math:`P(c)` are calculated the same object should then be used to calculate the partial sums of he Series. There is even built in methods to perform series extrapolations, like the Wynn epsilon method, with no extra effort. 


Defining the lattice and building blocks of LCE
````````````````````````````````````````````````

Now that we understand the general work flow we can discuss how to specify the lattice used in the LCE. First, its worth noting that expansions generated by the NLCE tools is the expansion is for the infinite lattice but because the series is truncated, the clusters have a bounded size. Therefore, it is sufficient to determine the clusters used in the expansion on a finite periodic lattice we will call this the Embedding Lattice (EL). Every vertex on the EL coreesponds to a building block of a cluster, be it a single site or a plaquet that contains multiple sites. In the plaquet expansion it is possible for plaquets to share sites. In that case one has to adjust the multiplicity :math:`L(c)`. In most cases this should be taken care of by QuSpin but there is also a way of overriding this manually. 

The EL is finite, but is large enough such that all the clusters used in the LCE will not be large enough to wrap around and connect on the other side. Note that QuSpin will not be able to check this, therefore it is on the user to make sure that the EL is large enough. Also note that the larger the EL the longer it takes to construct the cluster expansion. The symmetries of the underlying lattice sructure are defined by how the vertices of the EL transform under the lattice symmetries, e.g. for the site expansion the transofmrations are the actual point-group and translational symmetries of the lattice, while for more complex building blocks the point-group and translational symmetries may be a subgroup of the full lattice translations (see examples below for more details). 

Another ingredient required to build the LCE is how to define the verticies between sites within a cluster. Each cluster is defined as occuptation of verticies on the EL so one has to define the verticies between the sites on the EL. In the site expansion the verticies on the EL are also sites so all one has to provide is a list of sites that a given site connects to in the EL, e.g. a list of nearest neighbors. For a plaquet expansion one has to define the intra-plaquet verticies between sites within a given plaquet as well as inter-plaquet verticies between sites of neighboring plaquets. 


Example 1: Nearest Neighbor Square Lattice Site Expansion
`````````````````````````````````````````````````````````

In this first example, we will show how to use :code:`NCLE_site` to calculate the cluters in the Site based NLCE on the 2D square lattice. First we need to define an embedding Lattice. Here we are interested in the site expansion on the square lattice so our embedding lattice is going to be a finite square lattice. The length of the EL will have to be large enough such that no clusters will wrap around the boundarys, In the case of the site expansion the cluster with the longest linear size will be a chain. Given a maximum cluster size :code:`Ncl_max` we choose the Linear size of the Embedding Lattice to be :code:`L=Ncl_max+(Ncl_max%2)+2`. We add :code:`+2` to the length so that the chain will not be able to link when it wraps around and The extra factor of :code:`Ncl_max%2` is only neccesary if you would like to preserve the staggared patter on the cluster because in some cases the cluster can wrap around to the other side of the lattice and that will mess up the staggared pattern: (-1)^(x+y) on the cluster. Next we need to define the point-group and translational symmetries as well as the list of neighboring sites on the EL. Similar to the mappings in the for the :code:`*_basis_general` classes one must define the transformations by a array such that the mapping :math:`S:i\mapsto j` is stored as :code:`S[i]=j`. To get these transformations we first must have a labeling of each vertex of the EL. We use the standard way of mapping :code:`i_s=x_s+L*y_s` or :code:`x_s,y_s = i_s%L,i_s//L`. After defining the mappings the translations and point group symmetries they are packages together into a single array one for the translations and one for the point group symmetries. Finally the list of nearest neighbors list is created. This is formatted as an array whos i-th row corresponds to the sites connected to the i-th site on the graph. Any negative integer can be used as padding in the array if the number of neighbors in not the same for all sites. The Full code is listed below:

.. code-block:: python

	from quspin.basis import NLCE_site
	import numpy as np
	#
	# Maximum Cluster size
	Ncl_max = 6 
	#
	# size of Embedding Lattice
	L = Ncl_max + (Ncl_max%2) + 2
	N_EL = L**2
	#
	# coordinates on Embedding Lattice
	s = np.arange(N_EL)
	x = s%L
	y = s//L
	#
	# translation generators
	Tx = ((x+1)%L)+y*L
	Ty = x+((y+1)%L)*L
	#
	# point group generators, 4-fold rotation + reflection 
	R = np.rot90(s.reshape((L,L))).ravel()
	Pd = y + L * x
	#
	# EL lattice symmetries split up into two groups
	Pg = np.vstack((Pd,R)).astype(np.int32) # Point-Group
	Tr = np.vstack((Tx,Ty)).astype(np.int32) # Translations
	#
	# defining nearest neighbors
	nn1 = ((x+1)%L)+y*L # right
	nn2 = ((x-1)%L)+y*L # left
	nn3 = x+((y+1)%L)*L # up
	nn4 = x+((y-1)%L)*L # down
	nn_list = np.vstack((nn1,nn2,nn3,nn4)).T.astype(np.int32,order="C")
	#
	# creating cluster expansion object
	nlce = NLCE_site(Ncl_max,N_EL,nn_list,Tr,Pg)

Here is an illistration of the clusters that the LCE generates:

.. image:: images/sq_lat_nn.png
   :height: 100


Example 2: Constructing NLCE with multiple coupling constants
`````````````````````````````````````````````````````````````

In this Next example we will discuss the notion of weighted edges in NLCE and how to use weight edges to take into account two or more coupling constants in the Hmailtonian. In many cases one may have interactions that may not be the same value for every interaction on the lattice, e.g. the :math:`J_1-J_2` model. In this case for the site-based NLCE clusters may have different weights in the expansion even though they may have the same topology based on simply the connections in the graph. In order to take this into account we can use weighted graphs to distinquish the topologies of each graph when the couplinngs are not uniform. To illistrate this we will modify the first example to calculate the site expansion of the :math:`J_1-J_2` model on the 2D square lattice. Because the building blocks are the same, the EL is identical to the previous example. 

.. code-block:: python

	from quspin.basis import NLCE_site
	import numpy as np
	#
	# Maximum Cluster size
	Ncl_max = 6 
	#
	# size of Embedding Lattice
	L = Ncl_max + (Ncl_max%2) + 2
	N_EL = L**2
	#
	# coordinates on Embedding Lattice
	s = np.arange(N_EL)
	x = s%L
	y = s//L
	#
	# translation generators
	Tx = ((x+1)%L)+y*L
	Ty = x+((y+1)%L)*L
	#
	# point group generators, 4-fold rotation + reflection 
	R = np.rot90(s.reshape((L,L))).ravel()
	Pd = y + L * x
	#
	# EL lattice symmetries split up into two groups
	Pg = np.vstack((Pd,R)).astype(np.int32) # Point-Group
	Tr = np.vstack((Tx,Ty)).astype(np.int32) # Translations
	#
	# defining nearest neighbors
	nn1 = ((x+1)%L)+y*L # right
	nn2 = ((x-1)%L)+y*L # left
	nn3 = x+((y+1)%L)*L # up
	nn4 = x+((y-1)%L)*L # down
	# defining next nearest neighbors
	nn5 = ((x+1)%L)+((y+1)%L)*L # right
	nn6 = ((x+1)%L)+((y-1)%L)*L # right
	nn7 = ((x-1)%L)+((y+1)%L)*L # right
	nn8 = ((x-1)%L)+((y-1)%L)*L # right
	nn_list = np.vstack((nn1,nn2,nn3,nn4,nn5,nn6,nn7,nn8)).T.astype(np.int32,order="C")
	#
	# defining weights
	nn_weights = np.array((N*[[1,1,1,1,2,2,2,2]]),dtype=np.int32)
	#
	# creating cluster expansion object
	nlce = NLCE_site(Ncl_max,N_EL,nn_list,Tr,Pg)

Here is an illistration of the clusters that the LCE generates:

.. image:: images/sq_lat_nnn.png
   :height: 100


Example 3: Constructing NLCE with plaquet expansion
````````````````````````````````````````````````````

.. image:: images/plaquet_lattice_embed.png
   :height: 800
   :align: center

In this example we will show how to define more complicated building blocks for the cluster expansion. Here we choose plaquets of four sites on the square lattice. the embedding here is more complicated due to the cluster components being made up of multiple sites. In the picture above we show an illistration of one possible LE for the plaquet based LCE. Here the green squares represent the embedding lattice, e.g. the locations of each plaquet. The red dots represent the sites of the physical lattice. Each red dot is shared between two different plaquets. The blue arrows correspond to the different directions we can define for the translation transformation of the plaquets and the point-group symmetries are defined through the reflections about the purple lines. We also use these to define the symmetry transformations of the sites which we will use to generate the transformations of the plaquets under the different symmetries.  

Each site of the EL is assigned a number like in the site-based LCE. The difference in the plaquet expansion is that each site of the EL is now associated with multiple physical sites with each plaquet having graph edges both within the plaquet as well as inbetween the plaquets. This extra information is encoded two different data structures: 

plaquet_sites
-------------
This is simply an 2D array where the i-th row contain integers that correspond to the sites that belong to the i-th plaquet. In that list the order is not important.

plaquet_edges
-------------
The second data structure is the plaquet edge dictionary. This list has a two-fold usage: firstly this list defines the edges between the physical sites belonging to each plaquet, secondly the dictionary provides the list of neighboring plaquets and the edges between the neighboring plaquet sites and the current plaquet sites. Both of these are important in order to construct the clusters as well as determine the unqiue topology of each cluster and compare it to other clusters.  

Just like in the site expansion we need to specify how the plaquets transform under the different symmetry operations using the same format. It is difficult to write the the symmetry transformation directly on the EL, so in order to make this process easier we use the transformations of the physical lattice. To accomplish this we define each plaquet by a single corner site, then by by acting on that site with the given symmetry transformation we find how the plaquets transform. You can also use four corners of the plaquet as we do in the example below but the procedure is very similar but in order to be consistent we sort the plaquet sites in acending order. In the example below we calculate all of the quantities required for calculating the plaquet expansion on the square lattice with nearest neighbor interactions. 


.. code-block:: python

	from quspin.basis import NLCE_plaquet
	import numpy as np


	def site_to_plaq_map(maps,plaquet_sites,plaquet_index):
		# aux function to define transformation for plaquets
		plaq_maps = []
		for tr in maps:
			# transform columns of plaquet_sites based on tr
			k = tr[plaquet_sites]

			# using plaquet_index find the
			# plaquet that is transformed by tr.
			ind = []
			for b in k[:]:
				key = tuple(sorted(list(b)))
				ind.append(plaquet_index[key])

			plaq_maps.append(np.array(ind))

		return plaq_maps

	# maximumze number of physical sites for LE
	L = 2*(Ncl-1)+2
	# calculating x and y positions for each physical sites
	s = np.arange(L*L)
	x = s%L
	y = s//L
	# calculating the different transformations of the physics sites
	xp_s = ((x+1)%L)+((y+1)%L)*L
	yp_s = ((x-1)%L)+((y+1)%L)*L
	px_s = x[::-1]+y*L
	pr_s = np.rot90(s.reshape((L,L))).ravel()
	# defining the number of plaquets for the LE
	N = L*(L//2)
	# defining the plaquet index
	i = np.arange(N)
	# defining lists used to construct the plaquet site list
	bx = i%(L//2) # plaquet x-position
	by = i//(L//2) # plaquet y-position
	# defining x and y corners of the plaquet
	x1 = 2*bx+by%2 
	x2 = (x1+1)%L 
	y1 = by
	y2 = (by+1)%L
	# defining the sites for each corner of the plaquet
	i1 = x1 + L*y1
	i2 = x2 + L*y1
	i3 = x2 + L*y2
	i4 = x1 + L*y2
	# plaquet schematic
	#
	# i4-i3
	# |   |
	# i1-i2
	# definint plaquet site list
	plaquet_sites = np.vstack((i1,i2,i3,i4)).T.astype(np.int32,copy=True,order="C")
	# construct a dictionary which converts a set of sites to a plaquet index
	plaquet_index = {}
	for j,b in enumerate(plaquet_sites[:]):
		key = tuple(sorted(list(b)))
		plaquet_index[key] = j
	# convert to congiuous array
	plaquet_sites = np.ascontiguousarray(plaquet_sites)
	# define transformation for plaquets
	pgs = site_to_plaq_map([pr_s,px_s],plaquet_sites,plaquet_index)
	trs = site_to_plaq_map([xp_s,yp_s],plaquet_sites,plaquet_index)
	# defining plaquet edge dicionary
	plaquet_edges = {}
	for a in range(N):
		# edge dictionary for each plaquet
		edge_dict = {}
		# get plaquet sites
		i1,i2,i3,i4 = plaquet_sites[a]
		# get index of all plaquets that share a site with the current plaquet
		m = (plaquet_sites==i1) | (plaquet_sites==i2) | (plaquet_sites==i3) | (plaquet_sites==i4)
		x,*_ = np.where(m)
		b_list = x[x!=a]
		# intra-plaquet bonds
		edge_dict[a] = set([(i1,i2),(i2,i3),(i3,i4),(i4,i1)])
		# inter-plaquet bonds
		for b in b_list:
			# set to empty list because there are no edges connecting sites between plaquets.
			edge_dict[b] = set([])
		# generate plaquet element
		plaquet_edges[a] = edge_dict
	# construct plaquet expansion
	nlce = NLCE_plaquet(Ncl,plaquet_sites,plaquet_edges,trs,pgs)

Here is an illistration of the clusters that the LCE generates:

.. image:: images/sq_lat_plaq_nn.png
   :height: 150


Example 3: Constructing NLCE with weighted plaquet expansion
````````````````````````````````````````````````````````````

Just like in Example 2, when there are more than one kind of coupling between the physical sites we must use a weighted graph in order to destinquish topologies of cluster. The mechanism is identical here, however to specify the weights we introduce a new data structure that is parallel to the `plaquet_edges` dictionary. Also note that because we have next nearest neighbor interactions there are inter-plaquet couplings that we include in `plaquet_edges`. In the example below we show how to construct the plaquet LCE with next-nearest neighbor interactions using `edge_weights` dictionary:

.. code-block:: python

	from quspin.basis import NLCE_plaquet
	import numpy as np


	def site_to_plaq_map(maps,plaquet_sites,plaquet_index):
		# aux function to define transformation for plaquets
		plaq_maps = []
		for tr in maps:
			# transform columns of plaquet_sites based on tr
			k = tr[plaquet_sites]

			# using plaquet_index find the
			# plaquet that is transformed by tr.
			ind = []
			for b in k[:]:
				key = tuple(sorted(list(b)))
				ind.append(plaquet_index[key])

			plaq_maps.append(np.array(ind))

		return plaq_maps

	# maximumze number of physical sites for LE
	L = 2*(Ncl-1)+2
	# calculating x and y positions for each physical sites
	s = np.arange(L*L)
	x = s%L
	y = s//L
	# calculating the different transformations of the physics sites
	xp_s = ((x+1)%L)+((y+1)%L)*L
	yp_s = ((x-1)%L)+((y+1)%L)*L
	px_s = x[::-1]+y*L
	pr_s = np.rot90(s.reshape((L,L))).ravel()
	# defining the number of plaquets for the LE
	N = L*(L//2)
	# defining the plaquet index
	i = np.arange(N)
	# defining lists used to construct the plaquet site list
	bx = i%(L//2) # plaquet x-position
	by = i//(L//2) # plaquet y-position
	# defining x and y corners of the plaquet
	x1 = 2*bx+by%2 
	x2 = (x1+1)%L 
	y1 = by
	y2 = (by+1)%L
	# defining the sites for each corner of the plaquet
	i1 = x1 + L*y1
	i2 = x2 + L*y1
	i3 = x2 + L*y2
	i4 = x1 + L*y2
	# plaquet schematic
	#
	# i4-i3
	# |   |
	# i1-i2
	# definint plaquet site list
	plaquet_sites = np.vstack((i1,i2,i3,i4)).T.astype(np.int32,copy=True,order="C")
	# construct a dictionary which converts a set of sites to a plaquet index
	plaquet_index = {}
	for j,b in enumerate(plaquet_sites[:]):
		key = tuple(sorted(list(b)))
		plaquet_index[key] = j
	# convert to congiuous array
	plaquet_sites = np.ascontiguousarray(plaquet_sites)
	# define transformation for plaquets
	pgs = site_to_plaq_map([pr_s,px_s],plaquet_sites,plaquet_index)
	trs = site_to_plaq_map([xp_s,yp_s],plaquet_sites,plaquet_index)
	# defining plaquet edge dicionary and edge weights
	plaquet_edges = {}
	edge_weights = {}
	for a in range(N):
		edge_dict = {}
		i1,i2,i3,i4 = plaquet_sites[a]
		# get neighboring plaquets
		m = (plaquet_sites==i1) | (plaquet_sites==i2) | (plaquet_sites==i3) | (plaquet_sites==i4)
		x,*_ = np.where(m)
		b_list = x[x!=a]
		# intra-plaquet bonds
		edge_dict[a] = set([(i1,i2),(i2,i3),(i3,i4),(i4,i1),(i1,i3),(i2,i4)])
		# NOTE: the tuples in edge_weights must match the tuples in the edge_dict
		edge_weights[(i1,i2)]=1
		edge_weights[(i2,i3)]=1
		edge_weights[(i3,i4)]=1
		edge_weights[(i4,i1)]=1
		edge_weights[(i1,i3)]=2
		edge_weights[(i2,i4)]=2
		# inter-plaquet edges
		for b in b_list:
			j1,j2,j3,j4 = plaquet_sites[b]
			if i1==j3:
				edge_dict[b] = set([(i2,j2),(i4,j4)])
				edge_weights[(i2,j2)] = 2
				edge_weights[(i4,j4)] = 2
			elif i2==j4:
				edge_dict[b] = set([(i1,j1),(i3,j3)])
				edge_weights[(i1,j1)] = 2
				edge_weights[(i3,j3)] = 2
			elif i4==j2:
				edge_dict[b] = set([(i1,j1),(i3,j3)])
				edge_weights[(i1,j1)] = 2
				edge_weights[(i3,j3)] = 2
			elif i3==j1:
				edge_dict[b] = set([(i2,j2),(i4,j4)])
				edge_weights[(i2,j2)] = 2
				edge_weights[(i4,j4)] = 2
		plaquet_edges[a] = edge_dict
	# construct plaquet expansion
	nlce = NLCE_plaquet(Ncl,plaquet_sites,plaquet_edges,trs,pgs,edge_weights)

Here is an illistration of the clusters that the LCE generates:

.. image:: images/sq_lat_plaq_nnn.png
   :height: 150
