# **Tools.observables:** 

## Entanglement entropy:

```python
Entanglement_entropy(L,psi,subsys=[i for i in xrange( int(L/2) )],basis=None,alpha=1.0, DM=False, DM_basis=False) 
```

This is routine calculates the entanglement (Renyi) entropy if a pure squantum state in a subsystem of choice. It returns a dictionary in which the entanglement (Renyi) entropy has the key 'Sent'. The arguments are:

L: (compulsory) chain length. Always the first argument.

psi: (compulsory) a pure quantum state, to calculate the entanglement entropy of. Always the second argument.

basis: (semi-compulsory) basis of psi. If the state psi is written in a symmetry-reduced basis, then one must also parse the basis in which psi is given. However, if no symmetry is invoked and the basis of psi contains all 2^L states, one can ommit the basis argument.

subsys: (optional) a list of site numbers which define uniquely the subsystem of which the entanglement entropy (reduced and density matrix) are calculated. Notice that the site labelling of the chain goes as [0,1,....,L-1]. If not specified, the default subsystem chosen is [0,...,floor(L/2)].

alpha: (optional) Renyi parameter alpha. The default is 'alpha=1.0'.

DM: (optional) when set to True, the returned dictionary contains the reduced density matrix under the key 'DM'. Note that the
reduced DM is written in the full basis containing 2^L states. 









## Diagonal Ensemble Observables:
```python
Diag_Ens_Observables(L,V1,V2,E1,betavec=[],alpha=1.0, Obs=False, Ed=False, S_double_quench=False, Sd_Renyi=False, deltaE=False)
```

This is routine calculates the expectation values of physical quantities in the Diagonal ensemble (see eg. arXiv:1509.06411). It returns a dictionary. Equivalently, these are the infinite-time expectation values after a sudden quench at time t=0 from the Hamiltonian H1 to the Hamiltonian H2. 

L: (compulsory) chain length. Always the first argument.

V1: (compulsory) unitary square matrix. Contains the eigenvectors of H1 in the columns. The initial state is the first column of V1. Always the second argument.

V2: (compulsory) unitary square matrix. Contains the eigenvectors of H2 in the columns. Must have the same size as V1. Always the third argument.

E1: (compulsory) vector of real numbers. Contains the eigenenergies of H1. The order of the eigenvalues must correspond to the order of the columns of V1. Always the fourth argument.

Obs: (optional) hermitian matrix of the same size as V1. Infinite-time expectation value of the observable Obs in the state V1[:,0]. Has the key 'Obs' in the returned dictionary.

Ed: (optional) infinite-time expectation value of the Hamiltonian H1 in the state V1[:,0]. Hast the key 'Ed' in the returned dictionary.

deltaE: (optional) infinite-time fluctuations around the energy expectation Ed. Has the key 'deltaE' in the returned dictionary.

Sd_Renyi: (optional) diagonal Renyi entropy after a quench H1->H2. The default Renyi parameter is 'alpha=1.0'. Has the key 'Sd_Renyi' in the returned dictionary.

alpha: (optional) diagonal Renyi entropy parameter. Default value is 'alpha=1.0'.

S_double_quench: (optional) diagonal entropy after a double quench H1->H2->H1. Has the key 'S_double_quench' in the returned dictionary.

betavec: (optional) a list of INVERSE temperatures to specify the distribution of an initial thermal state. When passed the routine returns the corresponding finite-temperature expectation of each specified quantity defined above. The corresponding keys in the returned dictionary are 'Obs_T', 'Ed_T', 'deltaE_T', 'Sd_Renyi_T', 'S_double_quench_T'. 







## Kullback-Leibler divergence:
```python
Kullback_Leibler_div(p1,p2)
```
This routine returns the Kullback-Leibler divergence of the discrete probabilities p1 and p2. 







## Time Evolution:
```
Observable_vs_time(psi,V2,E2,Obs,times,return_state=False)
```
This routine calculate the expectation value as a function of time of an observable Obs. The initial state is psi and the time evolution is carried out under the Hamiltonian H2. Returns a dictionary in which the time-dependent expectation value has the key 'Expt_time'.

psi: (compulsory) initial state. Always first argument.

V2: (compulsory) unitary matrix containing in its columns all eigenstates of the Hamiltonian H2. Always second argument.

E2: (compulsory) real vector containing the eigenvalues of the Hamiltonian H2. The order of the eigenvalues must correspond to the order of the columns of V2. Always third argument.

Obs: (compulsory) hermitian matrix to calculate its time-dependent expectation value. Always fourth argument.

times: (compulsory) a vector of times to evaluate the expectation value at. always fifth argument.

return_state: (optional) when set to 'True', returns a matrix whose columns give the state vector at the times specified by the row index. The return dictonary key is 'psi_time'.



## Mean Level spacing:
```python
Mean_Level_Spacing(E)
```
This routine returns the mean-level spacing r_ave of the energy distribution E, see [arXiv:1212.5611](http://arxiv.org/abs/1212.5611). 

E: (compulsory) ordered list of ascending, nondegenerate eigenenergies. 






