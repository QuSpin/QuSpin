# **qspin**

This documentation is also available as a jupyter [notebook](https://github.com/weinbe58/qspin/blob/master/documentation.ipynb). 

qspin is a python library which wraps Scipy, Numpy, and custom fortran libraries together to do state-of-the art exact diagonalization calculations on one-dimensional spin-1/2 chains with length up to 32 sites (including). The interface allows the user to define any Hamiltonian which can be constructed from spin-1/2 operators. It also gives the user the flexibility of accessing many symmetries in 1d. Moreover, there is a convenient built-in way to specifying the time dependence of operators in the Hamiltonian, which is interfaced with user-friendly routines to solve the time dependent Schrödinger equation numerically. All the Hamiltonian data is stored either using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library for sparse Hamiltonians or dense Numpy [arrays](http://docs.scipy.org/doc/numpy/reference/index.html) which allows the user to access any powerful Python scientific computing tools.

# **Contents**
--------
* [Installation](#Installation)
 * [automatic install](#automatic-install)
 * [manual install](#manual-install)
 * [updating the package](#updating-the-package)
* [Basic package usage](#Basic-package-usage)
 * [constructing hamiltonians](#constructing-hamiltonians)
 * [using basis objects](#using-basis-objects)
 * [specifying symmetries](#using-symmetries)
* [List of package functions](#List-of-package-functions) 
	* [operator objects](#operator-objects)
	 * [hamiltonian class](#hamiltonian-class)
	 * [useful hamiltonian functions](#useful-hamiltonian-functions)
	 * [matrix exponential](#exp_op-class)
	 * [HamiltonianOperator class](#HamiltonianOperator-class)
	* [basis objects](#basis-objects)
	 * [spin_basis in 1d](#spin_basis_1d-class)
	 * [harmonic oscillator basis](#ho_basis-class)
	 * [tensor basis objects](#tensor_basis-class)
     * [photon basis in 1d](#photon_basis-class)
     * [symmetry and hermiticity checks](#symmetry-and-hermiticity-checks)
	 * [methods for basis objects](#methods-for-basis-objects)
	* [tools](#tools)
	 * [measusrements](#measurements)
	 * [floquet](#floquet)

# **Installation**

### **automatic install**

The latest version of the package has the compiled modules written in [Cython](cython.org) which has made the code far more portable across different platforms. We will support precompiled version of the package for Linux, OS X and Windows 64-bit systems. The automatic installation of qspin requires the Anaconda package manager for Python. Once Anaconda has been installed, all one has to do to install qspin is run:
```
$ conda install -c weinbe58 qspin
```
This will install the latest version on your computer. Right now the package is in its beta stages and so it may not be available for installation on all platforms using this method. In such a case one can also manually install the package.

### **manual install**

To install qspin manually, download the source code either from the [master](https://github.com/weinbe58/qspin/archive/master.zip) branch, or by cloning the git repository. In the top directory of the source code you can execute the following commands from bash:

Unix:
```
python setup.py install 
```
or Windows command line:
```
setup.py install
```
For the manual installation you must have all the prerequisite python packages: [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), and [joblib](https://pythonhosted.org/joblib/) installed. We recommend [Anaconda](https://www.continuum.io/downloads) or [Miniconda](http://conda.pydata.org/miniconda.html) to manage your python packages.

When installing the package manually, if you add the flag ```--record install.txt```, the location of all the installed files will be output to install.txt which stores information about all installed files. This can prove useful when updating the code. 

### **updating the package**

To update the package with Anaconda, all one has to do is run the installation command again.

To safely update a manually installed version of qspin, one must first manually delete the entire package from the python 'site-packages/' directory. In Unix, provided the flag ```--record install.txt``` has been used in the manual installation, the following command is sufficient to completely remove the installed files: ```cat install.txt | xargs rm -rf```. In Windows, it is easiest to just go to the folder and delete it from Windows Explorer. 