# **QuSpin**

QuSpin is an open-source Python package for exact diagonalization and quantum dynamics of arbitrary boson, fermion and spin many-body systems, supporting the use of various (user-defined) symmetries in one and higher dimensional lattice systems and (imaginary) time evolution following a user-specified driving protocol. 

The complete ***Documentation*** for QuSpin (including a number of recent tutorials) can be found at 

[http://weinbe58.github.io/QuSpin/](http://weinbe58.github.io/QuSpin/)

***Examples*** with python scripts and Jupyter notebooks which show how to use QuSpin can be downloaded at 

http://weinbe58.github.io/QuSpin/Examples.html

For an ***indepth introduction*** to the package, check out the following papers:
* [SciPost Phys. __2__, 003 (2017)](https://scipost.org/10.21468/SciPostPhys.2.1.003)
* [SciPost Phys. __7__, 020 (2019)](https://scipost.org/10.21468/SciPostPhys.7.2.020)

QuSpin wraps Scipy, Numpy, and custom Cython libraries together to offer state-of-the art exact diagonalization calculations. The interface allows the user to define any many-body (and single particle) Hamiltonian which can be constructed from local single-particle operators. It also gives the user the flexibility of accessing many pre-defined symmetries in 1d (e.g. translation, reflection, spin inversion), as well as user-defined symmetry based on elementary transformations, such as site and spin flips. Moreover, there are convenient built-in ways to specify the time and parameter dependence of operators in the Hamiltonian, which is interfaced with user-friendly routines to solve the time dependent Schrödinger equation numerically. All the Hamiltonian data is stored either using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library for sparse Hamiltonians or dense Numpy [arrays](http://docs.scipy.org/doc/numpy/reference/index.html) which allows the user to access any powerful Python scientific computing tools.



# **Contents**
--------
* [Installation](#installation)
  * [automatic install](#automatic-install)
  * [manual install](#manual-install)
  * [updating the package](#updating-the-package)
* [Documentation](#documentation)

# **Installation**

### **automatic install**

The latest version of the package has the compiled modules written in [Cython](cython.org) which has made the code far more portable across different platforms. We will support precompiled version of the package for Linux, OS X and Windows 64-bit systems. The automatic installation of QuSpin requires the Anaconda package manager for Python. Once Anaconda has been installed, all one has to do to install QuSpin is run:
```
$ conda install -c weinbe58 quspin
```
This will install the latest version on your computer. Right now the package is in its beta stages and so it may not be available for installation on all platforms using this method. In such a case one can also manually install the package.

OpenMP support is available starting from QuSpin 0.3.1. To instal quspin with OpenMP, use
```
$ conda install -c weinbe58 omp quspin
```

### **manual install**

To install QuSpin manually, download the source code either from the [master](https://github.com/weinbe58/QuSpin/archive/master.zip) branch, or by cloning the git repository. In the top directory of the source code you can execute the following commands from bash:

Unix:
```
python setup.py install 
```
or Windows command line:
```
setup.py install
```
For the manual installation you must have all the prerequisite python packages: [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org>), [joblib](https://pythonhosted.org/joblib/), [six](https://pythonhosted.org/six), [dill](https://pypi.python.org/pypi/dill), [gmpy2](https://gmpy2.readthedocs.io/en/latest/), [numba](http://numba.pydata.org/), llvm-openmp (osx omp version only) installed. For Windows machines one needs the correct verion of the Microsoft Visual Stuios compiler for the given python version you are building the package for. A good resource which can help you with this is [here](https://github.com/cython/cython/wiki/CythonExtensionsOnWindows). For OS-X and Linux the standard compilers should be fine for building the package. Note that some of the compiled extensions require Openmp 2.0 or above. For most setups We recommend [Anaconda](https://www.continuum.io/downloads) or [Miniconda](http://conda.pydata.org/miniconda.html) to manage your python packages and we have pre-built code which should be compatible with most 64-bit systems. When installing the package manually, if you add the flag ```--record install.txt```, the location of all the installed files will be output to install.txt which stores information about all installed files. This can prove useful when updating the code.

For manual installation with OpenMP, use:

Unix:
```
python setup.py install --omp
```
or Windows command line:
```
setup.py install --omp
```

### **updating the package**

To update the package with Anaconda, all one has to do is run the installation command again.

To safely update a manually installed version of QuSpin, one must first manually delete the entire package from the python 'site-packages/' directory. In Unix, provided the flag ```--record install.txt``` has been used in the manual installation, the following command is sufficient to completely remove the installed files: ```cat install.txt | xargs rm -rf```. In Windows, it is easiest to just go to the folder and delete it from Windows Explorer. 

# **Documentation**

The complete QuSpin documentation can be found under

[http://weinbe58.github.io/QuSpin/](http://weinbe58.github.io/QuSpin/)
