# **QuSpin**

QuSpin is an open-source Python package for exact diagonalization and quantum dynamics of arbitrary boson, fermion and spin many-body systems. QuSpin supports the use of various (user-defined) symmetries for one and higher-dimensional lattice systems, (imaginary) time evolution following arbitrary user-specified driving protocols, constrained Hilbert spaces, and parallel sparse linear algebra tools.

The complete ***Documentation*** for QuSpin (including a number of recent tutorials) can be found at 

[http://quspin.github.io/QuSpin/](http://quspin.github.io/QuSpin/)

***Examples*** with python scripts and Jupyter notebooks which show how to use QuSpin can be downloaded at 

http://quspin.github.io/QuSpin/Examples.html

For an ***indepth introduction*** to the package, check out the following papers:
* [SciPost Phys. __2__, 003 (2017)](https://scipost.org/10.21468/SciPostPhys.2.1.003)
* [SciPost Phys. __7__, 020 (2019)](https://scipost.org/10.21468/SciPostPhys.7.2.020)

QuSpin wraps Scipy, Numpy, and custom C++/Cython libraries together to offer state-of-the art exact diagonalization calculations. The interface allows the user to define any many-body (and single particle) Hamiltonian which can be constructed from local single-particle operators. It also gives the user the flexibility of accessing many pre-defined symmetries in 1d (e.g. translation, reflection, spin inversion), as well as user-defined symmetry based on elementary transformations, such as site and spin flips. Moreover, there are convenient built-in ways to specify the time and parameter dependence of operators in the Hamiltonian, which is interfaced with user-friendly routines to solve the time dependent Schrödinger equation numerically. All the Hamiltonian data is stored either using Scipy's [sparse matrix](http://docs.scipy.org/doc/scipy/reference/sparse.html) library for sparse Hamiltonians or dense Numpy [arrays](http://docs.scipy.org/doc/numpy/reference/index.html) which allows the user to access any powerful Python scientific computing tools.


# **Contents**
--------
* [Installation](#installation)
  * [automatic install](#automatic-install)
  * [developer install](#developer-install)
  * [updating the package](#updating-the-package)
* [Documentation](#documentation)


# **Installation**

### **automatic install**

The latest version of the package has the compiled modules written in [Cython](cython.org) which has made the code far more portable across different platforms. We will support precompiled version of the package for Linux, OS X and Windows 64-bit systems. The automatic installation of QuSpin requires the [pip](https://pypi.org/project/pip/) package manager for Python. Once pip has been installed, all one has to do to install QuSpin is run:
```
$ pip install quspin
```
This will install the latest version on your computer. Right now the package is in its beta stages and so it may not be available for installation on all platforms using this method. In such a case one can also manually install the package.

OpenMP support is automatically built in starting from QuSpin 1.0.0. 


### **developer install**


Clone the [QuSpin Workspace](https://github.com/QuSpin/QuSpin-workspace) repository:
```
$ git clone https://github.com/QuSpin/QuSpin-workspace
```

Initialize the submodules to pull the code:
```
$ git submodule init 
$ git submodule update
```
  
You will see three directories, each pointing to its own repository:
  * [sparse parallel tools extension](https://github.com/QuSpin/parallel-sparse-tools) which contains the cpp code for the `tools` module;
  * [QuSpin extension](https://github.com/QuSpin/QuSpin-Extensions) which contains the cpp code for the basis modules;
  * [QuSpin](https://github.com/QuSpin/QuSpin) with the quspin python package that uses the other two modules. 

Create a python>3.9 virtual environment. This can be done using [miniconda](http://conda.pydata.org/miniconda.html), or using python itself:
```
$ cd QuSpin-workspace/
$ python3 -m venv .quspin_env
```

Activate the environment:
```
$ source .quspin_env/bin/activate 
```

Double check if the python and pip binaries point to the environment path:
```
$ which python
$ which pip
```

Install extension modules (may take a bit of time to build the cpp code):
```
$ pip install -e parallel-sparse-tools/ -v
$ pip install -e QuSpin-Extensions/ -v
$ pip install -e QuSpin/ -v
```

Make sure you add an exhaustive test to test any code you want to add to the package. To run unit tests, you can use pytest:
```
$ pip install pytest
$ cd QuSpin-workspace/QuSpin/tests
$ pytest  `# runs all tests in all subdirectories`
$ pytest test_specific.py
```

Make sure you also comment your code and add a your new public functions to the documentation. For the documentation, you need to install in addition:
```
$ pip install -U sphinx
$ pip install numpydoc
$ pip install sphinx-rtd-theme
$ pip install sphinx-rtd-size
```

To build the documentation, do:
```
$ cd QuSpin-workspace/QuSpin/sphinx/
$ rm ./source/generated/*   `# removes previously generated doc files`
$ make clean
$ make html
$ open _build/html/index.html
```


### **updating the package**

To update the package with pip, all one has to do is run the command.
```
$ pip install --upgrade quspin
```

# **Documentation**

The complete QuSpin documentation can be found under

[http://quspin.github.io/QuSpin/](http://quspin.github.io/QuSpin/)
