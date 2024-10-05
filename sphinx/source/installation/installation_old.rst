.. _installation_old-label:

Installation (Outdated)
=======================

:red:`This istallation guide is deprecated starting with quspin 1.0.0; check out the current pip` :ref:`installation-label`.


QuSpin is currently being supported for Python 3 which is a prerequisite. We recommend the use of the free package manager `Anaconda <https://www.continuum.io/downloads>`_ which installs Python and manages its packages. For a lighter installation (preferred for computing clusters), one can use `miniconda <http://conda.pydata.org/miniconda.html>`_.



.. begin packages

For the **manual installation** you must have all the prerequisite python packages installed:
    * `scipy <https://www.scipy.org>`_>=0.19.1
    * `numpy <http://www.numpy.org>`_>=1.19.2
    * `cython <https://www.cython.org>`_>=0.29
    * `joblib <https://pythonhosted.org/joblib/>`_
    * `six <https://pythonhosted.org/six/>`_
    * `dill <https://pypi.python.org/pypi/dill>`_
    * `gmpy2 <https://gmpy2.readthedocs.io/en/latest/>`_
    * `numba <http://numba.pydata.org/>`_
    * `numexpr <https://numexpr.readthedocs.io/en/latest/user_guide.html>`_
    * `boost <https://www.boost.org/doc/libs/1_70_0/libs/python/doc/html/index.html>`_, installation must include header files for boost.
    * `llvm-openmp <http://openmp.llvm.org/>`_, osx openmp version only.

.. end packages





On Windows machines one needs the correct version of the Microsoft Visual Studios compiler for the given python version one is building the package for. A good resource which can help with this can be found `here <https://github.com/cython/cython/wiki/CythonExtensionsOnWindows>`_. For OS-X and Linux the standard compilers should be fine for building the package. Note that some of the compiled extensions require Openmp 2.0 or above. When installing the package manually, if you add the flag `--record install.txt`, the location of all the installed files will be output to the file `install.txt`. This is useful as most package managers will not be able to remove manually installed packages and so in order to delete this package completely one needs to manually remove all the files. 

	

Mac OS X/Linux

--------------



To install Anaconda/miniconda all one has to do is execute the installation script with administrative privilege. To do this, open up the terminal and go to the folder containing the downloaded installation file and execute the following command:

::



	$ sudo bash <installation_file>



You will be prompted to enter your password. Follow the prompts of the installation. We recommend that you allow the installer to prepend the installation directory to your PATH variable which will make sure this installation of Python will be called when executing a Python script in the terminal. If this is not done then you will have to do this manually in your bash profile file:

::



	export PATH="path_to/anaconda/bin:$PATH"





**Installing via Anaconda**



Once you have Anaconda/miniconda installed, all you have to do to install QuSpin is to execute the following command into the terminal: 

::



	$ conda install -c weinbe58 quspin



or if you require OpenMP support (see also :ref:`parallelization-label`)

::



	$ conda install -c weinbe58 omp quspin



If asked to install new packages just say `yes`. To keep the code up-to-date, just run this command regularly. If you would like to go back to the single-threaded (i.e. no-OpenMP) version of QuSpin run:

::



	$ conda remove --features omp -c weinbe58



upon which you will be asked by anaconda if you want to downgrade you QuSpin version to a version which no longer tracks the `omp` feature. 

	

**Installing Manually**



Installing the package manually is not recommended unless the above method failed. Note that you must have the requisite Python packages [see above] installed before installing QuSpin. Once all the prerequisite packages are installed, one can download the source code from `github <https://github.com/weinbe58/qspin/tree/master>`_ and then extract the code to whichever directory one desires. Open the terminal and go to the top level directory of the source code and execute:

:: 



	$ python setup.py install --default-compiler-flags --record install_file.txt



or if you require OpenMP support (see also :ref:`parallelization-label`)

::



	$ python setup.py install --omp --default-compiler-flags --record install_file.txt



This will compile the source code and copy it to the installation directory of Python recording the installation location to `install_file.txt`. To update the code, you must first completely remove the current version installed and then install the new code. The `install_file.txt` can be used to remove the package by running:  

::



	$ cat install_file.txt | xargs rm -rf. 

	

	

**Installing without sudo Privileges**



Sometimes, when one does not have sudo privileges (i.e. access to the root directory is denied), one may not be able to install QuSpin directly. This is often the case on computing clusters, where one can only install programs in one's home directory. To circumvent such problems, we advise the users to (i) download and install `miniconda <http://conda.pydata.org/miniconda.html>`_ in their home directory, (ii) create a `conda environment <https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands>`_ and activate it, and (iii) install QuSpin inside the environment. To use QuSpin, one always has to activate the environment first.  



Windows

-------



To install Anaconda/miniconda on Windows, download the installer and execute it to install the program. Once Anaconda/miniconda is installed open the conda terminal and do one of the following to install the package:

	

**Installing via Anaconda**



Once you have Anaconda/miniconda installed all you have to do to install QuSpin is to execute the following command into the terminal: 

::



	> conda install -c weinbe58 quspin



or if you require OpenMP support (see also :ref:`parallelization-label`)

::



	> conda install -c weinbe58 omp quspin



If asked to install new packages just say `yes`. To update the code just run this command regularly. 

	

**Installing Manually**



Installing the package manually is not recommended unless the above method failed. NNote that you must have the requisite Python packages [see above] installed before installing QuSpin. Once all the prerequisite packages are installed, one can download the source code from `github <https://github.com/weinbe58/qspin/tree/master>`_ and then extract the code to whichever directory one desires. Open the terminal and go to the top level directory of the source code and then execute:  

::



	> python setup.py install --default-compiler-flags --record install_file.txt



or if you require OpenMP support (see also :ref:`parallelization-label`)

::



	> python setup.py install --omp --default-compiler-flags --record install_file.txt



This will compile the source code and copy it to the installation directory of Python and record the installation location to `install_file.txt`. To update the code you must first completely remove the current version installed and then install the new code. 


