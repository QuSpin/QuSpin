.. _installation-label:

Installation
============

QuSpin is currently being supported for Python 3 which is a prerequisite. We recommend the use of the free python package installer `pip <https://pypi.org/project/pip/>`_ which installs Python packages and manages them automatically.

The old anaconda guide: :ref:`installation_old-label` has been **deprecated** with `quspin>=1.0.0`.


User Install (Automatic)
++++++++++++++++++++++++

The user install is meant for deploying `quspin` on your private machine or an HPC cluster:
::

	> pip install quspin


Developer Install (Manual)
++++++++++++++++++++++++++

Clone the `QuSpin Workspace <https://github.com/QuSpin/QuSpin-workspace>`_ repository:
::

	> git clone https://github.com/QuSpin/QuSpin-workspace


Initialize the submodules to pull the code:
::

	> git submodule init 
	> git submodule update

You will see three directories, each pointing to its own repository:
	- `sparse parallel tools extension <https://github.com/QuSpin/parallel-sparse-tools>`_ which contains the cpp code for the `tools` module;
	- `QuSpin extension <https://github.com/QuSpin/QuSpin-Extensions>`_ which contains the cpp code for the basis modules;
	- `QuSpin <https://github.com/QuSpin/QuSpin>`_ with the quspin python package that uses the other two modules. 

Create a `python>3.9` virtual environment. This can be done using `miniconda <http://conda.pydata.org/miniconda.html>`_, or using python itself:
::

	> cd QuSpin-workspace/
	> python3 -m venv .quspin_env

Activate the environment:
::

	> source .quspin_env/bin/activate 

Double check if the python and pip binaries point to the environment path:
::

	> which python
	> which pip

Install extension modules (may take a bit of time to build the cpp code):
::

	> pip install -e parallel-sparse-tools/ -v
	> pip install -e QuSpin-Extensions/ -v
	> pip install -e QuSpin/ -v


Make sure you add an exhaustive test to test any code you want to add to the package. To run unit tests, you can use `pytest`:
::

	> pip install pytest
	> cd QuSpin-workspace/QuSpin/tests
	> pytest  `# runs all tests in all subdirectories`
	> pytest test_specific.py

Make sure you also comment your code and add a your new public functions to the documentation. For the documentation, you need to install in addition:
::

> pip install -U sphinx
> pip install numpydoc
> pip install sphinx-rtd-theme
> pip install sphinx-rtd-size

To build the documentation, do:
::

	> cd QuSpin-workspace/QuSpin/sphinx/
	> rm ./source/generated/*   `# removes previously generated doc files`
	> make clean
	> make html
	> open _build/html/index.html



Basics of command line use
==========================



Let us review how to use the command line for Windows and OS X/Linux to navigate your computer's folders/directories and run the Python scripts.

	

Mac OS X/Linux
++++++++++++++



Some basic commands:

	* change directory:

		::

		

			$ cd < path_to_directory >

		

	* list files in current directory:

		::



			$ ls 

		

	* list files in another directory:

		::



			$ ls < path_to_directory >

		

	* make new directory:

		::



			$ mkdir <path>/< directory_name >

		

	* copy file:

		::



			$ cp < path >/< file_name > < new_path >/< new_file_name >

		

	* move file or change file name:

		::



			$ mv < path >/< file_name > < new_path >/< new_file_name >

		

	* remove file:

		::



			$ rm < path_to_file >/< file_name >

				

Unix also has an auto complete feature if one hits the TAB key. It will complete a word or stop when it matches more than one file/folder name. The current directory is denoted by "." and the directory above is "..".

	

	

Windows
+++++++



Some basic commands:

	* change directory:

		::



			> cd < path_to_directory >

		

	* list files in current directory:

		::



			> dir

		

	* list files in another directory:

		::



			> dir < path_to_directory >

		

	* make new directory:

		::



			> mkdir <path>\< directory_name >

		

	* copy file:

		::



			> copy < path >\< file_name > < new_path >\< new_file_name >

		

	* move file or change file name:

		::



			> move < path >\< file_name > < new_path >\< new_file_name >

		

	* remove file:

		::



			> erase < path >\< file_name >

		

		

Windows also has a auto complete feature using the TAB key but instead of stopping when there multiple files/folders with the same name, it will complete it with the first file alphabetically. The current directory is denoted by "." and the directory above is "..".

	

Execute Python Script (any operating system)
++++++++++++++++++++++++++++++++++++++++++++

	

To execute a Python script all one has to do is open up a terminal and navigate to the directory which contains the Python script. Python can be recognised by the extension `.py`. To execute the script just use the following command:

::



	python script.py



It's that simple! 