



def cython_files():
	import os,glob
	from Cython.Build import cythonize

	package_dir = os.path.dirname(os.path.realpath(__file__))
	package_dir = os.path.expandvars(package_dir)

	cython_src = [
					os.path.join(package_dir,"_basis_utils.pyx"),
				]
	cythonize(cython_src)



def configuration(parent_package='',top_path=None):
	from numpy.distutils.misc_util import Configuration
	import os,numpy
	config = Configuration('basis', parent_package, top_path)
	config.add_subpackage('basis_1d')
	config.add_subpackage('basis_general')
	config.add_subpackage('transformations')

	cython_files()

	package_dir = os.path.dirname(os.path.realpath(__file__))
	package_dir = os.path.expandvars(package_dir)

	basis_utils_src = os.path.join(package_dir,"_basis_utils.c")	
	config.add_extension('_basis_utils',sources=basis_utils_src,include_dirs=[numpy.get_include()],
							language="c")

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
