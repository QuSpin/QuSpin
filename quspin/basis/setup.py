
def get_include_dirs():
	from sysconfig import get_paths
	import numpy,os,sys

	package_dir = os.path.dirname(os.path.realpath(__file__))
	data_path = get_paths()["data"]

	package_dir = os.path.expandvars(package_dir)
	data_path = os.path.expandvars(data_path)

	include_dirs = [numpy.get_include()]
	include_dirs.append(os.path.join(package_dir,"_basis_utils"))

	
	if sys.platform == "win32":
		include_dirs.append(os.path.join(data_path,"Library","include"))
	else:
		include_dirs.append(os.path.join(data_path,"include"))

	return include_dirs


def cython_files():
	import os,glob
	from Cython.Build import cythonize

	package_dir = os.path.dirname(os.path.realpath(__file__))
	package_dir = os.path.expandvars(package_dir)

	cython_src = [
					os.path.join(package_dir,"_basis_utils.pyx"),
				]
	cythonize(cython_src,include_path=get_include_dirs())



def configuration(parent_package='',top_path=None):
	from numpy.distutils.misc_util import Configuration
	import os,numpy,sys
	config = Configuration('basis', parent_package, top_path)
	config.add_subpackage('basis_1d')
	config.add_subpackage('basis_general')
	config.add_subpackage('transformations')

	cython_files()

	package_dir = os.path.dirname(os.path.realpath(__file__))
	package_dir = os.path.expandvars(package_dir)
	extra_compile_args=["-fno-strict-aliasing"]
	extra_link_args=[]  
	  
	if sys.platform == "darwin":
		extra_compile_args.append("-std=c++11")

	depends = [os.path.join(package_dir,"_basis_utils","shuffle_sites.h")]
	basis_utils_src = os.path.join(package_dir,"_basis_utils.cpp")	
	config.add_extension('_basis_utils',sources=basis_utils_src,include_dirs=get_include_dirs(),
							language="c++",depends=depends,extra_compile_args=extra_compile_args,extra_link_args=extra_link_args)

	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
