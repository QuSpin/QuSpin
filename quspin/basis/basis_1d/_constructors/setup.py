









def cython_files():
	import os,glob
	try:
		from Cython.Build import cythonize
		USE_CYTHON = True
	except ImportError:
		USE_CYTHON = False

	package_dir = os.path.dirname(os.path.realpath(__file__))


	cython_src = glob.glob(os.path.join(package_dir,"*.pyx"))
	if USE_CYTHON:
		cythonize(cython_src,language="c++")



		



def configuration(parent_package='', top_path=None):
		import numpy,os
		from numpy.distutils.misc_util import Configuration
		config = Configuration('_constructors',parent_package, top_path)

		package_dir = os.path.dirname(os.path.realpath(__file__))

		hcp_src = os.path.join(package_dir,"hcp_basis_ops.cpp")	
		config.add_extension('hcp_basis_ops',sources=hcp_src,include_dirs=[numpy.get_include()],
								extra_compile_args=["-fno-strict-aliasing"],
								language="c++")

		boson_src = os.path.join(package_dir,"boson_basis_ops.cpp")	
		config.add_extension('boson_basis_ops',sources=boson_src,include_dirs=[numpy.get_include()],
								extra_compile_args=["-fno-strict-aliasing"],
								language="c++")

		return config

if __name__ == '__main__':
		from numpy.distutils.core import setup
		import sys
		try:
			instr = sys.argv[1]
			if instr == "build_templates":
				cython_files()
			else:
				setup(**configuration(top_path='').todict())
		except IndexError: pass





