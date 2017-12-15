









def cython_files():
	import os,glob
	try:
		from Cython.Build import cythonize
		USE_CYTHON = True
	except ImportError:
		USE_CYTHON = False

	package_dir = os.path.dirname(os.path.realpath(__file__))

	cython_src = [
					os.path.join(package_dir,"hcp_basis.pyx"),
					os.path.join(package_dir,"hcp_ops.pyx"),
					os.path.join(package_dir,"spf_basis.pyx"),
					os.path.join(package_dir,"spf_ops.pyx"),
					os.path.join(package_dir,"boson_basis.pyx"),
					os.path.join(package_dir,"boson_ops.pyx"),
				]
	if USE_CYTHON:
		cythonize(cython_src,language="c++")



		



def configuration(parent_package='', top_path=None):
		import numpy,os
		from numpy.distutils.misc_util import Configuration
		config = Configuration('_basis_1d_core',parent_package, top_path)

		package_dir = os.path.dirname(os.path.realpath(__file__))

		hcp_basis_src = os.path.join(package_dir,"hcp_basis.cpp")	
		config.add_extension('hcp_basis',sources=hcp_basis_src,include_dirs=[numpy.get_include()],
								extra_compile_args=["-fno-strict-aliasing"],
								language="c++")

		spf_ops_src = os.path.join(package_dir,"spf_basis.cpp")	
		config.add_extension('spf_basis',sources=spf_ops_src,include_dirs=[numpy.get_include()],
								extra_compile_args=["-fno-strict-aliasing"],
								language="c++")

		boson_basis_src = os.path.join(package_dir,"boson_basis.cpp")	
		config.add_extension('boson_basis',sources=boson_basis_src,include_dirs=[numpy.get_include()],
								extra_compile_args=["-fno-strict-aliasing"],
								language="c++")

		hcp_ops_src = os.path.join(package_dir,"hcp_ops.cpp")	
		config.add_extension('hcp_ops',sources=hcp_ops_src,include_dirs=[numpy.get_include()],
								extra_compile_args=["-fno-strict-aliasing"],
								language="c++")

		spf_ops_src = os.path.join(package_dir,"spf_ops.cpp")	
		config.add_extension('spf_ops',sources=spf_ops_src,include_dirs=[numpy.get_include()],
								extra_compile_args=["-fno-strict-aliasing"],
								language="c++")

		boson_ops_src = os.path.join(package_dir,"boson_ops.cpp")	
		config.add_extension('boson_ops',sources=boson_ops_src,include_dirs=[numpy.get_include()],
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





