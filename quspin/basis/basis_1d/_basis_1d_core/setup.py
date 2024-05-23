def cython_files():
	import os,glob
	from Cython.Build import cythonize

	package_dir = os.path.dirname(os.path.realpath(__file__))
	package_dir = os.path.expandvars(package_dir)

	cython_src = [
					os.path.join(package_dir,"hcp_basis.pyx"),
					os.path.join(package_dir,"hcp_ops.pyx"),
					os.path.join(package_dir,"spf_basis.pyx"),
					os.path.join(package_dir,"spf_ops.pyx"),
					os.path.join(package_dir,"boson_basis.pyx"),
					os.path.join(package_dir,"boson_ops.pyx"),
				]
	cythonize(cython_src)


def configuration(parent_package='', top_path=None):
	import numpy,os,sys
	from numpy.distutils.misc_util import Configuration
	config = Configuration('_basis_1d_core',parent_package, top_path)

	cython_files()

	if sys.platform == "win32":
		extra_compile_args=[]
	else:
		extra_compile_args=["-fno-strict-aliasing"]

	package_dir = os.path.dirname(os.path.realpath(__file__))
	package_dir = os.path.expandvars(package_dir)

	sources_dir = os.path.join(package_dir,"sources")

	hcp_basis_src = os.path.join(package_dir,"hcp_basis.cpp")	
	config.add_extension('hcp_basis',sources=hcp_basis_src,include_dirs=[numpy.get_include(),sources_dir],
							extra_compile_args=extra_compile_args,
							language="c++")

	spf_ops_src = os.path.join(package_dir,"spf_basis.cpp")	
	config.add_extension('spf_basis',sources=spf_ops_src,include_dirs=[numpy.get_include(),sources_dir],
							extra_compile_args=extra_compile_args,
							language="c++")

	boson_basis_src = os.path.join(package_dir,"boson_basis.cpp")	
	config.add_extension('boson_basis',sources=boson_basis_src,include_dirs=[numpy.get_include(),sources_dir],
							extra_compile_args=extra_compile_args,
							language="c++")

	hcp_ops_src = os.path.join(package_dir,"hcp_ops.cpp")	
	config.add_extension('hcp_ops',sources=hcp_ops_src,include_dirs=[numpy.get_include(),sources_dir],
							extra_compile_args=extra_compile_args,
							language="c++")

	spf_ops_src = os.path.join(package_dir,"spf_ops.cpp")	
	config.add_extension('spf_ops',sources=spf_ops_src,include_dirs=[numpy.get_include(),sources_dir],
							extra_compile_args=extra_compile_args,
							language="c++")

	boson_ops_src = os.path.join(package_dir,"boson_ops.cpp")	
	config.add_extension('boson_ops',sources=boson_ops_src,include_dirs=[numpy.get_include(),sources_dir],
							extra_compile_args=extra_compile_args,
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





