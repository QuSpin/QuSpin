def configuration(parent_package='',top_path=None):
	from numpy.distutils.misc_util import Configuration
	import os
	config = Configuration('exact_diag_py', parent_package, top_path)
	config.add_subpackage('Basis')
	config.add_subpackage('py_lapack')
	config.add_subpackage('spins1D',subpackage_path=os.path.sep)
	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
