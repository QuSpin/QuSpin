def configuration(parent_package='',top_path=None):
	from numpy.distutils.misc_util import Configuration
	import os
	config = Configuration('basis', parent_package, top_path)
	config.add_subpackage('basis_1d')
	config.add_subpackage('basis_general')
	config.add_subpackage('transformations')
	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
