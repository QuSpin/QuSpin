def configuration(parent_package='',top_path=None):
	from numpy.distutils.misc_util import Configuration
	import os
	config = Configuration('tools', parent_package, top_path)
	config.add_subpackage('expm_multiply_parallel_core')
	config.add_subpackage('matvec')
	config.add_subpackage('lanczos')
	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
