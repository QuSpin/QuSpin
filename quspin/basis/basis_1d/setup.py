def configuration(parent_package='', top_path=None):
		import numpy
		from numpy.distutils.misc_util import Configuration
		config = Configuration('basis_1d',parent_package, top_path)
		config.add_subpackage('_basis_1d_core')
		return config

if __name__ == '__main__':
		from numpy.distutils.core import setup
		setup(**configuration(top_path='').todict())
