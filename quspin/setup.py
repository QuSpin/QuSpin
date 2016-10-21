def configuration(parent_package='',top_path=None):
	from numpy.distutils.misc_util import Configuration
	config = Configuration('quspin', parent_package, top_path)
	config.add_subpackage('basis')
	config.add_subpackage('operators')
	config.add_subpackage('tools')
	return config

if __name__ == '__main__':
	from numpy.distutils.core import setup
	setup(**configuration(top_path='').todict())
