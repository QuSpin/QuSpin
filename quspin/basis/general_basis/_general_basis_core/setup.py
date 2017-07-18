

def cython_files():
  import os,glob
  try:
    from Cython.Build import cythonize
    USE_CYTHON = True
  except ImportError:
    USE_CYTHON = False

  package_dir = os.path.dirname(os.path.realpath(__file__))
  cython_src = glob.glob(os.path.join(package_dir,"*.pyx"))
  include_dirs = os.path.join(package_dir,"source")
  print include_dirs
  if USE_CYTHON:
    cythonize(cython_src,language="c++",include_path=[include_dirs])


def configuration(parent_package='', top_path=None):
    import numpy,os
    from numpy.distutils.misc_util import Configuration
    config = Configuration('_general_basis_core',parent_package, top_path)

    package_dir = os.path.dirname(os.path.realpath(__file__))

    include_dirs = os.path.join(package_dir,"source")

    hcp_basis_src = os.path.join(package_dir,"hcb_core.cpp") 
    config.add_extension('hcb_core',sources=hcp_basis_src,include_dirs=[numpy.get_include(),include_dirs],
                extra_compile_args=["-fno-strict-aliasing","-std=c++11","-fopenmp"],
                extra_link_args=["-lgomp"],
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



