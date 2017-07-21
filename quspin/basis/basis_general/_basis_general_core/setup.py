

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
  if USE_CYTHON:
    cythonize(cython_src,language="c++",include_path=[include_dirs])


def configuration(parent_package='', top_path=None):
    import numpy,os,sys
    from numpy.distutils.misc_util import Configuration
    config = Configuration('_basis_general_core',parent_package, top_path)

    extra_compile_args = []
    extra_link_args = []
    if sys.platform == "win32":
      pass
      extra_compile_args=["/openmp"]
      extra_link_args=["/openmp"]
    else:
      pass
      extra_compile_args = ["-fopenmp"]
      extra_link_args = ["-fopenmp"]
 
    package_dir = os.path.dirname(os.path.realpath(__file__))
    include_dirs = os.path.join(package_dir,"source")

    hcp_basis_src = os.path.join(package_dir,"hcb_core.cpp") 
    config.add_extension('hcb_core',sources=hcp_basis_src,include_dirs=[numpy.get_include(),include_dirs],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++")

    boson_basis_src = os.path.join(package_dir,"boson_core.cpp") 
    config.add_extension('boson_core',sources=boson_basis_src,include_dirs=[numpy.get_include(),include_dirs],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
                language="c++")

    boson_basis_src = os.path.join(package_dir,"higher_spin_core.cpp") 
    config.add_extension('higher_spin_core',sources=boson_basis_src,include_dirs=[numpy.get_include(),include_dirs],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
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



