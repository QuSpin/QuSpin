import glob,os
from numpy.distutils.core import Extension, setup

sources=glob.glob(os.path.join('sources','*.f90'))

setup(name='py_lapack_wrap',
       ext_modules=[Extension(name='py_lapack_wrap', sources=sources, extra_link_args=['-llapack'])],
       )


