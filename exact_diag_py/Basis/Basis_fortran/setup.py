import glob,os
from numpy.distutils.core import Extension, setup


m_src=glob.glob(os.path.join('sources', 'm', '*.f90'))
z_src=glob.glob(os.path.join('sources', 'z', '*.f90'))
p_src=glob.glob(os.path.join('sources', 'p', '*.f90'))
pz_src=glob.glob(os.path.join('sources', 'pz', '*.f90'))
p_z_src=glob.glob(os.path.join('sources', 'p_z', '*.f90'))
sources=glob.glob(os.path.join('sources','*.f90'))+m_src+z_src+p_src+pz_src+p_z_src

setup(name='Basis_fortran',
       ext_modules=[Extension(name='Basis_fortran', sources=sources)],
       )

