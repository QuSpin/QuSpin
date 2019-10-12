from __future__ import division, print_function, absolute_import

import os
import sys
import subprocess
import glob


def cython_files():
    import numpy
    from Cython.Build import cythonize

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    cython_src = glob.glob(os.path.join(package_dir,"*.pyx"))

    include_dirs = [numpy.get_include()]
    include_dirs.append(os.path.join(package_dir,"_oputils"))

    cythonize(cython_src,include_path=include_dirs)

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy
    config = Configuration('matvec', parent_package, top_path)


    subprocess.check_call([sys.executable,
                           os.path.join(os.path.dirname(__file__),
                                        'generate_oputils.py')])

    cython_files()



    extra_compile_args=["-fno-strict-aliasing"]
    extra_link_args=[]  
      
    if sys.platform == "darwin":
        extra_compile_args.append("-std=c++11")

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    include_dirs = [numpy.get_include()]
    include_dirs.append(os.path.join(package_dir,"_oputils"))

    depends =[
        os.path.join(package_dir,"_oputils","oputils_impl.h"),
    ]
    src = [os.path.join(package_dir,"_oputils.cpp")]
    config.add_extension('_oputils',sources=src,include_dirs=include_dirs,
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args,
                            depends=depends,
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

