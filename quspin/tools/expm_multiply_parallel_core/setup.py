
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
        config = Configuration('expm_multiply_parallel_core',parent_package, top_path)

        extra_compile_args = []
        extra_link_args = []
        if sys.platform == "win32":
            extra_compile_args=["/openmp"]
            extra_link_args=["/openmp"]
        elif sys.platform == "darwin":
            extra_compile_args = []
            extra_link_args = []
        else:
            extra_compile_args = ["-fopenmp"]
            extra_link_args = ["-lgomp"]
 
        package_dir = os.path.dirname(os.path.realpath(__file__))
        include_dirs = os.path.join(package_dir,"source")

        src = os.path.join(package_dir,"expm_multiply_parallel_wrapper.cpp") 
        config.add_extension('expm_multiply_parallel_wrapper',sources=src,include_dirs=[numpy.get_include(),include_dirs],
                                extra_compile_args=extra_compile_args,
                                extra_link_args=extra_link_args,
                                language="c++")

        src = os.path.join(package_dir,"csr_matvec_wrapper.cpp") 
        config.add_extension('csr_matvec_wrapper',sources=src,include_dirs=[numpy.get_include(),include_dirs],
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
