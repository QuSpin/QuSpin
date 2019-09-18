import os
import sys
import subprocess
import glob

def get_include_dirs():
    import numpy,os
    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)
    include_dirs = {}
    include_dirs["numpy"] = numpy.get_include()
    include_dirs["source"] = os.path.join(package_dir,"source")
    #include_dirs["_oputils"] = os.path.join(package_dir,"..","..","operators","_oputils")
    include_dirs["_oputils"] = os.path.join(package_dir,"..","matvec","_oputils")

    return include_dirs




def cython_files():
    import os,glob
    from Cython.Build import cythonize

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    cython_src = glob.glob(os.path.join(package_dir,"*.pyx"))
    include_dirs = get_include_dirs()

    cythonize(cython_src,include_path=list(include_dirs.values()))


def configuration(parent_package='', top_path=None):
    import numpy,os,sys
    from numpy.distutils.misc_util import Configuration
    config = Configuration('expm_multiply_parallel_core',parent_package, top_path)

    subprocess.check_call([sys.executable,
                           os.path.join(os.path.dirname(__file__),
                                        'generate_source.py')])

    cython_files()

    extra_compile_args = ["-fno-strict-aliasing"]
    extra_link_args = []
    if sys.platform == "darwin":
        extra_compile_args.append("-std=c++11")

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)
    
    include_dirs = get_include_dirs()

    depends_csr = [os.path.join(include_dirs["source"],"csr_matvec.h")]

    depends_expm = [os.path.join(include_dirs["source"],"expm_multiply_parallel_impl.h")]

    include_dirs_list = list(include_dirs.values())

    src = os.path.join(package_dir,"expm_multiply_parallel_wrapper.cpp") 
    config.add_extension('expm_multiply_parallel_wrapper',sources=src,
                            include_dirs=include_dirs_list,
                            depends=depends_expm,
                            extra_compile_args=extra_compile_args,
                            extra_link_args=extra_link_args,
                            language="c++")

    src = os.path.join(package_dir,"csr_matvec_wrapper.cpp") 
    config.add_extension('csr_matvec_wrapper',sources=src,
                            include_dirs=[include_dirs["source"]],
                            depends=depends_csr,
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
