
def cython_files():
    import os,glob,numpy
    from Cython.Build import cythonize

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    cython_src = glob.glob(os.path.join(package_dir,"*.pyx"))

    include_dirs = [numpy.get_include()]
    include_dirs.append(os.path.join(package_dir,"_oputils"))

    cythonize(cython_src,include_path=include_dirs)

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import os,numpy,sys
    config = Configuration('operators', parent_package, top_path)

    cython_files()

    extra_compile_args=["-fno-strict-aliasing"]
    extra_link_args=[]  
      
    if sys.platform == "darwin":
        extra_compile_args.append(["-std=c++11"])

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    include_dirs = [numpy.get_include()]
    include_dirs.append(os.path.join(package_dir,"_oputils"))

    depends =[
        os.path.join(package_dir,"_oputils","matvec.h"),
        os.path.join(package_dir,"_oputils","matvecs.h"),
        os.path.join(package_dir,"_oputils","csrmv_merge.h"),
    ]
    src = os.path.join(package_dir,"_oputils.cpp") 
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

