

def get_include_dirs():
    from sysconfig import get_paths
    import numpy,os,sys

    package_dir = os.path.dirname(os.path.realpath(__file__))
    data_path = get_paths()["data"]

    package_dir = os.path.expandvars(package_dir)
    data_path = os.path.expandvars(data_path)

    include_dirs = [numpy.get_include()]
    include_dirs.append(os.path.join(package_dir,"source"))
    if sys.platform == "win32":
        include_dirs.append(os.path.join(data_path,"Library","include"))
    else:
        include_dirs.append(os.path.join(data_path,"include"))

    return include_dirs


def cython_files():
    import os,glob
    from Cython.Build import cythonize

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    cython_src = glob.glob(os.path.join(package_dir,"*.pyx"))
    # cython_src = [os.path.join(package_dir,"hcb_core.pyx")]

    include_dir = os.path.join(package_dir,"source")
    cythonize(cython_src,include_path=get_include_dirs())


def configuration(parent_package='', top_path=None):
        import numpy,os,sys,glob
        from numpy.distutils.misc_util import Configuration
        config = Configuration('_basis_general_core',parent_package, top_path)

        cython_files()

        extra_compile_args=["-fno-strict-aliasing"]
        extra_link_args=[]  
          
        if sys.platform == "darwin":
            extra_compile_args.append("-std=c++11")
    
        package_dir = os.path.dirname(os.path.realpath(__file__))
        package_dir = os.path.expandvars(package_dir)
        all_headers = set(glob.glob(os.path.join(package_dir,"source","*.h")))

        src_spec_headers = set(glob.glob(os.path.join(package_dir,"source","*_basis_core.h")))
        # needed for utils.
        src_spec_headers.add(os.path.join(package_dir,"source","general_basis_bitops.h"))
        # remove as it is actually a global header.
        src_spec_headers.remove(os.path.join(package_dir,"source","general_basis_core.h"))

        # remove specific headers from all the headers to get the shared headers.
        global_headers = list(all_headers - src_spec_headers)

        extension_kwargs = dict(include_dirs=get_include_dirs(),
                                extra_compile_args=extra_compile_args,
                                extra_link_args=extra_link_args,
                                language="c++")

        utils_src = os.path.join(package_dir,"general_basis_utils.cpp")
        utils_src_header = os.path.join(package_dir,"source","general_basis_bitops.h")
        depends = global_headers+[utils_src_header]
        config.add_extension('general_basis_utils',sources=utils_src,
            depends=depends,**extension_kwargs)

        hcp_src = os.path.join(package_dir,"hcb_core.cpp")
        hcp_src_header = os.path.join(package_dir,"source","hcb_basis_core.h")
        depends = global_headers+[hcp_src_header]
        config.add_extension('hcb_core',sources=hcp_src,
            depends=depends,**extension_kwargs)

        boson_src = os.path.join(package_dir,"boson_core.cpp")
        boson_src_header = os.path.join(package_dir,"source","boson_basis_core.h")
        depends = global_headers+[boson_src_header]
        config.add_extension('boson_core',sources=boson_src,
            depends=depends,**extension_kwargs)

        higher_spin_src = os.path.join(package_dir,"higher_spin_core.cpp") 
        higher_spin_src_header = os.path.join(package_dir,"source","higher_spin_basis_core.h")
        depends = global_headers+[higher_spin_src_header]
        config.add_extension('higher_spin_core',sources=higher_spin_src,
            depends=depends,**extension_kwargs)

        spinless_fermion_src = os.path.join(package_dir,"spinless_fermion_core.cpp") 
        spinless_fermion_src_header = os.path.join(package_dir,"source","spinless_fermion_basis_core.h")
        depends = global_headers+[spinless_fermion_src_header]
        config.add_extension('spinless_fermion_core',sources=spinless_fermion_src,
            depends=depends,**extension_kwargs)

        spinful_fermion_src = os.path.join(package_dir,"spinful_fermion_core.cpp") 
        spinful_fermion_src_header = os.path.join(package_dir,"source","spinful_fermion_basis_core.h")
        depends = global_headers+[spinful_fermion_src_header]
        config.add_extension('spinful_fermion_core',sources=spinful_fermion_src,
            depends=depends,**extension_kwargs)

        return config

if __name__ == '__main__':
        from numpy.distutils.core import setup
        setup(**configuration(top_path='').todict())



