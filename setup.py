def configuration(parent_package="", top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration(None, parent_package, top_path)
    config.set_options(
        assume_default_configuration=True,
        delegate_options_to_subpackages=True,
        quiet=True,
    )

    config.add_subpackage("quspin")

    return config


def setup_package():
    try:
        import numpy
    except:
        raise ImportError("build requires numpy for fortran extensions")

    import os, sys

    if "--omp" in sys.argv:
        sys.argv.remove("--omp")
        if sys.platform == "win32":
            if "CFLAGS" in os.environ:
                os.environ["CFLAGS"] = os.environ["CFLAGS"] + " /openmp"
            else:
                os.environ["CFLAGS"] = "/openmp"
        else:
            if "CFLAGS" in os.environ:
                os.environ["CFLAGS"] = os.environ["CFLAGS"] + " -fopenmp"
            else:
                os.environ["CFLAGS"] = "-fopenmp"

    if "--default-compiler-flags" in sys.argv:
        sys.argv.remove("--default-compiler-flags")
        if sys.platform == "win32":
            pass
        else:
            os.environ["CFLAGS"] = os.environ["CFLAGS"] + " -O3 -march=native"

    io = open("./conda.recipe/quspin/meta.yaml", "r")
    meta_file = io.read()
    io.close()

    meta_file = meta_file.split()
    ind = meta_file.index("version")
    version = meta_file[ind + 2].replace('"', "")

    metadata = dict(
        name="quspin",
        version=version,
        author="Phillip Weinberg, Marin Bukov",
        author_email="weinbe58@gmail.com",
        maintainer="Phillip Weinberg, Marin Bukov, Markus Schmitt",
        maintainer_email="weinbe58@gmail.com",
        download_url="https://github.com/weinbe58/QuSpin.git",
        license="BSD",
        platforms=["Unix", "Windows"],
    )

    from numpy.distutils.core import setup

    metadata["configuration"] = configuration

    setup(**metadata)


if __name__ == "__main__":
    setup_package()
