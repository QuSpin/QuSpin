"""
this script goes through and updates the version-strings for QuSpin both inside the package
as well as for the conda-build recipe. It also builds the documentation and moves it into the 'docs' folder
"""

quspin_ver = '"0.3.4"'
build_num = '"0"'

meta_template = """{version_text:s}

package:
  name: quspin
  version: {{{{ version }}}}

source:
  url: https://github.com/weinbe58/QuSpin/archive/dev_{{{{ version }}}}.zip

build:
  number: {{{{ build_num }}}}
  string: py{{{{ py_version | replace(".", "") }}}}h{{{{PKG_HASH}}}}_{{{{ build_num }}}}
  script: python setup.py install 
  ignore_run_exports:
    - boost

requirements:
  build:
    - {{{{compiler('cxx')}}}}

  host:{host_recipe:s}
 
  run:{run_recipe:s}
    
test:
  imports:
    - quspin 

about:
  home: https://github.com/weinbe58/QuSpin.git
  license: BSD-3
"""

meta_omp_template = """{version_text:s}

package:
  name: quspin
  version: {{{{ version }}}}

source:
  url: https://github.com/weinbe58/QuSpin/archive/dev_{{{{ version }}}}.zip

build:
  number: {{{{ build_num }}}}
  string: py{{{{ py_version | replace(".", "") }}}}h{{{{PKG_HASH}}}}_{{{{ build_num }}}}
  script: python setup.py install --omp
  features:
    - omp
  ignore_run_exports:
    - boost

requirements:
  build:
    - {{{{compiler('cxx')}}}}

  host:{host_recipe:s}
 
  run:{run_recipe:s}
    
test:
  imports:
    - quspin 

about:
  home: https://github.com/weinbe58/QuSpin.git
  license: BSD-3
"""

conda_build_config_template="""py_version:{python_text:s}

numpy:{numpy_text:s}
"""



numpy_versions = ["1.17.2"]
python_versions = ["3.6","3.7","3.8 # [not win]"]

pkg_vers = {
	"scipy":">=1.0.0",
	"joblib":"",
	"six":"",
	"dill":"",
	"numba":">=0.41",
	"numexpr":"",
	"gmpy2":"",
	"cython":">=0.29",
	"boost":"",
	"numpy":">="+numpy_versions[0],
	"python":">="+python_versions[0],
	"llvm-openmp":"",
}

pkg_urls = {
	"scipy":("https://www.scipy.org",),
	"numpy":("http://www.numpy.org",),
	"cython":("https://www.cython.org",),
	"joblib":("https://pythonhosted.org/joblib/",),
	"six":("https://pythonhosted.org/six/",),
	"dill":("https://pypi.python.org/pypi/dill",),
	"gmpy2":("https://gmpy2.readthedocs.io/en/latest/",),
	"numba":("http://numba.pydata.org/",),
	"numexpr":("https://numexpr.readthedocs.io/en/latest/user_guide.html",),
	"boost":("https://www.boost.org/doc/libs/1_70_0/libs/python/doc/html/index.html", "installation must include header files for boost."),
	"llvm-openmp":("http://openmp.llvm.org/", "osx openmp version only."),
}




host_pkg = {
	"numpy":"{{ numpy }}",
	"python":"{{ py_version }}",
}
host_pkg["cython"] = pkg_vers["cython"]
host_pkg["boost"] = pkg_vers["boost"]


run_pkg = {
	"python": "{{ py_version }}",
	"{{ pin_compatible('numpy') }}":"",
}
run_pkg["scipy"] = pkg_vers["scipy"]
run_pkg["joblib"] = pkg_vers["joblib"]
run_pkg["six"] = pkg_vers["six"]
run_pkg["dill"] = pkg_vers["dill"]
run_pkg["numexpr"] = pkg_vers["numexpr"]
run_pkg["gmpy2"] = pkg_vers["gmpy2"]
run_pkg["numba"] = pkg_vers["numba"]


host_recipe = "\n    "+("\n    ".join("- "+pkg+" "+ver for pkg,ver in host_pkg.items()))
run_recipe = "\n    "+("\n    ".join("- "+pkg+" "+ver for pkg,ver in run_pkg.items()))
version_text = """{%  set version = """+quspin_ver+""" %}\n{%  set build_num = """+build_num+""" %}"""

python_text="\n  "+("\n  ".join("- "+ver for ver in python_versions))
numpy_text="\n  "+("\n  ".join("- "+ver for ver in numpy_versions))

meta = meta_template.format(version_text=version_text,run_recipe=run_recipe,host_recipe=host_recipe)
meta_omp = meta_omp_template.format(version_text=version_text,run_recipe=run_recipe,host_recipe=host_recipe)
conda_build_config = conda_build_config_template.format(python_text=python_text,numpy_text=numpy_text)


with open("conda.recipe/quspin/meta.yaml","w") as IO:
	IO.write(meta)

with open("conda.recipe/quspin/conda_build_config.yaml","w") as IO:
	IO.write(conda_build_config)

with open("conda.recipe/quspin-omp/meta.yaml","w") as IO:
	IO.write(meta_omp)

with open("conda.recipe/quspin-omp/conda_build_config.yaml","w") as IO:
	IO.write(conda_build_config)

md_url = "[{text}]({url})"
rst_url = "`{text} <{url}>`_"

md_text = ["For the **manual installation** you must have all the prerequisite python packages installed:"]
rst_text = ["For the **manual installation** you must have all the prerequisite python packages installed:"]

for pkg,url in pkg_urls.items():
	if len(url) > 1:
		md_text += [" * "+md_url.format(text=pkg,url=url[0])+pkg_vers[pkg]+", "+url[1]]
		rst_text += ["    * "+rst_url.format(text=pkg,url=url[0])+pkg_vers[pkg]+", "+url[1]]
	else:
		md_text += [" * "+md_url.format(text=pkg,url=url[0])+pkg_vers[pkg]]
		rst_text += ["    * "+rst_url.format(text=pkg,url=url[0])+pkg_vers[pkg]]

with open("sphinx/source/Installation.rst","r") as IO:
	Installation_rst = IO.read()

Installation_rst_lines = Installation_rst.split("\n")
begin = Installation_rst_lines.index(".. begin packages")
end = Installation_rst_lines.index(".. end packages")

Installation_rst_lines = Installation_rst_lines[:begin+1]+rst_text+Installation_rst_lines[end:]

Installation_rst_new = "\n".join(Installation_rst_lines)

with open("sphinx/source/Installation.rst","w") as IO:
	IO.write(Installation_rst_new)

with open("README.md","r") as IO:
	readme_md = IO.read()

readme_md_lines = readme_md.split("\n")
begin = readme_md_lines.index("<!--- begin packages --->")
end = readme_md_lines.index("<!--- end packages --->")

readme_md_lines = readme_md_lines[:begin+1]+md_text+readme_md_lines[end:]

readme_md_new = "\n".join(readme_md_lines)

with open("README.md","w") as IO:
	IO.write(readme_md_new)