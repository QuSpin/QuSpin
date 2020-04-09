# QuSpin_recipes
Repository for conda-build recipes of QuSpin

# conda build from recipes and upload

1) rm ~/miniconda3/conda-bld/src_cache/*.zip
2) conda build purge-all
3) run the builds 

conda build QuSpin_recipes/quspin

conda build QuSpin_recipes/omp
conda build QuSpin_recipes/quspin-omp

4) upload to anaconda cloud

conda install conda-build anaconda-client # if using miniconda

upload omp build!!!
anaconda upload -u weinbe58 (path_to_zip)

# automatic upload
conda config --set anaconda_upload yes
