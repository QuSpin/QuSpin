# QuSpin_recipes
Repository for conda-build recipes of QuSpin


% conda build from recipes and upload

1) `rm ~/miniconda3/conda-bld/src_cache/*.zip`
2) `conda build purge-all`
3) run the builds 

`conda build conda.recipe/quspin --no-anaconda-upload`

`conda build conda.recipe/omp --no-anaconda-upload`
`conda build conda.recipe/quspin-omp --no-anaconda-upload`

% to test dependencies use -t option


3a) test in a local conda env before uploading to anaconda cloud


`conda install quspin --use-local`
`conda install quspin omp --use-local` 

4) upload to anaconda cloud

`conda install conda-build anaconda-client` # if using miniconda

% upload omp build: screws up number of downloads
`anaconda upload -u weinbe58 (path_to_zip)`

% automatic upload # do NOT use
`conda config --set anaconda_upload yes`
