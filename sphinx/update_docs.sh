rm -rf ../docs/*
rm source/generated/*
make clean
make html
cp -rf _build/html/* ../docs/