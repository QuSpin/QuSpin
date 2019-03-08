rm failed.txt

cd tests/
sh run_test.sh

cd ../sphinx/doc_examples/
sh run_examples.sh

cd ../../examples/scripts/
sh run_examples.sh

cd ../notebooks/
sh run_scripts.sh
