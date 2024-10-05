
rm failed.txt

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4

for filename in example*.py 
do
	python $filename ${OMP_NUM_THREADS} ${OMP_NUM_THREADS} # command line arguments needed in example12.py 
	if [ $? -ne 0 ]; then
		echo $filename"\n" >> failed.txt
		echo $filename"\n" >> ../../failed.txt
	fi
done

rm *.pdf