
rm failed.txt

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4


find . -type f -name '*test.py' -exec sh -c '
	for pathname
	do
		python $pathname
		if [ $? -ne 0 ]; then
			echo $pathname"\n" >> failed.txt
			echo $pathname"\n" >> ../failed.txt
		fi
	done' sh {} +

	