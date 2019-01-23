
export OMP_NUM_THREADS=2

rm failed.txt

export OMP_NUM_THREADS=2

for filename in *test.py
do
	python $filename
	if [ $? -ne 0 ]; then
		echo $filename"\n" >> failed.txt
		echo $filename"\n"
	fi
done
