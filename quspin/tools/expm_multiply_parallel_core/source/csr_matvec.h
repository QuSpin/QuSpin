#ifndef __CSR_MATVEC_H__
#define __CSR_MATVEC_H__


#if defined(_OPENMP)

#include <omp.h>
#include <algorithm>
#include <complex>

// See work my Merrill et. al. (http://ieeexplore.ieee.org/abstract/document/7877136/) for original work and implementation.
// This code contains modified versions of algorithms 2 and 3.

#pragma message ("This header file is deprecated, this should only be used for csr_matvec_wrapper.pyx")

template<class I>
class CountingInputIterator{
	const I init;
public:
	CountingInputIterator(I _init) : init(_init) {}
	I operator[](I i){return init+i;}
};

template<class I>
struct CoordinateT{
	I x,y;
	CoordinateT(I _x,I _y) : x(_x), y(_y) {}
};


template<class I,class AIteratorT,class BIteratorT>
CoordinateT<I> MergePathSearch(I diagonal, I a_len, I b_len, AIteratorT a, BIteratorT b)
{
	// Diagonal search range (in x coordinate space)
	I zero = 0;
	I x_min = std::max(diagonal - b_len, zero);
	I x_max = std::min(diagonal, a_len);
	// 2D binary-search along the diagonal search range
	while (x_min < x_max) {
		I pivot = (x_min + x_max) >> 1;
		if (a[pivot] <= b[diagonal - pivot - 1]) {
			// Keep top-right half of diagonal range
			x_min = pivot + 1;
		} else {
			// Keep bottom-left half of diagonal range
			x_max = pivot;
		}
	}
	return CoordinateT<I>(
	std::min(x_min, a_len), // x coordinate in A
	diagonal - x_min); // y coordinate in B
}

template<class I,class T1,class T2,class T3>
void csrmv_merge(const bool overwrite_y,
				const I num_rows,
				const I row_offsets[],
				const I column_indices[],
				const T1 values[],
				const T2 alpha,
				const T3 x[], 
					  I row_carry_out[],
					  T3 value_carry_out[],
					  T3 y[])
{

	const I* row_end_offsets = row_offsets + 1; // Merge list A: row end-offsets
	const I num_nonzeros = row_offsets[num_rows];
	int num_threads = omp_get_num_threads();
	CountingInputIterator<I> nz_indices(0); // Merge list B: Natural numbers(NZ indices)
	I num_merge_items = num_rows + num_nonzeros; // Merge path total length
	I items_per_thread = (num_merge_items + num_threads - 1) / num_threads; // Merge items per thread
	T3 alpha_cast = T3(alpha);
	if(overwrite_y){
		#pragma omp for schedule(static)
		for(I i=0;i<num_rows;i++){
			y[i] = T3(0);
		}
	}
	// Spawn parallel threads
	#pragma omp for schedule(static,1)
	for (int tid = 0; tid < num_threads; tid++)
	{
		// Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for each thread
		I diagonal = std::min(items_per_thread * tid, num_merge_items);
		I diagonal_end = std::min(diagonal + items_per_thread, num_merge_items);
		CoordinateT<I> thread_coord = MergePathSearch(diagonal, num_rows, num_nonzeros, row_end_offsets, nz_indices);
		CoordinateT<I> thread_coord_end = MergePathSearch(diagonal_end, num_rows, num_nonzeros,row_end_offsets, nz_indices);

		// Consume merge items, whole rows first
		T3 running_total = 0.0;
		for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x)
		{
			for (; thread_coord.y < row_end_offsets[thread_coord.x]; ++thread_coord.y)
			running_total += T3(values[thread_coord.y]) * x[column_indices[thread_coord.y]];

			y[thread_coord.x] += alpha_cast*running_total;
			running_total = 0.0;
		}

		// Consume partial portion of thread's last row
		for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y)
			running_total += T3(values[thread_coord.y]) * x[column_indices[thread_coord.y]];

		// Save carry-outs
		row_carry_out[tid] = thread_coord_end.x;
		value_carry_out[tid] = running_total;
	}

	// Carry-out fix-up (rows spanning multiple threads)
	#pragma omp single
	{
		for (int tid = 0; tid < num_threads - 1; ++tid)
		if (row_carry_out[tid] < num_rows)
		y[row_carry_out[tid]] += alpha_cast*value_carry_out[tid];
	}

}


template<typename I, typename T1,typename T2,typename T3>
void inline csr_matvec(const bool overwrite_y,
						const I n,
						const I Ap[],
						const I Aj[],
						const T1 Ax[],
						const T2 a,
						const T3 x[],
							  I rco[],
							  T3 vco[],
							  T3 y[])
{
	csrmv_merge(overwrite_y,n,Ap,Aj,Ax,a,x,rco,vco,y);
}
#else

#include <complex>
inline int omp_get_max_threads(void){return 1;}
inline int omp_get_num_threads(void){return 1;}
inline int omp_get_thread_num(void){return 0;}

template<typename I, typename T1,typename T2,typename T3>
void csr_matvec(const bool overwrite_y,
				const I n,
				const I Ap[],
				const I Aj[],
				const T1 Ax[],
				const T2 a,
				const T3 x[],
					  I rco[],
					  T3 vco[],
					  T3 y[])
{
	if(overwrite_y){
		for(I k = 0; k<n; k++){
			T3 sum = 0;
			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				sum += T3(Ax[jj]) * x[Aj[jj]];
			}
			y[k] = T3(a)*sum;
		}
	}else{
		for(I k = 0; k<n; k++){
			T3 sum = 0;
			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				sum += T3(Ax[jj]) * x[Aj[jj]];
			}
			y[k] += T3(a)*sum;
		}
	}

}
#endif

#endif