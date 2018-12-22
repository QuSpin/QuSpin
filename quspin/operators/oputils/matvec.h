#ifndef __MATVEC_H__
#define __MATVEC_H__


#include <algorithm>

// y += a*x
template <typename I, typename T>
void axpy(const I n, const T a, const T * x, T * y){
	for(I i = 0; i < n; i++){
		y[i] += a * x[i];
	}
}

#if defined(_OPENMP)
#include "csrmv_merge.h"
template<typename I, typename T1,typename T2>
void inline csr_matvec(const bool overwrite_y,
						const I n,
						const I Ap[],
						const I Aj[],
						const T1 Ax[],
						const T1 a,
						const T2 x[],
							  I rco[],
							  T2 vco[],
							  T2 y[])
{
	csrmv_merge(overwrite_y,n,Ap,Aj,Ax,a,x,rco,vco,y);
}



template <typename I, typename T1, typename T2>
void dia_matvec(const bool overwrite_y,
				const I n_row,
                const I n_col,
                const I n_diags,
                const I L,
	            const I offsets[], 
	            const T1 diags[], 
	            const T1 a,
	            const T2 x[],
	                  T2 y[])
{

	const int nthread = omp_get_num_threads();
	const int threadn = omp_get_thread_num();

	if(overwrite_y){
		const I items_per_thread = n_row/nthread;
		const I begin = items_per_thread * threadn;
		I end = items_per_thread * ( threadn + 1 );

		if(threadn == nthread-1){
			end += n_row%nthread;
		}

		for(I i = begin; i < end; i++){
			y[i] = 0;
		}
	}
	
	#pragma omp barrier

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T1 * diag = diags + (npy_intp)i*L + j_start;
        const T2 * x_row = x + j_start;
              T2 * y_row = y + i_start;

        // calculate loop chunks
		const I items_per_thread = N/nthread;
		const I begin = items_per_thread * threadn;
		I end = items_per_thread * ( threadn + 1 );

		if(threadn == nthread-1){
			end += N%nthread;
		}

        for(I n = begin; n < end; n++){
            y_row[n] += (T2)(a * diag[n]) * x_row[n]; 
        }

        #pragma omp barrier

    }
}


template <typename I, typename T1, typename T2>
void dia_matvecs(const bool overwrite_y,
				const I n_row,
                const I n_col,
                const I n_vecs,
                const I n_diags,
                const I L,
	            const I offsets[], 
	            const T1 diags[], 
	            const T1 a,
	            const T2 x[],
	                  T2 y[])
{
	const int nthread = omp_get_num_threads();
	const int threadn = omp_get_thread_num();

	if(overwrite_y){
		const npy_intp n = (npy_intp)n_row * n_vecs;
		const npy_intp items_per_thread = n/nthread;
		const npy_intp begin = items_per_thread * threadn;
		npy_intp end = items_per_thread * ( threadn + 1 );

		if(threadn == nthread-1){
			end += n%nthread;
		}

		for(npy_intp i = begin; i < end; i++){
			y[i] = 0;
		}
	}

	#pragma omp barrier

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T1 * diag = diags + (npy_intp)i*L + j_start;
        const T2 * x_row = x + j_start * n_vecs;
              T2 * y_row = y + i_start * n_vecs;

		const I items_per_thread = N/nthread;
		const I begin = items_per_thread * threadn;
		I end = items_per_thread * ( threadn + 1 );

		if(threadn == nthread-1){
			end += N%nthread;
		}

        for(I n = begin; n < end; n++){
            axpy(n_vecs,(T2)(a * diag[n]), x_row + (npy_intp)n_vecs * n, y_row + (npy_intp)n_vecs * n);
        }

        #pragma omp barrier

    }
}



#else

template<typename I, typename T1,typename T2>
void csr_matvec(const bool overwrite_y,
				const I n,
				const I Ap[],
				const I Aj[],
				const T1 Ax[],
				const T1 a,
				const T2 x[],
					  I rco[],
					  T2 vco[],
					  T2 y[])
{
	const T2 a_cast = a;
	if(overwrite_y){
		for(I k = 0; k<n; k++){
			T2 sum = 0;
			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				sum += (T2)Ax[jj] * x[Aj[jj]];
			}
			y[k] = a_cast * sum;
		}
	}else{
		for(I k = 0; k<n; k++){
			T2 sum = 0;
			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				sum += (T2)Ax[jj] * x[Aj[jj]];
			}
			y[k] += a_cast * sum;
		}
	}

}



template <typename I, typename T1, typename T2>
void dia_matvec(const bool overwrite_y,
				const I n_row,
                const I n_col,
                const I n_diags,
                const I L,
	            const I offsets[], 
	            const T1 diags[], 
	            const T1 a,
	            const T2 x[],
	                  T2 y[])
{

	if(overwrite_y){
		for(I i = 0; i < n_row; i++){
			y[i] = 0;
		}
	}

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T1 * diag = diags + (npy_intp)i*L + j_start;
        const T2 * x_row = x + j_start;
              T2 * y_row = y + i_start;

        for(I n = 0; n < N; n++){
            y_row[n] += (T2)(a * diag[n]) * x_row[n]; 
        }
    }
}


template <typename I, typename T1, typename T2>
void dia_matvecs(const bool overwrite_y,
				const I n_row,
                const I n_col,
                const I n_vecs,
                const I n_diags,
                const I L,
	            const I offsets[], 
	            const T1 diags[], 
	            const T1 a,
	            const T2 x[],
	                  T2 y[])
{
	if(overwrite_y){
		const npy_intp n = (npy_intp)n_row * n_vecs;

		for(npy_intp i = 0; i < n; i++){
			y[i] = 0;
		}
	}

    for(I i = 0; i < n_diags; i++){
        const I k = offsets[i];  //diagonal offset

        const I i_start = std::max<I>(0,-k);
        const I j_start = std::max<I>(0, k);
        const I j_end   = std::min<I>(std::min<I>(n_row + k, n_col),L);

        const I N = j_end - j_start;  //number of elements to process

        const T1 * diag = diags + (npy_intp)i*L + j_start;
        const T2 * x_row = x + (npy_intp)j_start * n_vecs;
              T2 * y_row = y + (npy_intp)i_start * n_vecs;

        for(I n = 0; n < N; n++){
            axpy(n_vecs,(T2)(a * diag[n]), x_row + (npy_intp)n_vecs * n, y_row + (npy_intp)n_vecs * n);
        }
    }
}


#endif




template<typename I, typename T1,typename T2>
void csr_matvecs(const bool overwrite_y,
				const I n,
				const I n_vecs,
				const I Ap[],
				const I Aj[],
				const T1 Ax[],
				const T1 a,
				const T2 x[],
					  T2 y[])
{
	
	if(overwrite_y){
		for(I k = 0; k<n; k++){

			T2 * y_row = y + (npy_intp)n_vecs * k;
			for(I jj=0;jj<n_vecs;jj++){y_row[jj] = 0;}

			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				const I j = Aj[jj];
				const T2 ax = a * Ax[jj];
				const T2 * x_row = x + (npy_intp)n_vecs * j;
				axpy(n_vecs, ax, x_row, y_row);

			}
		}
	}else{
		for(I k = 0; k<n; k++){
			T2 * y_row = y + (npy_intp)n_vecs * k;

			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				const I j = Aj[jj];
				const T2 ax = a * Ax[jj];
				const T2 * x_row = x + (npy_intp)n_vecs * j;
				axpy(n_vecs, ax, x_row, y_row);

			}
		}
	}

}


template<typename I, typename T1,typename T2>
void csc_matvec(const bool overwrite_y,
						const I n_row,
						const I n_col,
						const I Ap[],
						const I Ai[],
						const T1 Ax[],
						const T1 a,
						const T2 x[],
							  T2 y[])
{
	if(overwrite_y){
		for(I j = 0; j < n_row; j++){
			y[j] = 0;
		}
	}

	

	for(I j = 0; j < n_col; j++){
		I col_start = Ap[j];
		I col_end   = Ap[j+1];

		for(I ii = col_start; ii < col_end; ii++){
			const I i = Ai[ii];
			y[i] += (T2)(a * Ax[ii]) * x[j];;
		}
	}
}


template<typename I, typename T1,typename T2>
void csc_matvecs(const bool overwrite_y,
						const I n_row,
						const I n_col,
						const I n_vecs,
						const I Ap[],
						const I Ai[],
						const T1 Ax[],
						const T1 a,
						const T2 x[],
							  T2 y[])
{
	if(overwrite_y){
		for(I i = 0; i < n_row*n_vecs; i++){
			y[i] = 0;
		}
	}

	

	for(I j = 0; j < n_col; j++){
		I col_start = Ap[j];
		I col_end   = Ap[j+1];

		for(I ii = col_start; ii < col_end; ii++){
			I i	= Ai[ii];
			const T2 ax = a * Ax[ii];

			 axpy(n_vecs, ax, x + (npy_intp)n_vecs * j, y + (npy_intp)n_vecs * i);
		}
	}
}




#endif