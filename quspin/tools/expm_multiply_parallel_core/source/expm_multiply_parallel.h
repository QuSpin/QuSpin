
#ifndef _EXPM_MULTIPLY_H
#define _EXPM_MULTIPLY_H

#include "complex_ops.h"
#include <stdio.h>
#include <algorithm>
#include <vector>
#include "math_functions.h"

// #include "iterators.h"
// #include <algorithm>
// #include <numeric>

#include "openmp.h"
#if defined(_OPENMP)
#include "csrmv_merge.h"
#else
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
				sum += Ax[jj] * x[Aj[jj]];
			}
			y[k] = a * sum;
		}
	}else{
		for(I k = 0; k<n; k++){
			T3 sum = 0;
			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				sum += Ax[jj] * x[Aj[jj]];
			}
			y[k] += a * sum;
		}
	}

}

template<typename I, typename T1, typename T2,typename T3>
void csr_matvec_multi(const bool overwrite_y,
                      const I n_row,
                      const npy_intp n_vecs,
                      const I Ap[],
                      const I Aj[],
                      const T1 Ax[],
                      const T2 a,
                      const T3 x[],
                            I rco[],
                            T3 vco[],
                            T3 y[])
{
	T3 * y_row = y;
	if(overwrite_y){
		for(I k = 0; k<n_row; k++, y_row+=n_vecs){
			std::fill(vco,vco+n_vecs,0);

			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				const T3 * x_row = &x[n_vecs * Aj[jj]];
				const T2 val = Ax[jj];

				for(npy_intp vec_n=0;vec_n<n_vecs;vec_n++){
					vco[vec_n] += val * x_row[vec_n];
				}
				
			}


			for(npy_intp vec_n=0;vec_n<n_vecs;vec_n++){
				y_row[vec_n] = a * vco[vec_n];
			}
			
		}
	}else{
		for(I k = 0; k<n_row; k++, y_row+=n_vecs){
			std::fill(vco,vco+n_vecs,0);

			for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
				const T3 * x_row = &x[n_vecs * Aj[jj]];
				const T2 val = Ax[jj];

				for(npy_intp vec_n=0;vec_n<n_vecs;vec_n++){
					vco[vec_n] += val * x_row[vec_n];
				}
				
			}

			for(npy_intp vec_n=0;vec_n<n_vecs;vec_n++){
				y_row[vec_n] += a * vco[vec_n];
			}
		}
	}
    
}
#endif


// #include <valarray>     // std::valarray, std::slice

template<typename I, typename T1,typename T2,typename T3>
void expm_multiply(const I n,
					const I Ap[],
					const I Aj[],
					const T1 Ax[],
					const int s,
					const int m_star,
					const T2 tol,
					const T3 mu,
					const T3 a,
			 			  T3 F[],
						  T3 work[]
			)
{

	const int num_threads = omp_get_max_threads();
	std::vector<I> rco_vec(num_threads,0);
	std::vector<T3> vco_vec(num_threads,0);
	std::vector<T2> c1_threads_vec(num_threads,0);
	std::vector<T2> c2_threads_vec(num_threads,0);
	std::vector<T2> c3_threads_vec(num_threads,0);

	T3 * B1 = work;
	T3 * B2 = work + n;
	I * rco = &rco_vec[0];
	T3 * vco = &vco_vec[0];
	T2 * c1_threads = &c1_threads_vec[0];
	T2 * c2_threads = &c2_threads_vec[0];
	T2 * c3_threads = &c3_threads_vec[0];
	bool exit_loop=false;

	#pragma omp parallel shared(exit_loop,c1_threads,c2_threads,c3_threads,F,B1,B2,rco,vco) firstprivate(num_threads)
	{
		const int tid = omp_get_thread_num();
		const I items_per_thread = (n+num_threads-1)/num_threads;
		const I begin = std::min(items_per_thread * tid, n);
		const I end = std::min(begin+items_per_thread, n);

		const T3 eta = math_functions::exp(a*(mu/T2(s)));
		T2 c1_thread=0,c2_thread=0,c3_thread=0,c1=0,c2=0,c3=0;

		c1_thread = 0;
		for(I k=begin;k<end;k++){ 
			T3 f = F[k];
			B1[k] = f;
			c1_thread = math_functions::compare_abs(c1_thread,f);
		}

		#pragma omp barrier 

		if(tid==0){
			c1 = *std::max_element(c1_threads,c1_threads+num_threads);
		}
		
		for(int i=0;i<s;i++){

			#pragma omp single
			{
				exit_loop = false;
			}

			for(int j=1;j<m_star+1 && !exit_loop;j++){

				#if defined(_OPENMP)
				csrmv_merge<I,T1,T3,T3>(true,n,Ap,Aj,Ax,a/T2(j*s),B1,rco,vco,B2); // implied barrier
				#else
				csr_matvec<I,T1,T3,T3>(true,n,Ap,Aj,Ax,a/T2(j*s),B1,rco,vco,B2);
				#endif

				c2_thread = 0; c3_thread = 0;
				const T3 b = a * mu / T2(j*s);

				for(I k=begin;k<end;k++){
					T3 b2 = B2[k] - b * B1[k];
					T3 f  = F[k] += b2;
					B1[k] = b2;
					// used cached values to compute comparisons for infinite norm
					c2_thread = math_functions::compare_abs(c2_thread,b2);
					c3_thread = math_functions::compare_abs(c3_thread,f);
				}
				c2_threads[tid] = c2_thread;
				c3_threads[tid] = c3_thread;

				#pragma omp barrier 

				if(tid==0){
					c2 = *std::max_element(c2_threads,c2_threads+num_threads);
					c3 = *std::max_element(c3_threads,c3_threads+num_threads);
					exit_loop = ((c1+c2)<=(tol*c3));
					c1 = c2;
				}

				#pragma omp barrier
			}

			c1_thread = 0;
			for(I k=begin;k<end;k++){
				T3 f = F[k] *= eta;
				B1[k] = f;
				// used cached values to compute comparisons for infinite norm
				c1_thread = math_functions::compare_abs(c1_thread,f);
			}
			c1_threads[tid] = c1_thread;

			#pragma omp barrier 

			if(tid==0){
				c1 = *std::max_element(c1_threads,c1_threads+num_threads);
			}
		}
	}
}


template<typename I, typename T1,typename T2,typename T3>
void expm_multiply_batch(const I n,
						const npy_intp n_vecs,
						const I Ap[],
						const I Aj[],
						const T1 Ax[],
						const int s,
						const int m_star,
						const T2 tol,
						const T3 mu,
						const T3 a,
				 			  T3 F[],
							  T3 work[]
			)
{

	const npy_intp N = n * n_vecs;
	const int num_threads = omp_get_max_threads();
	std::vector<I> rco_vec(n_vecs * num_threads,0);
	std::vector<T3> vco_vec(n_vecs * num_threads,0);
	std::vector<T2> c1_threads_vec(num_threads,0);
	std::vector<T2> c2_threads_vec(num_threads,0);
	std::vector<T2> c3_threads_vec(num_threads,0);

	T3 * B1 = work;
	T3 * B2 = work + N;
	I * rco = &rco_vec[0];
	T3 * vco = &vco_vec[0];
	T2 * c1_threads = &c1_threads_vec[0];
	T2 * c2_threads = &c2_threads_vec[0];
	T2 * c3_threads = &c3_threads_vec[0];
	bool exit_loop=false;

	#pragma omp parallel shared(exit_loop,c1_threads,c2_threads,c3_threads,F,B1,B2,rco,vco) firstprivate(num_threads)
	{
		const int tid = omp_get_thread_num();

		const npy_intp items_per_thread = (N+num_threads-1)/num_threads;
		const npy_intp begin = std::min(items_per_thread * tid, N);
		const npy_intp end = std::min(begin+items_per_thread, N);
		const T3 eta = math_functions::exp(a*(mu/T2(s)));
		T2 c1_thread=0,c2_thread=0,c3_thread=0,c1=0,c2=0,c3=0;

		c1_thread = 0;
		for(I k=begin;k<end;k++){ 
			T3 f = F[k];
			B1[k] = f;
			c1_thread = math_functions::compare_abs(c1_thread,f);
		}

		#pragma omp barrier 

		if(tid==0){
			c1 = *std::max_element(c1_threads,c1_threads+num_threads);
		}
		
		for(int i=0;i<s;i++){

			#pragma omp single
			{
				exit_loop = false;
			}

			for(int j=1;j<m_star+1 && !exit_loop;j++){

				#if defined(_OPENMP)
				csrmv_merge_multi<I,T1,T3,T3>(true,n,n_vecs,Ap,Aj,Ax,a/T2(j*s),B1,rco,vco,B2); // implied barrier
				#else
				csr_matvec_multi<I,T1,T3,T3>(true,n,n_vecs,Ap,Aj,Ax,a/T2(j*s),B1,rco,vco,B2);
				#endif

				c2_thread = 0; c3_thread = 0;
				const T3 b = a * mu / T2(j*s);

				for(I k=begin;k<end;k++){
					T3 b2 = B2[k] - b * B1[k];
					T3 f  = F[k] += b2;
					B1[k] = b2;
					// used cached values to compute comparisons for infinite norm
					c2_thread = math_functions::compare_abs(c2_thread,b2);
					c3_thread = math_functions::compare_abs(c3_thread,f);
				}
				c2_threads[tid] = c2_thread;
				c3_threads[tid] = c3_thread;

				#pragma omp barrier 

				if(tid==0){
					c2 = *std::max_element(c2_threads,c2_threads+num_threads);
					c3 = *std::max_element(c3_threads,c3_threads+num_threads);
					exit_loop = ((c1+c2)<=(tol*c3));
					c1 = c2;
				}

				#pragma omp barrier
			}

			c1_thread = 0;
			for(I k=begin;k<end;k++){
				T3 f = F[k] *= eta;
				B1[k] = f;
				// used cached values to compute comparisons for infinite norm
				c1_thread = math_functions::compare_abs(c1_thread,f);
			}
			c1_threads[tid] = c1_thread;

			#pragma omp barrier 

			if(tid==0){
				c1 = *std::max_element(c1_threads,c1_threads+num_threads);
			}
		}
	}
}


#endif
