
#ifndef _EXPM_MULTIPLY_H
#define _EXPM_MULTIPLY_H

#include "openmp.h"
#include "csr_matvec.h"
#include <cmath>
#include <algorithm>
#include <complex>
#include <iostream>

template <class I, class T>
T csr_trace(const I n,
			const I n_col, 
			const I Ap[], 
			const I Aj[], 
			const T Ax[])
{

	T trace = 0;
	const I N = (n<n_col?n_col:n);

	for(I i = 0; i < N; i++){
		const I row_start = Ap[i];
		const I row_end   = Ap[i+1];

		T diag = 0;
		for(I jj = row_start; jj < row_end; jj++){
			if (Aj[jj] == i)
				diag += Ax[jj];
		}

		trace += diag;
	}
	return trace;
}

template<typename T,typename I>
T inf_norm_chunk(T * arr,const I begin,const I end){
	T max = 0;
	for(I i=begin;i<end;i++){
		T a = arr[i]*arr[i];
		max = std::max(max,a);
	}
	return std::sqrt(max);
}

template<typename T,typename I>
T inf_norm_chunk(std::complex<T> * arr,const I begin,const I end){
	T max = 0;
	for(I i=begin;i<end;i++){
		T re = arr[i].real();
		T im = arr[i].imag();
		T a = re*re+im*im;
		max = std::max(max,a);
	}
	return std::sqrt(max);
}


template<typename I, typename T1,typename T2,typename T3>
void _expm_multiply(const I n,
					const I Ap[],
					const I Aj[],
					const T1 Ax[],
					const int s,
					const int m_star,
					const T2 tol,
					const T1 mu,
					const T3 a,
			 			  T3 F[],
						  T3 B1[], 
						  T3 B2[]
			)
{

	T2  c1,c2,c3;
	bool flag=false;
	I rco[128];
	T3 vco[128];
	
	#pragma omp parallel shared(c1,c2,c3,flag,F,B1,B2,rco,vco)
	{
		const int nthread = omp_get_num_threads();
		const int threadn = omp_get_thread_num();
		const I items_per_thread = n/nthread;
		const I begin = items_per_thread * threadn;
		I end0 = items_per_thread * ( threadn + 1 );
		if(threadn == nthread-1){
			end0 += n%nthread;
		}
		const I end = end0;

		const T3 eta = std::exp(a*mu/T2(s));

		for(I k=begin;k<end;k++){ 
			B1[k] = F[k];
			B2[k] = 0;
		}
		for(int i=0;i<s;i++){

			T2 c1_thread = inf_norm_chunk(B1,begin,end);

			#pragma omp single
			{
				c1 = 0;
				flag = false;
			}

			#pragma omp critical
			{
				c1 = std::max(c1,c1_thread);
			}	

			for(int j=1;j<m_star+1 && !flag;j++){

				csr_matvec(true,n,Ap,Aj,Ax,a/T2(j*s),B1,rco,vco,B2);

				for(I k=begin;k<end;k++){
					F[k] += B1[k] = B2[k];
				}

				T2 c2_thread = inf_norm_chunk(B2,begin,end);
				T2 c3_thread = inf_norm_chunk(F,begin,end);

				#pragma omp single
				{
					c2 = c3 = 0;
				}

				#pragma omp critical
				{
					c2 = std::max(c2,c2_thread);
					c3 = std::max(c3,c3_thread);
				}	

				#pragma omp barrier

				#pragma omp single
				{
					if((c1+c2)<=(tol*c3)){
						flag=true;
					}
					c1 = c2;
				}

			}

			for(I k=begin;k<end;k++){
				F[k] *= eta;
				B1[k] = F[k];
			}

			#pragma omp barrier
		}
	}
}

#endif
