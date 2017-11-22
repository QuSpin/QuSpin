
#ifndef _EXPM_MULTIPLY_H
#define _EXPM_MULTIPLY_H

#include "openmp.h"
#include "csr_matvec.h"
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>




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



template<typename T>
T inline my_max(T norm,T x){
	T a = x*x;
	// return (a<norm?norm:a);
	return std::max(norm,a);
}

template<typename T>
T inline my_max(T norm,std::complex<T> x){
	T re = x.real();
	T im = x.imag();
	T a = re*re+im*im;
	// return (a<norm?norm:a);
	return std::max(norm,a);
}


template<typename T,typename I>
T inf_norm_chunk(T * arr, I n,int nthread,int threadn){
	T max = 0;
	for(I i=threadn;i<n;i+=nthread){
		T a = arr[i]*arr[i];
		max = (a<max?max:a);
	}
	return std::sqrt(max);
}

template<typename T,typename I>
T inf_norm_chunk(std::complex<T> * arr, I n,int nthread,int threadn){
	T max = 0;
	for(I i=threadn;i<n;i+=nthread){
		T re = arr[i].real();
		T im = arr[i].imag();
		T a = re*re+im*im;
		max = (a<max?max:a);
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
		int nthread = omp_get_num_threads();
		int threadn = omp_get_thread_num();
		int CHUNK = n/nthread;
		T3 eta = std::exp(a*mu/T2(s));

		#pragma omp for schedule(static,CHUNK)
		for(I k=0;k<n;k++){ 
			B1[k] = F[k];
			B2[k] = 0;
		}
		for(int i=0;i<s;i++){

			T2 c1_thread = inf_norm_chunk(B1,n,nthread,threadn);

			#pragma omp single
			{
				c1 = 0;
				flag = false;
			}

			#pragma omp critical
			{
				c1 = (c1_thread<c1?c1:c1_thread);
			}	

			for(int j=1;j<m_star+1 && !flag;j++){

				csr_matvec(true,n,Ap,Aj,Ax,a/T2(j*s),B1,rco,vco,B2);

				#pragma omp for schedule(static,CHUNK)
				for(I k=0;k<n;k++){
					F[k] += B2[k];
					B1[k] = B2[k];
				}

				T2 c2_thread = inf_norm_chunk(B2,n,nthread,threadn);
				T2 c3_thread = inf_norm_chunk(F,n,nthread,threadn);

				#pragma omp single
				{
					c2 = 0;
					c3 = 0;
				}

				#pragma omp critical
				{
					c2 = (c2_thread<c2?c2:c2_thread);
					c3 = (c3_thread<c3?c3:c3_thread);
				}	

				#pragma omp barrier

				#pragma omp single
				{
					c2 = std::sqrt(c2);
					c3 = std::sqrt(c3);

					if((c1+c2)<=(tol*c3)){
						flag=true;
					}
					c1 = c2;
				}

			}

			#pragma omp for schedule(static,CHUNK)
			for(I k=0;k<n;k++){
				F[k] *= eta;
				B1[k] = F[k];
			}
		}
	}
}



#endif
