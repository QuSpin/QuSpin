
#ifndef _EXPM_MULTIPLY_H
#define _EXPM_MULTIPLY_H

#include "complex_ops.h"
#include <stdio.h>

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
#endif

#include <algorithm>
#include <vector>
#include "math_functions.h"



template<typename I, typename T1,typename T2,typename T3>
void expm_multiply(const I n,
					const I Ap[],
					const I Aj[],
					const T1 Ax[],
					const int s,
					const int m_star,
					const T2 tol,
					const T1 mu,
					const T3 a,
			 			  T3 F[],
						  T3 work[]
			)
{

	T2  c1,c2,c3;
	bool flag=false;

	const int nthread = omp_get_max_threads();
	std::vector<I> rco_vec(nthread);
	std::vector<T3> vco_vec(nthread);

	T3 * B1 = work;
	T3 * B2 = work + n;
	I * rco = &rco_vec[0];
	T3 * vco = &vco_vec[0];
	
	#pragma omp parallel shared(c1,c2,c3,flag,F,B1,B2,rco,vco) firstprivate(nthread)
	{
		const int threadn = omp_get_thread_num();
		const I items_per_thread = n/nthread;
		const I begin = items_per_thread * threadn;
		I end0 = items_per_thread * ( threadn + 1 );
		if(threadn == nthread-1){
			end0 = n;
		}
		const I end = end0;

		const T3 eta = math_functions::exp(a*mu/T2(s));

		for(I k=begin;k<end;k++){ 
			B1[k] = F[k];
		}
		for(int i=0;i<s;i++){

			T2 c1_thread = math_functions::inf_norm(B1,begin,end);

			#pragma omp single // implied barrier
			{
				c1 = 0;
				flag = false;
			}

			#pragma omp critical 
			{
				c1 = std::max(c1,c1_thread);
			}

			for(int j=1;j<m_star+1 && !flag;j++){

				#if defined(_OPENMP)
				csrmv_merge<I,T1,T3,T3>(true,n,Ap,Aj,Ax,a/T2(j*s),B1,rco,vco,B2); // implied barrier
				#else
				csr_matvec<I,T1,T3,T3>(true,n,Ap,Aj,Ax,a/T2(j*s),B1,rco,vco,B2);
				#endif

				for(I k=begin;k<end;k++){
					F[k] += B1[k] = B2[k];
				}

				T2 c2_thread = math_functions::inf_norm(B2,begin,end);
				T2 c3_thread = math_functions::inf_norm(F,begin,end);

				#pragma omp single // implied barrier
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
		}
	}
}

#endif
