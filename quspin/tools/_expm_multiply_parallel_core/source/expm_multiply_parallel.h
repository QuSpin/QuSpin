
#ifndef _EXPM_MULTIPLY_H
#define _EXPM_MULTIPLY_H

#include <omp.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <iomanip>




template <class I, class T>
T csr_trace(const I n_row,
            const I n_col, 
	        const I Ap[], 
	        const I Aj[], 
	        const T Ax[])
{

    T trace = 0;
    const I N = std::min(n_row, n_col);

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
	return fmax(norm,x*x);
}

template<typename T>
T inline my_max(T norm,std::complex<T> x){
	T re = x.real();
	T im = x.imag();
	return fmax(norm,re*re+im*im);
}

// template<typename I,typename T>
// T inf_norm(I N,std::complex<T> x[]){
// 	T norm = 0;

// 	// #pragma omp for reduction(max:norm)
// 	for(I i=0;i<N;i++){
// 		T re = x[i].real();
// 		T im = x[i].imag();
// 		norm = fmax(norm,re*re+im*im);
// 	}

// 	return std::sqrt(norm);

// }

// template<typename I,typename T>
// T inf_norm(I N,T x[]){
// 	T norm = 0;

// 	// #pragma omp for reduction(max:norm)
// 	for(I i=0;i<N;i++){
// 		T xi = x[i];
// 		norm = fmax(norm,(xi>0 ? xi : -xi ));
// 	}

// 	return norm;
// }



template<typename I, typename T1,typename T2,typename T3>
void _expm_multiply(const I n_row,
				  const I n_col,
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
	
	
	#pragma omp parallel shared(c1,c2,c3,flag,F,B1,B2)
	{

		int CHUNK = n_row/omp_get_num_threads();
		T3 eta = std::exp(a*mu/T2(s));

		#pragma omp for schedule(static,CHUNK)
		for(I k=0;k<n_row;k++){ 
			B1[k] = F[k];
			B2[k] = 0;
		}

		for(int i=0;i<s;i++){

			#pragma omp for schedule(static,CHUNK) reduction(max:c1)
			for(I i=0;i<n_row;i++){
				c1 = my_max(c1,B1[i]);
			}

			#pragma omp single 
			c1 = std::sqrt(c1);

			for(int j=1;j<m_star+1 && !flag;j++){

				#pragma omp for schedule(static,CHUNK)
			    for(I k = 0; k<n_row; k++){
					T3 sum = 0;
			        for(I jj = Ap[k]; jj < Ap[k+1]; jj++){
			            sum += Ax[jj] * B1[Aj[jj]];
			        }
			        sum *= a/T2(j*s);
			        B2[k] = sum;
			        F[k] += sum;
			    }

				#pragma omp for schedule(static,CHUNK)
				for(I k=0;k<n_row;k++){
					B1[k] = B2[k];
				}

				#pragma omp for schedule(static,CHUNK) reduction(max:c2)
				for(I i=0;i<n_row;i++){
					c2 = my_max(c2,B2[i]);
				}

				#pragma omp for schedule(static,CHUNK) reduction(max:c3)
				for(I i=0;i<n_row;i++){
					c3 = my_max(c3,F[i]);
				}

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
			for(I k=0;k<n_row;k++){
				F[k] *= eta;
				B1[k] = F[k];
			}
		}
	}
}



#endif
