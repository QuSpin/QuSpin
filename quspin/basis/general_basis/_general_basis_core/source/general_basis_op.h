#ifndef _GENERAL_BASIS_OP_H
#define _GENERAL_BASIS_OP_H

#include <complex>
#include <iostream>
#include <iomanip>
#include <limits>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"


template<class K,class I>
K binary_search(const K N,const I A[],const I s){
	K b,bmin,bmax;
	bmin = 0;
	bmax = N-1;
	while(bmin<=bmax){
		b = (bmax+bmin)/2;
		I a = A[b];
		if(s==a){
			return b;
		}
		else if(s<A[b]){
			bmin = b + 1;
		}
		else{
			bmax = b - 1;
		}
	}
	return -1;
}

template<class T>
int inline check_imag(std::complex<double> m,std::complex<T> *M){
	M[0].real(m.real());
	M[0].imag(m.imag());
	return 0;
}

template<class T>
int inline check_imag(std::complex<double> m,T *M){
	if(std::abs(m.imag())>1.1e-15){
		return 1;
	}
	else{
		M[0] = m.real();
		return 0;
	}
}



template<class I, class J, class K, class T>
int general_op(general_basis_core<I> *B,
						  const int n_op,
						  const unsigned char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const K Ns,
						  const I basis[],
						  const J n[],
						  		K row[],
						  		K col[],
						  		T M[]
						  )
{
	const int nt = B->get_nt();
	// const int N = B->get_N();
	int err = 0;
	int g[128],gg[128];

	#pragma omp parallel for schedule(static,1) private(g,gg)
	for(npy_intp i=0;i<Ns;i++){
		if(err != 0){
			continue;
		}

		I r = basis[i];
		std::complex<double> m = A;
		int local_err = B->op(r,m,n_op,opstr,indx);

		if(local_err == 0){
			
			I rr = B->ref_state(r,g,gg);
			// for(int ee=0;ee<N;ee++){std::cout << ((r>>ee)&1);}
			// std::cout << std::setw(5);
			// for(int ee=0;ee<N;ee++){std::cout << ((rr>>ee)&1);}
			// std::cout << std::setw(5);
			// for(int ee=0;ee<nt;ee++){std::cout << g[ee] << std::setw(5);}
			// std::cout << std::endl;
		
			K j = binary_search(Ns,basis,rr);
			if(j >= 0){
				for(int k=0;k<nt;k++){
					double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
					m *= std::exp(std::complex<double>(0,q));
				}
				m *= std::sqrt(double(n[j])/double(n[i]));
				local_err = check_imag(m,&M[i]);
				col[i]=i;
				row[i]=j;
			}
			else{
				col[i] = i;
				row[i] = i;
				M[i] = std::numeric_limits<T>::quiet_NaN();
			}
		}

		if(local_err != 0){
			#pragma omp critical
			err = local_err;
		}
	}

	return err;
}







#endif
