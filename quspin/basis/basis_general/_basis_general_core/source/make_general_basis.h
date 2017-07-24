#ifndef _MAKE_GENERAL_BASIS_H
#define _MAKE_GENERAL_BASIS_H

#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include <iostream>

#include <omp.h>

template<class I,class J>
npy_intp make_basis(general_basis_core<I> *B,npy_intp MAX,I basis[],J n[]){
	npy_intp Ns = 0;
	#pragma omp parallel shared(Ns)
	{
		int n_th = omp_get_num_threads();
		int id = omp_get_thread_num();
		npy_intp ii = id;

		for(npy_intp s=id;s<MAX;s+=n_th){
			if(B->check_state(s)){
				J nn = B->get_norm(s);
				if(nn>0){
					basis[ii] = s;
					n[ii] = nn;
					ii+=n_th;

				}
			}
		}
		#pragma omp critical
		Ns += ii/n_th;

	}
	return Ns;
}

template<class I,class J>
npy_intp make_basis_pcon(general_basis_core<I> *B,npy_intp MAX,I s,I basis[],J n[]){
	npy_intp Ns = 0;
	#pragma omp parallel firstprivate(s) shared(Ns)
	{
		int n_th = omp_get_num_threads();
		int id = omp_get_thread_num();
		npy_intp ii = id;

		for(int j=0;j<id;j++){
			s = B->next_state_pcon(s);
		}
		
		for(npy_intp i=id;i<MAX;i+=n_th){
			if(B->check_state(s)){
				J nn = B->get_norm(s);
				if(nn>0){
					basis[ii] = s;
					n[ii] = nn;
					ii+=n_th;
				}
			}

			for(int j=0;j<n_th;j++){
				s = B->next_state_pcon(s);
			}
		}

		#pragma omp critical
		Ns += ii/n_th;
	}

	return Ns;
}

#endif
