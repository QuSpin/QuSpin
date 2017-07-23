#ifndef _MAKE_GENERAL_BASIS_H
#define _MAKE_GENERAL_BASIS_H

#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include <iostream>

#include <omp.h>

template<class I,class J>
npy_intp make_basis(general_basis_core<I> *B,npy_intp MAX,I basis[],J n[]){
	npy_intp ii = 0;
	#pragma omp parallel for schedule(static,1)
	for(npy_intp s=0;s<MAX;s++){
		if(B->check_state(s)){
			J nn = B->get_norm(s);
			if(nn>0){
				#pragma omp critical
				{
					basis[ii] = s;
					n[ii] = nn;
					ii++;
				}
			}
		}
	}
	return ii;
}

template<class I,class J>
npy_intp make_basis_pcon(general_basis_core<I> *B,npy_intp MAX,I s,I basis[],J n[]){
	npy_intp ii = 0;
	#pragma omp parallel firstprivate(s) shared(ii)
	{
		int n_threads = omp_get_num_threads();
		int id = omp_get_thread_num();

		for(int j=0;j<id;j++){
			s = B->next_state_pcon(s);
		}
		
		for(npy_intp i=id;i<MAX;i+=n_threads){
			if(B->check_state(s)){
				J nn = B->get_norm(s);
				if(nn>0){
					#pragma omp critical
					{
						basis[ii] = s;
						n[ii] = nn;
						ii++;
					}
				}
			}

			for(int j=0;j<n_threads;j++){
				s = B->next_state_pcon(s);
			}
		}

	}
	return ii;
}

#endif
