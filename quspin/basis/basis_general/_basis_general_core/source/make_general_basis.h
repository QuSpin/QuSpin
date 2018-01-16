#ifndef _MAKE_GENERAL_BASIS_H
#define _MAKE_GENERAL_BASIS_H

#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"

template<class I,class J>
npy_intp make_basis(general_basis_core<I> *B,npy_intp MAX,npy_intp mem_MAX,I basis[],J n[]){
	npy_intp Ns = 0;
	I s = 0;
	bool unsuff_mem = false;

	while(MAX != 0){
		if(Ns>=mem_MAX){
			unsuff_mem = true;
			break;
		}
		double norm = B->check_state(s);
		J int_norm = norm;
		
		if(!std::isnan(norm) && int_norm>0 ){
			basis[Ns] = s;
			n[Ns] = norm;
			Ns++;
		}
		s++;
		MAX--;
	}

	if(unsuff_mem){
		return -1;
	}
	else{
		return Ns;
	}
}

template<class I,class J>
npy_intp make_basis_pcon(general_basis_core<I> *B,npy_intp MAX,npy_intp mem_MAX,I s,I basis[],J n[]){
	npy_intp Ns = 0;
	bool unsuff_mem = false;

	while(MAX!=0){
		if(Ns>=mem_MAX){
			unsuff_mem = true;
			break;
		}

		double norm = B->check_state(s);
		J int_norm = norm;

		if(!std::isnan(norm) && int_norm>0 ){
			basis[Ns] = s;
			n[Ns] = norm;
			Ns++;
		}
		s = B->next_state_pcon(s);
		MAX--;
	}

	if(unsuff_mem){
		return -1;
	}
	else{
		return Ns;
	}
}

#endif
