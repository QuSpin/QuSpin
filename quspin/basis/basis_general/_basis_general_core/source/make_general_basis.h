#ifndef _MAKE_GENERAL_BASIS_H
#define _MAKE_GENERAL_BASIS_H

#include <iostream>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "openmp.h"
#include "misc.h"
#include <cmath>
#include <cfloat>
#include <vector>
#include <utility>
#include <algorithm>
#include <functional>



namespace basis_general {


template<class I,class P>
int general_make_basis_blocks(general_basis_core<I,P> *B,const int N_p,const npy_intp Ns,const I basis[],npy_intp basis_begin[],npy_intp basis_end[]){


	if(N_p==0){
		basis_begin[0] = 0;
		basis_end[0] = Ns;
		return 0;
	}


	npy_intp begin = 0;
	npy_intp end   = 0;

	npy_intp s_p = B->get_prefix(basis[0],N_p);
	npy_intp s_p_next = 0;

	if(s_p < 0){
		return -1;
	}

	for(npy_intp i=0;i<Ns;i++){
		s_p_next = B->get_prefix(basis[i],N_p);
		if(s_p_next < 0){
			return -1;
		}
		else if(s_p_next == s_p){
			end++;
		}
		else{
			basis_begin[s_p] = begin;
			basis_end[s_p] = end;
			begin = end++;
			s_p = s_p_next;
		}
	}
	
	basis_begin[s_p_next] = begin;
	basis_end[s_p_next] = end;


	return 0;
}

template<class I,class J,class P=signed char>
npy_intp make_basis_sequential(general_basis_core<I,P> *B,npy_intp MAX,npy_intp mem_MAX,I basis[],J n[]){
	npy_intp Ns = 0;
	I s = 0;
	bool insuff_mem = false;

	while(MAX != 0){
		if(Ns>=mem_MAX){
			insuff_mem = true;
			break;
		}
		double norm = B->check_state(s);
		npy_intp int_norm = norm;
		
		if(!check_nan(norm) && int_norm>0 ){
			basis[Ns] = s;
			n[Ns] = norm;
			Ns++;
		}
		s++;
		MAX--;
	}

	if(insuff_mem){
		return -1;
	}
	else{
		std::reverse(basis,basis+Ns);
		std::reverse(n,n+Ns);
		return Ns;
	}
}


template<class I,class J,class P=signed char>
npy_intp make_basis_pcon_sequential(general_basis_core<I,P> *B,npy_intp MAX,npy_intp mem_MAX,I s,I basis[],J n[]){
	npy_intp Ns = 0;
	I nns = 0; // number of next_state calls
	bool insuff_mem = false;

	while(MAX!=0){
		if(Ns>=mem_MAX){
			insuff_mem = true;
			break;
		}
		double norm = B->check_state(s);
		npy_intp int_norm = norm;

		if(!check_nan(norm) && int_norm>0 ){
			basis[Ns] = s;
			n[Ns] = norm;
			Ns++;
		}
		s = B->next_state_pcon(s,nns++);
		MAX--;
	}



	if(insuff_mem){
		return -1;
	}
	else{
		std::reverse(basis,basis+Ns);
		std::reverse(n,n+Ns);
		return Ns;
	}
}





template<class I,class J,class P=signed char>
npy_intp make_basis_parallel(general_basis_core<I,P> *B,const npy_intp MAX,const npy_intp mem_MAX,I basis[],J n[]){
	npy_intp Ns = 0;

	bool insuff_mem = false;
	std::vector<npy_intp> master_pos(omp_get_max_threads()+1);
	npy_intp * master_pos_data = &master_pos[0];


	#pragma omp parallel firstprivate(MAX,mem_MAX) shared(master_pos_data,Ns,insuff_mem)
	{
		const int nthread = omp_get_num_threads();
		const int threadn = omp_get_thread_num();
		std::vector<std::pair<I,J> > thread_block;

		npy_intp chunk = MAX - threadn;

		I s = threadn;

		while(chunk>0){
			double norm = B->check_state(s);
			npy_intp int_norm = norm;

			if(!check_nan(norm) && int_norm>0 ){
				thread_block.push_back(std::make_pair(s,int_norm));
			}
			s += nthread;
			chunk -= nthread;

		}

		master_pos_data[threadn+1] = thread_block.size(); // get sizes for each thread block into shared memory

		#pragma omp barrier

		#pragma omp single // calculate the cumulative sum to get data paritions of master_block
		{
			for(int i=0;i<nthread;i++){
				master_pos_data[i+1] += master_pos_data[i];
			}
			Ns = master_pos_data[nthread];
			insuff_mem = Ns > mem_MAX;

		}

		if(!insuff_mem){

			const npy_intp start = master_pos_data[threadn];
			const npy_intp end = master_pos_data[threadn+1];
			npy_intp i = 0;

			for(npy_intp j=start;j<end;j++){
				basis[j] = thread_block[i].first;
				n[j] = thread_block[i++].second;
			}

		}
	}

	if(insuff_mem){
		return -1;
	}
	else{
		return Ns;
	}
}

template<class I,class J,class P=signed char>
npy_intp make_basis_pcon_parallel(general_basis_core<I,P> *B,const npy_intp MAX,const npy_intp mem_MAX,I s,I basis[],J n[]){
	npy_intp Ns = 0;

	bool insuff_mem = false;
	std::vector<npy_intp> master_pos(omp_get_max_threads()+1);
	npy_intp * master_pos_data = &master_pos[0];


	#pragma omp parallel firstprivate(MAX,mem_MAX,s) shared(master_pos_data,Ns,insuff_mem)
	{
		
		const int nthread = omp_get_num_threads();
		const int threadn = omp_get_thread_num();
		std::vector<std::pair<I,J> > thread_block; // local array to store values found by each thread. this reduces the number of critical sections. 

		npy_intp chunk = MAX - threadn;
		I nns = 0;// number of next_state calls
		for(int i=0;i<threadn;i++){s=B->next_state_pcon(s,nns++);}

		while(chunk>0){
			double norm = B->check_state(s);
			npy_intp int_norm = norm;

			if(!check_nan(norm) && int_norm>0 ){
				thread_block.push_back(std::make_pair(s,int_norm));
			}

			for(int i=0;i<nthread;i++){s=B->next_state_pcon(s,nns++);}
			chunk -= nthread;
		}

		master_pos_data[threadn+1] = thread_block.size(); // get sizes for each thread block into shared memory

		#pragma omp barrier

		#pragma omp single // calculate the cumulative sum to get data paritions
		{
			for(int i=0;i<nthread;i++){
				master_pos_data[i+1] += master_pos_data[i];
			}
			Ns = master_pos_data[nthread];
			insuff_mem = Ns > mem_MAX;

		}


		if(!insuff_mem){

			const npy_intp start = master_pos_data[threadn];
			const npy_intp end = master_pos_data[threadn+1];
			npy_intp i = 0;

			for(npy_intp j=start;j<end;j++){
				basis[j] = thread_block[i].first;
				n[j] = thread_block[i++].second;
			}

		}
	}

	if(insuff_mem){
		return -1;
	}
	else{
		// sort list based on basis and then fill ndarray values with the sorted list. 
		// master_block.resize(Ns);
		// std::sort(master_block.begin(),master_block.end(), compare_pair<I,J>());
		// for(npy_intp i=0;i<Ns;i++){
		// 	basis[i] = master_block[i].first;
		// 	n[i] = master_block[i].second;
		// }
		return Ns;
	}
}




template<class I,class J,class P=signed char>
npy_intp make_basis(general_basis_core<I,P> *B,npy_intp MAX,npy_intp mem_MAX,I basis[],J n[]){
	const int nt =  B->get_nt();
	const int nthreads = omp_get_max_threads();

	if(nthreads>1 && MAX > nthreads && (nt>0 || B->pre_check)){
		return make_basis_parallel(B,MAX,mem_MAX,basis,n);
	}
	else{
		// If there are no symmetries it does not make sense to use parallel version.
		// This is because it requires extra memory as well as extra time to sort
		// the basis states that are produced by the parallel code.
		return make_basis_sequential(B,MAX,mem_MAX,basis,n);
	}
}

template<class I,class J,class P=signed char>
npy_intp make_basis_pcon(general_basis_core<I,P> *B,npy_intp MAX,npy_intp mem_MAX,I s,I basis[],J n[]){
	const int nt =  B->get_nt();
	const int nthreads = omp_get_max_threads();

	if(nthreads>1 && MAX > nthreads && (nt>0 || B->pre_check)){
		return make_basis_pcon_parallel(B,MAX,mem_MAX,s,basis,n);
	}
	else{
		// If there are no symmetries it does not make sense to use parallel version.
		// This is because it requires extra memory as well as extra time to sort
		// the basis states that are produced by the parallel code.
		return make_basis_pcon_sequential(B,MAX,mem_MAX,s,basis,n);
	}
}



// template<class I,class J>
// npy_intp inline make_basis_wrapper(void *B,npy_intp MAX,npy_intp mem_MAX,void * basis,J n[]){
// 	return make_basis(reinterpret_cast<general_basis_core<I> *>(B),MAX,mem_MAX,(I*)basis,n);
// }

// template<class I,class J>
// npy_intp inline make_basis_pcon_wrapper(void *B,npy_intp MAX,npy_intp mem_MAX,npy_uint64 s,void * basis,J n[]){
// 	return make_basis_pcon(reinterpret_cast<general_basis_core<I> *>(B),MAX,mem_MAX,(I)s,(I*)basis,n);
// }

}

#endif
