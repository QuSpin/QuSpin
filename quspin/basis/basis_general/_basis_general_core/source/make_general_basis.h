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






template<class I,class J>
npy_intp make_basis_sequential(general_basis_core<I> *B,npy_intp MAX,npy_intp mem_MAX,I basis[],J n[]){
	npy_intp Ns = 0;
	I s = 0;
	bool insuff_mem = false;

	while(MAX != 0){
		if(Ns>=mem_MAX){
			insuff_mem = true;
			break;
		}
		double norm = B->check_state(s);
		J int_norm = norm;

		#if defined(_WIN64)
			// x64 version
			bool isnan = _isnanf(norm) != 0;
		#elif defined(_WIN32)
			bool isnan = _isnan(norm) != 0;
		#else
			bool isnan = std::isnan(norm);
		#endif
		
		if(!isnan && int_norm>0 ){
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
		return Ns;
	}
}


template<class I, class J>
struct compare_pair : std::binary_function<std::pair<I,J>,std::pair<I,J>,bool>
{
	bool operator()(std::pair<I,J> a, std::pair<I,J> b){return std::get<0>(a) < std::get<0>(b);}
};

template<class I,class J>
npy_intp make_basis_pcon_sequential(general_basis_core<I> *B,npy_intp MAX,npy_intp mem_MAX,I s,I basis[],J n[]){
	npy_intp Ns = 0;
	bool insuff_mem = false;

	while(MAX!=0){
		if(Ns>=mem_MAX){
			insuff_mem = true;
			break;
		}
		double norm = B->check_state(s);
		J int_norm = norm;

		if(!check_nan(norm) && int_norm>0 ){
			basis[Ns] = s;
			n[Ns] = norm;
			Ns++;
		}
		s = B->next_state_pcon(s);
		MAX--;
	}

	if(insuff_mem){
		return -1;
	}
	else{
		return Ns;
	}
}

template<class I,class J>
npy_intp make_basis_parallel(general_basis_core<I> *B,npy_intp MAX,npy_intp mem_MAX,I basis[],J n[]){
	npy_intp Ns = 0;
	npy_intp index = 0;
	bool insuff_mem = false;
	std::vector<std::pair<I,J> > master_block(mem_MAX);
	std::pair<I,J> * master_block_data = &master_block[0];;

	#pragma omp parallel firstprivate(MAX) shared(master_block_data,index,Ns,insuff_mem)
	{
		std::vector<std::pair<I,J> > thread_block;
		const int nthread = omp_get_num_threads();
		const int threadn = omp_get_thread_num();
		I s = threadn;
		MAX -= threadn;

		while(MAX>0 && Ns < mem_MAX){
			double norm = B->check_state(s);
			J int_norm = norm;

			if(!check_nan(norm) && int_norm>0 ){
				thread_block.push_back(std::make_pair(s,int_norm));
				#pragma omp atomic
				Ns++;
			}
			s += nthread;
			MAX -= nthread;
		}

		if(Ns < mem_MAX){
			#pragma omp critical
			{
				const npy_intp Ns_block = thread_block.size();
				for(npy_intp j=0;j<Ns_block;j++){
					master_block_data[index] = thread_block[j];
					index++;
				}
			}
		}
		else{
			#pragma omp single
			insuff_mem = true;
		}

	}

	if(insuff_mem){
		return -1;
	}
	else{
		master_block.resize(Ns);
		std::sort(master_block.begin(),master_block.end(), compare_pair<I,J>());
		for(npy_intp i=0;i<Ns;i++){
			basis[i] = std::get<0>(master_block[i]);
			n[i] = std::get<1>(master_block[i]);
		}
		return Ns;
	}
}

template<class I,class J>
npy_intp make_basis_pcon_parallel(general_basis_core<I> *B,npy_intp MAX,npy_intp mem_MAX,I s,I basis[],J n[]){
	npy_intp Ns = 0;
	npy_intp index = 0;
	bool insuff_mem = false;
	std::vector<std::pair<I,J> > master_block(mem_MAX);
	std::pair<I,J> * master_block_data = &master_block[0];

	#pragma omp parallel firstprivate(MAX,s) shared(master_block_data,index,Ns,insuff_mem)
	{
		std::vector<std::pair<I,J> > thread_block;
		const int nthread = omp_get_num_threads();
		const int threadn = omp_get_thread_num();
		MAX -= threadn;
		for(int i=0;i<threadn;i++){s=B->next_state_pcon(s);}

		while(MAX>0 && Ns < mem_MAX){
			double norm = B->check_state(s);
			J int_norm = norm;

			if(!check_nan(norm) && int_norm>0 ){
				thread_block.push_back(std::make_pair(s,int_norm));
				#pragma omp atomic
				Ns++;
			}
			for(int i=0;i<nthread;i++){s=B->next_state_pcon(s);}
			MAX -= nthread;	
		}

		if(Ns < mem_MAX){
			#pragma omp critical
			{
				const npy_intp Ns_block = thread_block.size();
				for(npy_intp j=0;j<Ns_block;j++){
					master_block_data[index] = thread_block[j];
					index++;
				}
			}
		}
		else{
			#pragma omp single
			insuff_mem = true;
		}

	}

	if(insuff_mem){
		return -1;
	}
	else{
		master_block.resize(Ns);
		std::sort(master_block.begin(),master_block.end(), compare_pair<I,J>());
		for(npy_intp i=0;i<Ns;i++){
			basis[i] = std::get<0>(master_block[i]);
			n[i] = std::get<1>(master_block[i]);
		}
		return Ns;
	}
}




template<class I,class J>
npy_intp make_basis(general_basis_core<I> *B,npy_intp MAX,npy_intp mem_MAX,I basis[],J n[]){
	const int nt =  B->get_nt();
	const int nthreads = omp_get_max_threads();
	if(nthreads>1 && MAX > nthreads && nt>0){
		return make_basis_parallel(B,MAX,mem_MAX,basis,n);
	}
	else{
		return make_basis_sequential(B,MAX,mem_MAX,basis,n);
	}
}

template<class I,class J>
npy_intp make_basis_pcon(general_basis_core<I> *B,npy_intp MAX,npy_intp mem_MAX,I s,I basis[],J n[]){
	const int nt =  B->get_nt();
	const int nthreads = omp_get_max_threads();
	if(nthreads>1 && MAX > nthreads && nt>0){
		return make_basis_pcon_parallel(B,MAX,mem_MAX,s,basis,n);
	}
	else{
		return make_basis_pcon_sequential(B,MAX,mem_MAX,s,basis,n);
	}
}



template<class I,class J>
npy_intp inline make_basis_wrapper(void *B,npy_intp MAX,npy_intp mem_MAX,void * basis,J n[]){
	return make_basis(reinterpret_cast<general_basis_core<I> *>(B),MAX,mem_MAX,(I*)basis,n);
}

template<class I,class J>
npy_intp inline make_basis_pcon_wrapper(void *B,npy_intp MAX,npy_intp mem_MAX,npy_uint64 s,void * basis,J n[]){
	return make_basis_pcon(reinterpret_cast<general_basis_core<I> *>(B),MAX,mem_MAX,(I)s,(I*)basis,n);
}



#endif
