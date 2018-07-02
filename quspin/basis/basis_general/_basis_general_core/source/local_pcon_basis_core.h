#ifndef _LOCAL_PCON_BASIS_CORE_OP_H
#define _LOCAL_PCON_BASIS_CORE_OP_H

#include <complex>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"

template<class I>
class local_pcon_basis_core : public general_basis_core<I>
{
	public:
		local_pcon_basis_core(const int _N) : \
		general_basis_core<I>::general_basis_core(_N) {}

		local_pcon_basis_core(const int _N,const int _nt,const int _maps[], \
								   const int _pers[], const int _qs[]) : \
		general_basis_core<I>::general_basis_core(_N,_nt,_maps,_pers,_qs) {}

		~local_pcon_basis_core() {}

		// I inline map_state(I s,int n_map,int &sign){
		// 	if(general_basis_core<I>::nt<=0){
		// 		return s;
		// 	}
		// 	const int n = general_basis_core<I>::N << 1;
		// 	return hcb_map_bits(s,&general_basis_core<I>::maps[n_map*n],n);
			
		// }

		// void map_state(I s[],npy_intp M,int n_map,signed char sign[]){
		// 	if(general_basis_core<I>::nt<=0){
		// 		return;
		// 	}
		// 	const int n = general_basis_core<I>::N << 1;
		// 	const int * map = &general_basis_core<I>::maps[n_map*n];
		// 	#pragma omp for schedule(static,1)
		// 	for(npy_intp i=0;i<M;i++){
		// 		s[i] = local_pcon_map_bits(s[i],map,n);
		// 		sign[i] *= 1;
		// 	}
		// }

		// void print(I s){
		// 	I s_left,s_right;
		// 	split_state(s,s_left,s_right);

		// 	std::cout << "|";
		// 	for(int i=0;i<general_basis_core<I>::N;i++){
		// 		std::cout << (s_left&1) << " ";
		// 		s_left>>=1;
		// 	}
		// 	std::cout << ">";

		// 	std::cout << "|";
		// 	for(int i=0;i<general_basis_core<I>::N;i++){
		// 		std::cout << (s_right&1) << " ";
		// 		s_right>>=1;
		// 	}
		// 	std::cout << ">";
		// }


		I next_state_pcon(I) = 0;
		int op(I&,std::complex<double>&,const int,const char[],const int[]) = 0;

		
};


#endif