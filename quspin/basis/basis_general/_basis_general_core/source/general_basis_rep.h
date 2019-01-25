#ifndef _GENERAL_BASIS_REP_H
#define _GENERAL_BASIS_REP_H

#include <complex>
#include <limits>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"


template<class I,class J>
int general_normalization(general_basis_core<I> *B,
								  I s[],
								  J n[],
							const npy_intp Ns
				)
{	int err = 0;
	const npy_intp chunk = std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1); // check_state has variable workload 

	#pragma omp parallel for schedule(dynamic,chunk)
	for(npy_intp i=0;i<Ns;i++){

		double norm = B->check_state(s[i]);
		J int_norm = norm;

		if(err != 0){
			continue;
		}


		if( norm < std::numeric_limits<J>::max() ){ // checks if data type is large enough
			if(!check_nan(norm) && int_norm>0 ){
				n[i] = norm;
			}
			else{
				n[i] = 0;
			}

		}
		else{
			err = 1;
		}

	}

	return err;
}


template<class I>
void general_representative(general_basis_core<I> *B,
							const I s[],
									I r[],
									int *g_out_ptr,
									npy_int8 *sign_out_ptr,
							const npy_intp Ns
						  )
{
	const int nt = B->get_nt();
	const npy_intp chunk = Ns/omp_get_num_threads(); // NOTE: refstate time has a constant workload
	int g[nt];
	int sign;

	if(g_out_ptr && sign_out_ptr){
		#pragma omp parallel for schedule(static,chunk)
		for(npy_intp i=0;i<Ns;i++){

			int temp_sign = 1;
			r[i] = B->ref_state(s[i],&g_out_ptr[i*nt],temp_sign);
			sign_out_ptr[i] = temp_sign;
		}
	}
	else if(g_out_ptr){
		#pragma omp parallel for schedule(static,chunk)
		for(npy_intp i=0;i<Ns;i++){

			sign = 1;
			r[i] = B->ref_state(s[i],&g_out_ptr[i*nt],sign);
		}
	}
	else if(sign_out_ptr){
		#pragma omp parallel for schedule(static,chunk) private(g)
		for(npy_intp i=0;i<Ns;i++){

			int temp_sign = 1;
			r[i] = B->ref_state(s[i],g,temp_sign);
			sign_out_ptr[i] = temp_sign;
		}
	}
	else{
		#pragma omp parallel for schedule(static,chunk) private(g)
		for(npy_intp i=0;i<Ns;i++){
			sign = 1;
			r[i] = B->ref_state(s[i],g,sign);
		}
	}

	

}



#endif
