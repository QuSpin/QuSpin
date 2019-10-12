#ifndef _GENERAL_BASIS_REP_H
#define _GENERAL_BASIS_REP_H

#include <complex>
#include <limits>
#include <iostream>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"

namespace basis_general {

template<class I,class J,class P=signed char>
int general_normalization(general_basis_core<I,P> *B,
								  I s[],
								  J n[],
							const npy_intp Ns
				)
{	int err = 0;

	int nt=B->get_nt();
	int per_factor=1.0;
	for(int i=0;i<nt;i++){
		per_factor *= B->pers[i];
	}

	const npy_intp chunk = std::max(Ns/(100*omp_get_max_threads()),(npy_intp)1); // check_state has variable workload 

	#pragma omp parallel for schedule(dynamic,chunk)
	for(npy_intp i=0;i<Ns;i++){
		if(err != 0){
			continue;
		}

		double norm = B->check_state(s[i]);
		npy_intp int_norm = norm;
		
		// checks if data type is large enough
		if(!check_nan(norm) && int_norm>0 ){
			if( (npy_uintp)(int_norm * per_factor) > std::numeric_limits<J>::max() ){
				#pragma omp critical
				err = 1;
			}

			n[i] = (J)norm * per_factor;
		}
		else{
			n[i] = 0;
		}

	}

	return err;
}


template<class I,class P=signed char>
void general_representative(general_basis_core<I,P> *B,
							const I s[],
									I r[],
									int *g_out_ptr,
									P *phase_out_ptr,
							const npy_intp Ns
						  )
{

	const int nt = B->get_nt();
	if(g_out_ptr && phase_out_ptr){
		#pragma omp parallel
		{
			#pragma omp for schedule(static) // NOTE: refstate time has a constant workload
			for(npy_intp i=0;i<Ns;i++){
				P temp_phase = 1;
				r[i] = B->ref_state(s[i],&g_out_ptr[i*nt],temp_phase);
				phase_out_ptr[i] = temp_phase;
			}				
		}
	}
	else if(g_out_ptr){
		#pragma omp parallel
		{
			#pragma omp parallel for schedule(static) // NOTE: refstate time has a constant workload
			for(npy_intp i=0;i<Ns;i++){
				P temp_phase = 1;
				r[i] = B->ref_state(s[i],&g_out_ptr[i*nt],temp_phase);
			}				
		}
	}
	else if(phase_out_ptr){
		#pragma omp parallel
		{
			int g[__GENERAL_BASIS_CORE__max_nt];
			#pragma omp for schedule(static) // NOTE: refstate time has a constant workload
			for(npy_intp i=0;i<Ns;i++){
				P temp_phase = 1;
				r[i] = B->ref_state(s[i],g,temp_phase);
				phase_out_ptr[i] = temp_phase;
			}
			
		}

	}
	else{
		#pragma omp parallel
		{
			int g[__GENERAL_BASIS_CORE__max_nt];
			#pragma omp for schedule(static) // NOTE: refstate time has a constant workload
			for(npy_intp i=0;i<Ns;i++){
				P temp_phase = 1;
				r[i] = B->ref_state(s[i],g,temp_phase);
			}
			
		}
	}
}

}

#endif
