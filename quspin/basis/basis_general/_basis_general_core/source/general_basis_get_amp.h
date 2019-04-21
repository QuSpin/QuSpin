#ifndef _GENERAL_BASIS_GET_AMP_H
#define _GENERAL_BASIS_GET_AMP_H

#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"
//#include <complex>


namespace basis_general {



template<class I>
std::complex<double> get_amp_rep(general_basis_core<I> *B,
								 const int nt,
									   I r, // start out with representative state and iterate over all transofmrations. 
								 const I s, // target states to find amplitude of
								       double k = 0.0,
									   int sign = 1,
								 const int depth = 0
								)
{
	if(nt<=0){
		return 1.0;
	}

	std::complex<double> phase_factor = 0.0;
	const int per = B->pers[depth];
	const double q = (2.0*M_PI*B->qs[depth])/per;

	if(depth < nt-1){
		for(int j=0;j<per;j++){
			phase_factor += get_amp_rep(B,nt,r,s,k,sign,depth+1);
			k += q;
			r = B->map_state(r,depth,sign);
		}
	}
	else{
		for(int j=0;j<per;j++){
			if(r==s){
				phase_factor += double(sign)*std::exp(std::complex<double>(0,-k));
			}
			k += q;
			r = B->map_state(r,depth,sign);
		}
	}
	return phase_factor;
}


template<class I,class J>
int get_amp_general(general_basis_core<I> *B,
						  I s[],   // input states in the full basis
					      J out[], // state amplitudes of state s (full basis)
					const int Ns   // length of above arrays (should be the same)			

	)

{
	int err=0;
	int per_factor = 1.0;
	//int q_sum = 0; // sum of quantum numbers
	int g[__GENERAL_BASIS_CORE__max_nt];
	const int nt = B->get_nt();
	std::complex<double> phase_factor, out_tmp;
	//double k=0.0;

	for(int i=0;i<nt;i++){
		per_factor *= B->pers[i];
		//q_sum += std::abs(B->qs[i]);
	}


	//if(q_sum > 0 || B->fermionic){   // a non-zero quantum number, or fermionic basis => need a nontrivial phase_factor

	for(npy_intp i=0;i<Ns;i++){

		if(err == 0){

			int sign=1;
			I ss=s[i];

			I r = B->ref_state(ss,g,sign);
			double norm_r = B->check_state(r);

			s[i] = r; // update state with representative

			if(!check_nan(norm_r) && norm_r > 0){ // ref_state is a representative

				phase_factor = get_amp_rep(B,nt,r,ss);
				out_tmp = phase_factor/std::sqrt(norm_r * per_factor);
			}
			else{
				out_tmp = 0.0;
			}

			err = check_imag(out_tmp, &out[i]); // compute and assign amplitude in full basis
			 		
		
		}
		
	}

	//}


	return err;


}





}
#endif
