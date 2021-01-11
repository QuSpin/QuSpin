#ifndef _GENERAL_BASIS_GET_AMP_H
#define _GENERAL_BASIS_GET_AMP_H

//#include <limits>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"
//#include <complex>


namespace basis_general {



template<class I,class P=signed char>
std::complex<double> get_amp_rep(general_basis_core<I,P> *B,
								 const int nt,
									   I r, // start out with representative state and iterate over all transofmrations. 
								 const I s, // target states to find amplitude of
								       double k = 0.0,
									   P sign = 1,
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


template<class I,class J,class P=signed char>
int get_amp_general(general_basis_core<I,P> *B,
						  I s[],   // input states in the full basis
					      J out[], // state amplitudes of state s (full basis)
					const npy_intp Ns   // length of above arrays (should be the same)			

	)

{
	int err=0;
	double per_factor = 1.0;
	int q_sum = 0; // sum of quantum numbers
	const int nt = B->get_nt();

	
	for(int i=0;i<nt;i++){
		per_factor *= B->pers[i];
		q_sum += std::abs(B->qs[i]);
	}

	const npy_intp chunk = std::max(Ns/(100*omp_get_max_threads()),(npy_intp)1); // check_state has variable workload 

	if(q_sum > 0 || B->fermionic){   // a non-zero quantum number, or fermionic basis => need a nontrivial phase_factor

		#pragma omp parallel for schedule(dynamic,chunk)
		for(npy_intp i=0;i<Ns;i++){

			if(err == 0){
				std::complex<double> phase_factor, out_tmp;
				int g[__GENERAL_BASIS_CORE__max_nt];
				P sign=1;
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

				int local_err = type_checks(out_tmp, &out[i]); // compute and assign amplitude in full basis
				if(local_err){
					#pragma omp critical
					err = local_err;
				}					 		
			
			}
		
		}

		

	}
	else{
		#pragma omp parallel for schedule(dynamic,chunk)
		for(npy_intp i=0;i<Ns;i++){

			if(err == 0){
				std::complex<double> phase_factor, out_tmp;
				int g[__GENERAL_BASIS_CORE__max_nt];
				P sign=1;
				I ss=s[i];

				I r = B->ref_state(ss,g,sign);
				double norm_r = B->check_state(r);

				s[i] = r; // update state with representative

				if(!check_nan(norm_r) && norm_r > 0){ // ref_state is a representative

					//phase_factor = get_amp_rep(B,nt,r,ss);
					out_tmp = std::sqrt(norm_r/per_factor);
				}
				else{
					out_tmp = 0.0;
				}

				int local_err = type_checks(out_tmp, &out[i]); // compute and assign amplitude in full basis
				if(local_err){
					#pragma omp critical
					err = local_err;
				}					 		
			
			}
		
		}

	}


	return err;


}





// same as get_amp_rep, but w/o calling ref_state and check_state
template<class I,class J,class P=signed char>
int get_amp_general_light(general_basis_core<I,P> *B,
						  I s[],   // input states in the symmetry-reduced basis
					      J out[], // state amplitudes of state s (symmetry-reduced basis)
					const npy_intp Ns   // length of above arrays (should be the same)			

	)

{
	int err=0;
	double per_factor = 1.0;
	int q_sum = 0; // sum of quantum numbers
	const int nt = B->get_nt();
	
	for(int i=0;i<nt;i++){
		per_factor *= B->pers[i];
		q_sum += std::abs(B->qs[i]);
	}

	const npy_intp chunk = std::max(Ns/(100*omp_get_max_threads()),(npy_intp)1); // check_state has variable workload 

	if(q_sum > 0 || B->fermionic){   // a non-zero quantum number, or fermionic basis => need a nontrivial phase_factor

		#pragma omp parallel for schedule(dynamic,chunk)
		for(npy_intp i=0;i<Ns;i++){

			if(err == 0){
				std::complex<double> phase_factor, out_tmp;

				I ss=s[i];

				double norm_r = B->check_state(ss);

				phase_factor = get_amp_rep(B,nt,ss,ss);
				out_tmp = phase_factor/std::sqrt(norm_r * per_factor);
				
				int local_err = type_checks(out_tmp, &out[i]); // compute and assign amplitude in full basis
				if(local_err){
					#pragma omp critical
					err = local_err;
				}				 		
			
			}
		
		}
	}
	else{

		#pragma omp parallel for schedule(dynamic,chunk)
		for(npy_intp i=0;i<Ns;i++){

			if(err == 0){
				std::complex<double> phase_factor, out_tmp;
		
				double norm_r = B->check_state(s[i]);

				out_tmp = std::sqrt(norm_r/per_factor);
				
				int local_err = type_checks(out_tmp, &out[i]); // compute and assign amplitude in full basis
				if(local_err){
					#pragma omp critical
					err = local_err;
				}
			
			}


		
		}

	}


	return err;


}




}
#endif
