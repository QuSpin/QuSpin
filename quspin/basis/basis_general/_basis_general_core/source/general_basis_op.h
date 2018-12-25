#ifndef _GENERAL_BASIS_OP_H
#define _GENERAL_BASIS_OP_H

#include <iostream>
#include <complex>
#include <limits>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"



template<class I>
int count_bits (I s) 
{
    int count=0;
    while (s!=0)
    {
        s = s & (s-1);
        count++;
    }
    return count;
}


template<class K,class I>
K binary_search(const K N,const I A[],const I s){
	K b,bmin,bmax;
	bmin = 0;
	bmax = N-1;
	while(bmin<=bmax){
		b = (bmax+bmin)/2;
		I a = A[b];
		if(s==a){
			return b;
		}
		else if(s<A[b]){
			bmin = b + 1;
		}
		else{
			bmax = b - 1;
		}
	}
	return -1;
}

template<class T>
int inline check_imag(std::complex<double> m,std::complex<T> *M){
	M[0].real(m.real());
	M[0].imag(m.imag());
	return 0;
}

template<class T>
int inline check_imag(std::complex<double> m,T *M){
	if(std::abs(m.imag())>1.1e-15){
		return 1;
	}
	else{
		M[0] = m.real();
		return 0;
	}
}



template<class I, class J, class K, class T>
int general_op(general_basis_core<I> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const npy_intp Ns,
						  const I basis[],
						  const J n[],
						  		K row[],
						  		K col[],
						  		T M[]
						  )
{
	const int nt = B->get_nt();
	int err = 0;
	int g[128],gg[128];
	#pragma omp parallel for schedule(static,1) private(g,gg)
	for(npy_intp i=0;i<Ns;i++){
		if(err != 0){
			continue;
		}

		I r = basis[i];
		std::complex<double> m = A;
		int local_err = B->op(r,m,n_op,opstr,indx);

		if(local_err == 0){
			int sign = 1;

			for(int k=0;k<nt;k++){
				gg[k]=g[k]=0;
			}

			K j = i;
			if(r != basis[i]){
				I rr = B->ref_state(r,g,gg,sign);
				j = binary_search(Ns,basis,rr);
			}

			if(j >= 0){
				for(int k=0;k<nt;k++){
					double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
					m *= std::exp(std::complex<double>(0,-q));
				}
				m *= sign * std::sqrt(double(n[j])/double(n[i]));
				local_err = check_imag(m,&M[i]);
				col[i]=i;
				row[i]=j;
			}
			else{
				col[i] = i;
				row[i] = i;
				M[i] = std::numeric_limits<T>::quiet_NaN();
			}
		}

		if(local_err != 0){
			#pragma omp critical
			err = local_err;
		}
	}

	return err;
}


template<class I, class T>
int general_op_int_state_pcon(general_basis_core<I> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const npy_intp Ns,
						  const int Npcon_blocks, // total number of particle-conserving sectors
						  const I Np[], // array with particle conserving sectors
						  const I states[],
						  		I ket[], // row
						  		I bra[], // col
						  		T M[]
						  )
{
	const int nt = B->get_nt();
	int err = 0;
	int g[128],gg[128];
		
	#pragma omp parallel for schedule(static,1) private(g,gg)

	for(npy_intp i=0;i<Ns;i++){
		if(err != 0){
			continue;
		}

		std::complex<double> m = A;
		const I s = states[i];
		I r = states[i];
		
		int local_err = B->op(r,m,n_op,opstr,indx);

		if(local_err == 0){
			int sign = 1;
				
			if(r != s){ // off-diagonal matrix element
				r = B->ref_state(r,g,gg,sign);

				I np = count_bits(r); // compute particle number

				bool pcon_bool = 0;
				for(int n=0; n<Npcon_blocks; n++){
					pcon_bool |= (np==Np[n]);
				}

				if(pcon_bool){ // reference state within same particle-number sector(s)

					// use check_state to determine if state is a representative (same routine as in make-general_basis)
					double norm_r = B->check_state(r);
					double int_norm = norm_r;

					#if defined(_WIN64)
						// x64 version
						bool isnan = _isnanf(norm_r) != 0;
					#elif defined(_WIN32)
						bool isnan = _isnan(norm_r) != 0;
					#else
						bool isnan = std::isnan(norm_r);
					#endif


					if(!isnan && int_norm > 0){ // ref_state is a representative

						for(int k=0;k<nt;k++){
							double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
							m *= std::exp(std::complex<double>(0,-q));
						}

						double norm_s = B->check_state(s);
						m *= sign * std::sqrt(norm_r/norm_s);

						local_err = check_imag(m,&M[i]); // assigns value to M[i]
						bra[i] = s;
						ket[i] = r;

					}
					else{ // ref_state not a representative
						bra[i] = s;
						ket[i] = s;
						M[i] = std::numeric_limits<T>::quiet_NaN();

					}

				}
				else{ // ref state in different particle number sector
					bra[i] = s;
					ket[i] = s;
					M[i] = std::numeric_limits<T>::quiet_NaN();
				}

				
			}
			else{ // diagonal matrix element

				for(int k=0;k<nt;k++){
					double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
					m *= std::exp(std::complex<double>(0,-q));
				}

				m *= sign;

				local_err = check_imag(m,&M[i]); // assigns value to M[i]
				bra[i] = s;
				ket[i] = s;
			}
			
			
		}


		if(local_err != 0){
			#pragma omp critical
			err = local_err;
		}
	}

	return err;
}


template<class I, class T>
int general_op_int_state(general_basis_core<I> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const npy_intp Ns,
						  const I states[],
						  		I ket[], // row
						  		I bra[], // col
						  		T M[]
						  )
{
	const int nt = B->get_nt();
	int err = 0;
	int g[128],gg[128];
		
	#pragma omp parallel for schedule(static,1) private(g,gg)

	for(npy_intp i=0;i<Ns;i++){
		if(err != 0){
			continue;
		}

		std::complex<double> m = A;
		const I s = states[i];
		I r = states[i];
		
		int local_err = B->op(r,m,n_op,opstr,indx);

		if(local_err == 0){
			int sign = 1;
				
			if(r != s){ // off-diagonal matrix element
				r = B->ref_state(r,g,gg,sign);

			
				// use check_state to determine if state is a representative (same routine as in make-general_basis)
				double norm_r = B->check_state(r);
				double int_norm = norm_r;

				#if defined(_WIN64)
					// x64 version
					bool isnan = _isnanf(norm_r) != 0;
				#elif defined(_WIN32)
					bool isnan = _isnan(norm_r) != 0;
				#else
					bool isnan = std::isnan(norm_r);
				#endif


				if(!isnan && int_norm > 0){ // ref_state is a representative

					for(int k=0;k<nt;k++){
						double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
						m *= std::exp(std::complex<double>(0,-q));
					}

					double norm_s = B->check_state(s);
					m *= sign * std::sqrt(norm_r/norm_s);

					local_err = check_imag(m,&M[i]); // assigns value to M[i]
					bra[i] = s;
					ket[i] = r;

					

				}
				else{ // ref state in different particle number sector
					bra[i] = s;
					ket[i] = s;
					M[i] = std::numeric_limits<T>::quiet_NaN();
				}

				
			}
			else{ // diagonal matrix element

				for(int k=0;k<nt;k++){
					double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
					m *= std::exp(std::complex<double>(0,-q));
				}

				m *= sign;

				local_err = check_imag(m,&M[i]); // assigns value to M[i]
				bra[i] = s;
				ket[i] = s;
			}
			
			
		}


		if(local_err != 0){
			#pragma omp critical
			err = local_err;
		}
	}

	return err;
}


#endif
