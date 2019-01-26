#ifndef _GENERAL_BASIS_OP_H
#define _GENERAL_BASIS_OP_H

#include <iostream>
#include <complex>
#include <algorithm>
#include <limits>
#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"
#include "misc.h"
#include "openmp.h"




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
	const npy_intp chunk = std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1);

	int err = 0;
	int g[nt];
	#pragma omp parallel for schedule(dynamic,chunk) private(g)
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
				g[k]=0;
			}

			K j = i;
			if(r != basis[i]){
				I rr = B->ref_state(r,g,sign);
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



/*

template<class T>
int inline atomic_add(const std::complex<double> m,std::complex<T> *M){
	T * M_v = reinterpret_cast<T*>(M);
	const T m_real = m.real();
	const T m_imag = m.imag();
	#pragma omp atomic
	M[0] += m_real;
	#pragma omp atomic
	M[1] += m_imag;
	return 0;
}

template<class T>
int inline atomic_add(const std::complex<double> m,T *M){
	if(std::abs(m.imag())>1.1e-15){
		return 1;
	}
	else{
		const T m_real = m.real();
		#pragma omp atomic
		M[0] += m_real;
		return 0;
	}
}

template<class I, class J, class K>
int general_inplace_op(general_basis_core<I> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const npy_intp Ns,
						  const npy_intp nvecs,
						  const I basis[],
						  const J n[],
						  		K v_in[],
						  		K v_out[])
{
	const int nt = B->get_nt();
	const npy_intp chunk = std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1);

	int err = 0;
	int g[nt];
	#pragma omp parallel for schedule(dynamic,chunk) private(g)
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
				g[k]=0;
			}

			npy_intp j = i;
			if(r != basis[i]){
				I rr = B->ref_state(r,g,sign);
				j = binary_search(Ns,basis,rr);
			}

			if(j >= 0){
				for(int k=0;k<nt;k++){
					double q = (2.0*M_PI*B->qs[k]*g[k])/B->pers[k];
					m *= std::exp(std::complex<double>(0,-q));
				}
				m *= sign * std::sqrt(double(n[j])/double(n[i]))
				const K * v_in_col  = v_in  + i * nvecs;
					  K * v_out_row = v_out + j * nvecs;

				for(int k=0;k<nvecs;k++){
					const std::complex<double> ME = v_in_col[k] * m;
					local_err = atomic_add(ME,&v_out_col[k]);
					if(local_err){
						break;
					}
				}
			}
		}

		if(local_err != 0){
			#pragma omp critical
			err = local_err;
		}
	}

	return err;
}

*/

template<class I, class T>
int general_op_bra_ket(general_basis_core<I> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const npy_intp Ns,
						  const I ket[], // col
						  		I bra[], // row
						  		T M[]
						  )
{
	const int nt = B->get_nt();
	const npy_intp chunk = std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1);

	int err = 0;
	int g[nt];
		
	#pragma omp parallel for schedule(dynamic,chunk) private(g)
	for(npy_intp i=0;i<Ns;i++){
		if(err != 0){
			continue;
		}

		std::complex<double> m = A;
		const I s = ket[i];
		I r = ket[i];
		
		int local_err = B->op(r,m,n_op,opstr,indx);

		if(local_err == 0){
			int sign = 1;

			for(int k=0;k<nt;k++){
				g[k]=0;
			}
				
			if(r != s){ // off-diagonal matrix element
				r = B->ref_state(r,g,sign);

			
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
					bra[i] = r;

				}
				else{ // ref state in different particle number sector
					M[i] = std::numeric_limits<T>::quiet_NaN();
					bra[i] = s;
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
int general_op_bra_ket_pcon(general_basis_core<I> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const npy_intp Ns,
						  const int Npcon_blocks, // total number of particle-conserving sectors
						  //const unsigned long int Np[], // array with particle conserving sectors
						  const std::set<std::vector<int>> Np_set, // array with particle conserving sectors
						  const	I ket[], // col
						  		I bra[], // row
						  		T M[]
						  )
{
	const int nt = B->get_nt();
	const npy_intp chunk = std::max(Ns/(100*omp_get_num_threads()),(npy_intp)1);
	int err = 0;
	int g[nt];
		
	#pragma omp parallel for schedule(dynamic,chunk) private(g)
	for(npy_intp i=0;i<Ns;i++){
		if(err != 0){
			continue;
		}

		std::complex<double> m = A;
		const I s = ket[i];
		I r = ket[i];
		
		int local_err = B->op(r,m,n_op,opstr,indx);

		if(local_err == 0){
			int sign = 1;

			for(int k=0;k<nt;k++){
				g[k]=0;
			}
				
			if(r != s){ // off-diagonal matrix element
				r = B->ref_state(r,g,sign);

				bool pcon_bool = B->check_pcon(r,Np_set);

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
						bra[i] = r;

					}
					else{ // ref_state not a representative
						M[i] = std::numeric_limits<T>::quiet_NaN();
						bra[i] = s;
					}

				}
				else{ // ref state in different particle number sector
					M[i] = std::numeric_limits<T>::quiet_NaN();
					bra[i] = s;
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
