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
//#include "complex_ops.h"

namespace basis_general {



template<class I, class J, class K, class T,class P=signed char>
int general_op(general_basis_core<I,P> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const bool full_basis,
						  const npy_intp Ns,
						  const I basis[],
						  const J n[],
						  		K row[],
						  		K col[],
						  		T M[]
						  )
{
	int err = 0;
	#pragma omp parallel 
	{
		const int nt = B->get_nt();
		const int N = B->get_N();
		const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
		int g[__GENERAL_BASIS_CORE__max_nt];
		double kk[__GENERAL_BASIS_CORE__max_nt];

		for(int k=0;k<nt;k++)
			kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];


		#pragma omp for schedule(dynamic,chunk)
		for(npy_intp i=0;i<Ns;i++){
			if(err != 0){
				continue;
			}

			I r = basis[i];
			std::complex<double> m = A;
			int local_err = B->op(r,m,n_op,opstr,indx);

			if(local_err == 0){
				P sign = 1;

				for(int k=0;k<nt;k++){
					g[k]=0;
				}

				K j = i;
				if(r != basis[i]){
					I rr = B->ref_state(r,g,sign);
					if(full_basis){
						j = Ns - (npy_intp)rr - 1;
					}
					else{
						j = rep_position(Ns,basis,rr);
					}
					
				}
				if(j >= 0){
					double q = 0;
					for(int k=0;k<nt;k++){
						q += kk[k]*g[k];
					}
					m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));
					
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
			else{
				#pragma omp critical
				err = local_err;
			}
		}

		
	}
	return err;
}


template<class I, class J, class K, class T,class P=signed char>
int general_op(general_basis_core<I,P> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const bool full_basis,
						  const npy_intp Ns,
						  const I basis[],
						  const J n[],
						  const npy_intp basis_begin[],
						  const npy_intp basis_end[],
						  const int N_p,
						  		K row[],
						  		K col[],
						  		T M[]
						  )
{
	int err = 0;
	const int nt = B->get_nt();

	#pragma omp parallel 
	{
		const int nt = B->get_nt();
		const int N = B->get_N();

		const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
		int g[__GENERAL_BASIS_CORE__max_nt];
		double kk[__GENERAL_BASIS_CORE__max_nt];

		for(int k=0;k<nt;k++)
			kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];


		#pragma omp for schedule(dynamic,chunk)
		for(npy_intp i=0;i<Ns;i++){
			if(err != 0){
				continue;
			}

			I r = basis[i];
			std::complex<double> m = A;
			int local_err = B->op(r,m,n_op,opstr,indx);

			if(local_err == 0){
				P sign = 1;

				for(int k=0;k<nt;k++){
					g[k]=0;
				}

				K j = i;
				if(r != basis[i]){
					I rr = B->ref_state(r,g,sign);
					if(full_basis){
						j = Ns - (npy_intp)rr - 1;
					}
					else{
						npy_intp rr_prefix = B->get_prefix(rr,N_p);
						j = rep_position<K,I>(basis_begin,basis_end,basis,rr_prefix,rr);
					}
					
				}
				if(j >= 0){
					if(nt > 0){
						double q = 0;
						for(int k=0;k<nt;k++){
							q += kk[k]*g[k];
						}
						m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));
					}
					
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
			else{
				#pragma omp critical
				err = local_err;
			}
		}

		
	}
	return err;
}


int general_inplace_op_get_switch_num(PyArray_Descr * dtype)
{
	int T = dtype->type_num;
	if(T==NPY_COMPLEX128){return 0;}
	else if(T==NPY_FLOAT64){return 1;}
	else if(T==NPY_COMPLEX64){return 2;}
	else if(T==NPY_FLOAT32){return 3;}
	else {return -1;}
}



template<class I, class J, class K,class P=signed char>
int general_inplace_op_core(general_basis_core<I,P> *B,
						  const bool conjugate,
						  const bool transpose,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const bool full_basis,
						  const npy_intp Ns,
						  const npy_intp nvecs,
						  const I basis[],
						  const J n[],
						  const npy_intp basis_begin[],
						  const npy_intp basis_end[],
						  const int N_p,
						  const K v_in[],
						  		K v_out[])
{
	int err = 0;
	const int nt = B->get_nt();
	if(full_basis){
		if(transpose){
			if(conjugate){
				#pragma omp parallel
				{
					const int nt = B->get_nt();
					const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
					
					#pragma omp for schedule(dynamic,chunk)
					for(npy_intp i=0;i<Ns;i++){
						if(err != 0){
							continue;
						}

						I r = basis[i];
						std::complex<double> m = A;
						int local_err = B->op(r,m,n_op,opstr,indx);

						if(local_err == 0){
							npy_intp j = Ns - (npy_intp)r - 1;

							const K * v_in_col  = v_in  + j * nvecs;
								  K * v_out_row = v_out + i * nvecs;

							for(int k=0;k<nvecs;k++){
								const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
								local_err = atomic_add(ME,&v_out_row[k]);
								if(local_err){
									break;
								}
							}
						}
						else{
							#pragma omp critical
							err = local_err;
						}
					}
				}	
			}
			else{
				#pragma omp parallel
				{
					const int nt = B->get_nt();
					const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
					
					#pragma omp for schedule(dynamic,chunk)
					for(npy_intp i=0;i<Ns;i++){
						if(err != 0){
							continue;
						}

						I r = basis[i];
						std::complex<double> m = A;
						int local_err = B->op(r,m,n_op,opstr,indx);

						if(local_err == 0){
							npy_intp j = Ns - (npy_intp)r - 1;

							const K * v_in_col  = v_in  + j * nvecs;
								  K * v_out_row = v_out + i * nvecs;

							for(int k=0;k<nvecs;k++){
								const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
								local_err = atomic_add(ME,&v_out_row[k]);
								if(local_err){
									break;
								}
							}
						}
						else{
							#pragma omp critical
							err = local_err;
						}
					}
				}			
			}
		}
		else{
			if(conjugate){
				#pragma omp parallel
				{
					const int nt = B->get_nt();
					const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
					
					#pragma omp for schedule(dynamic,chunk)
					for(npy_intp i=0;i<Ns;i++){
						if(err != 0){
							continue;
						}

						I r = basis[i];
						std::complex<double> m = A;
						int local_err = B->op(r,m,n_op,opstr,indx);

						if(local_err == 0){
							npy_intp j = Ns - (npy_intp)r - 1;

							const K * v_in_col  = v_in  + j * nvecs;
								  K * v_out_row = v_out + i * nvecs;

							for(int k=0;k<nvecs;k++){
								const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
								local_err = atomic_add(ME,&v_out_row[k]);
								if(local_err){
									break;
								}
							}
						}
						else{
							#pragma omp critical
							err = local_err;
						}
					}
				}	
			}
			else{
				#pragma omp parallel
				{
					const int nt = B->get_nt();
					const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
					
					#pragma omp for schedule(dynamic,chunk)
					for(npy_intp i=0;i<Ns;i++){
						if(err != 0){
							continue;
						}

						I r = basis[i];
						std::complex<double> m = A;
						int local_err = B->op(r,m,n_op,opstr,indx);

						if(local_err == 0){
							npy_intp j = Ns - (npy_intp)r - 1;

							const K * v_in_col  = v_in  + i * nvecs;
								  K * v_out_row = v_out + j * nvecs;

							for(int k=0;k<nvecs;k++){
								const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
								local_err = atomic_add(ME,&v_out_row[k]);
								if(local_err){
									break;
								}
							}
						}
						else{
							#pragma omp critical
							err = local_err;
						}
					}
				}			
			}
		}
	}
	else if(nt==0){
		if(N_p>0){
			if(transpose){
				if(conjugate){
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;
								npy_intp j = i;
								if(r != basis[i]){
									npy_intp r_prefix = B->get_prefix(r,N_p);
									j = rep_position<npy_intp,I>(basis_begin,basis_end,basis,r_prefix,r);
								}

								if(j >= 0){
									const K * v_in_col  = v_in  + j * nvecs;
										  K * v_out_row = v_out + i * nvecs;

									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
				else{
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									npy_intp r_prefix = B->get_prefix(r,N_p);
									j = rep_position<npy_intp,I>(basis_begin,basis_end,basis,r_prefix,r);
								}

								if(j >= 0){
									const K * v_in_col  = v_in  + j * nvecs;
										  K * v_out_row = v_out + i * nvecs;
									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
			}
			else{
				if(conjugate){
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									npy_intp r_prefix = B->get_prefix(r,N_p);
									j = rep_position<npy_intp,I>(basis_begin,basis_end,basis,r_prefix,r);
								}

								if(j >= 0){
									const K * v_in_col  = v_in  + i * nvecs;
										  K * v_out_row = v_out + j * nvecs;
									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
				else{
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									npy_intp r_prefix = B->get_prefix(r,N_p);
									j = rep_position<npy_intp,I>(basis_begin,basis_end,basis,r_prefix,r);
								}

								if(j >= 0){
									const K * v_in_col  = v_in  + i * nvecs;
										  K * v_out_row = v_out + j * nvecs;

									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}			
				}
			}
		}
		else{
			if(transpose){
				if(conjugate){
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
					
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;
								npy_intp j = i;
								if(r != basis[i]){
									j = rep_position(Ns,basis,r);
								}

								if(j >= 0){

									const K * v_in_col  = v_in  + j * nvecs;
										  K * v_out_row = v_out + i * nvecs;

									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
				else{
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									j = rep_position(Ns,basis,r);						
								}

								if(j >= 0){
									const K * v_in_col  = v_in  + j * nvecs;
										  K * v_out_row = v_out + i * nvecs;
									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
			}
			else{
				if(conjugate){
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									j = rep_position(Ns,basis,r);;
								}

								if(j >= 0){
									const K * v_in_col  = v_in  + i * nvecs;
										  K * v_out_row = v_out + j * nvecs;
									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
				else{
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									j = rep_position(Ns,basis,r);
								}

								if(j >= 0){
									const K * v_in_col  = v_in  + i * nvecs;
										  K * v_out_row = v_out + j * nvecs;

									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}			
				}
			}			
		}
	}
	else{
		if(N_p>0){
			if(transpose){
				if(conjugate){
					#pragma omp parallel
					{

						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						int g[__GENERAL_BASIS_CORE__max_nt];
						double kk[__GENERAL_BASIS_CORE__max_nt];

						for(int k=0;k<nt;k++)
							kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;
								npy_intp j = i;
								if(r != basis[i]){
									I rr = B->ref_state(r,g,sign);
									npy_intp rr_prefix = B->get_prefix(rr,N_p);
									j = rep_position<npy_intp,I>(basis_begin,basis_end,basis,rr_prefix,rr);
								}

								if(j >= 0){
									double q = 0;
									for(int k=0;k<nt;k++){
										q += kk[k]*g[k];								
									}
									m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));

									const K * v_in_col  = v_in  + j * nvecs;
										  K * v_out_row = v_out + i * nvecs;

									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
				else{
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						int g[__GENERAL_BASIS_CORE__max_nt];
						double kk[__GENERAL_BASIS_CORE__max_nt];

						for(int k=0;k<nt;k++)
							kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									I rr = B->ref_state(r,g,sign);
									npy_intp rr_prefix = B->get_prefix(rr,N_p);
									j = rep_position<npy_intp,I>(basis_begin,basis_end,basis,rr_prefix,rr);								
								}

								if(j >= 0){
									double q = 0;
									for(int k=0;k<nt;k++){
										q += kk[k]*g[k];								
									}
									m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));

									const K * v_in_col  = v_in  + j * nvecs;
										  K * v_out_row = v_out + i * nvecs;
									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
			}
			else{
				if(conjugate){
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						int g[__GENERAL_BASIS_CORE__max_nt];
						double kk[__GENERAL_BASIS_CORE__max_nt];

						for(int k=0;k<nt;k++)
							kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									I rr = B->ref_state(r,g,sign);
									npy_intp rr_prefix = B->get_prefix(rr,N_p);
									j = rep_position<npy_intp,I>(basis_begin,basis_end,basis,rr_prefix,rr);
								}

								if(j >= 0){
									double q = 0;
									for(int k=0;k<nt;k++){
										q += kk[k]*g[k];								
									}
									m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));


									const K * v_in_col  = v_in  + i * nvecs;
										  K * v_out_row = v_out + j * nvecs;
									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
				else{
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						int g[__GENERAL_BASIS_CORE__max_nt];
						double kk[__GENERAL_BASIS_CORE__max_nt];

						for(int k=0;k<nt;k++)
							kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									I rr = B->ref_state(r,g,sign);
									npy_intp rr_prefix = B->get_prefix(rr,N_p);
									j = rep_position<npy_intp,I>(basis_begin,basis_end,basis,rr_prefix,rr);
								}

								if(j >= 0){
									double q = 0;
									for(int k=0;k<nt;k++){
										q += kk[k]*g[k];								
									}
									m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));

									const K * v_in_col  = v_in  + i * nvecs;
										  K * v_out_row = v_out + j * nvecs;

									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}			
				}
			}
		}
		else{
			if(transpose){
				if(conjugate){
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						int g[__GENERAL_BASIS_CORE__max_nt];
						double kk[__GENERAL_BASIS_CORE__max_nt];

						for(int k=0;k<nt;k++)
							kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;
								npy_intp j = i;
								if(r != basis[i]){
									I rr = B->ref_state(r,g,sign);
									j = rep_position(Ns,basis,rr);
								}

								if(j >= 0){
									double q = 0;
									for(int k=0;k<nt;k++){
										q += kk[k]*g[k];								
									}
									m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));

									const K * v_in_col  = v_in  + j * nvecs;
										  K * v_out_row = v_out + i * nvecs;

									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
				else{
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						int g[__GENERAL_BASIS_CORE__max_nt];
						double kk[__GENERAL_BASIS_CORE__max_nt];

						for(int k=0;k<nt;k++)
							kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									I rr = B->ref_state(r,g,sign);
									j = rep_position(Ns,basis,rr);							
								}

								if(j >= 0){
									double q = 0;
									for(int k=0;k<nt;k++){
										q += kk[k]*g[k];								
									}
									m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));


									const K * v_in_col  = v_in  + j * nvecs;
										  K * v_out_row = v_out + i * nvecs;
									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
			}
			else{
				if(conjugate){
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						int g[__GENERAL_BASIS_CORE__max_nt];
						double kk[__GENERAL_BASIS_CORE__max_nt];

						for(int k=0;k<nt;k++)
							kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									I rr = B->ref_state(r,g,sign);
									j = rep_position(Ns,basis,rr);
								}

								if(j >= 0){

									double q = 0;
									for(int k=0;k<nt;k++){
										q += kk[k]*g[k];								
									}
									m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));

									const K * v_in_col  = v_in  + i * nvecs;
										  K * v_out_row = v_out + j * nvecs;
									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * std::conj(m);
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}
				}
				else{
					#pragma omp parallel
					{
						const npy_intp chunk = std::max(Ns/(1000*omp_get_num_threads()),(npy_intp)1);
						int g[__GENERAL_BASIS_CORE__max_nt];
						double kk[__GENERAL_BASIS_CORE__max_nt];

						for(int k=0;k<nt;k++)
							kk[k] = 2.0*M_PI*B->qs[k]/B->pers[k];
						
						#pragma omp for schedule(dynamic,chunk)
						for(npy_intp i=0;i<Ns;i++){
							if(err != 0){
								continue;
							}

							I r = basis[i];
							std::complex<double> m = A;
							int local_err = B->op(r,m,n_op,opstr,indx);

							if(local_err == 0){
								P sign = 1;

								npy_intp j = i;
								if(r != basis[i]){
									I rr = B->ref_state(r,g,sign);
									j = rep_position(Ns,basis,rr);
								}

								if(j >= 0){
									double q = 0;
									for(int k=0;k<nt;k++){
										q += kk[k]*g[k];								
									}
									m *= sign * std::sqrt(double(n[j])/double(n[i])) * std::exp(std::complex<double>(0,-q));

									const K * v_in_col  = v_in  + i * nvecs;
										  K * v_out_row = v_out + j * nvecs;

									for(int k=0;k<nvecs;k++){
										const std::complex<double> ME = std::complex<double>(v_in_col[k]) * m;
										local_err = atomic_add(ME,&v_out_row[k]);
										if(local_err){
											break;
										}
									}
								}
							}
							else{
								#pragma omp critical
								err = local_err;
							}
						}
					}			
				}
			}			
		}
	}
	return err;
}

template<class I, class J,class P=signed char>
int general_inplace_op(general_basis_core<I,P> *B,
						  const bool conjugate,
						  const bool transpose,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  		void * A,
						  const bool full_basis,
						  const npy_intp Ns,
						  const npy_intp nvecs,
						  const I basis[],
						  const J n[],
						  const npy_intp basis_begin[],
						  const npy_intp basis_end[],
						  const int N_p,
						  PyArray_Descr * dtype,
						  		void * v_in,
						  		void * v_out)
{
	int type_num = dtype->type_num;
	switch (type_num)
	{
		case NPY_COMPLEX128:
			return general_inplace_op_core<I,J,npy_cdouble_wrapper,P>(B,conjugate,transpose,n_op,opstr,indx,*(npy_cdouble_wrapper *)A,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const npy_cdouble_wrapper *)v_in,(npy_cdouble_wrapper *)v_out);
		case NPY_FLOAT64:
			return general_inplace_op_core<I,J,double,P>(B,conjugate,transpose,n_op,opstr,indx,*(npy_cdouble_wrapper *)A,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const double *)v_in,(double *)v_out);
		case NPY_COMPLEX64:
			return general_inplace_op_core<I,J,npy_cfloat_wrapper,P>(B,conjugate,transpose,n_op,opstr,indx,*(npy_cdouble_wrapper *)A,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const npy_cfloat_wrapper *)v_in,(npy_cfloat_wrapper *)v_out);
		case NPY_FLOAT32:
			return general_inplace_op_core<I,J,float,P>(B,conjugate,transpose,n_op,opstr,indx,*(npy_cdouble_wrapper *)A,full_basis,Ns,nvecs,basis,n,basis_begin,basis_end,N_p,(const float *)v_in,(float *)v_out);
		default:
	        return -2;
	}	
}



template<class I, class T, class P=signed char>
int general_op_bra_ket(general_basis_core<I,P> *B,
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
	int err = 0;
	#pragma omp parallel
	{
		const int nt = B->get_nt();
		int g[__GENERAL_BASIS_CORE__max_nt];
			
		#pragma omp for schedule(static)
		for(npy_intp i=0;i<Ns;i++){
			if(err != 0){
				continue;
			}

			std::complex<double> m = A;
			const I s = ket[i];
			I r = ket[i];
			
			int local_err = B->op(r,m,n_op,opstr,indx);

			if(local_err == 0){
				P sign = 1;

				if(r != s){ // off-diagonal matrix element
					r = B->ref_state(r,g,sign);
					// use check_state to determine if state is a representative (same routine as in make-general_basis)
					double norm_r = B->check_state(r);

					if(!check_nan(norm_r) && norm_r > 0){ // ref_state is a representative

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
					m *= sign;
					local_err = check_imag(m,&M[i]); // assigns value to M[i]
					bra[i] = s;
				}
				
				
			}
			else{
				#pragma omp critical
				err = local_err;
			}
		}

		
	}
	return err;
}






template<class I, class T, class P=signed char>
int general_op_bra_ket_pcon(general_basis_core<I,P> *B,
						  const int n_op,
						  const char opstr[],
						  const int indx[],
						  const std::complex<double> A,
						  const npy_intp Ns,
						  const std::set<std::vector<int>> &Np_set, // array with particle conserving sectors
						  const	I ket[], // col
						  		I bra[], // row
						  		T M[]
						  )
{
	int err = 0;

	#pragma omp parallel
	{
		const std::set<std::vector<int>> Np_set_local = Np_set;
		const int nt = B->get_nt();
		int g[__GENERAL_BASIS_CORE__max_nt];
		
		#pragma omp for schedule(static) 
		for(npy_intp i=0;i<Ns;i++){
			if(err != 0){
				continue;
			}

			std::complex<double> m = A;
			const I s = ket[i];
			I r = ket[i];
			
			int local_err = B->op(r,m,n_op,opstr,indx);

			if(local_err == 0){
				P sign = 1;


				if(r != s){ // off-diagonal matrix element
					r = B->ref_state(r,g,sign);

					bool pcon_bool = B->check_pcon(r,Np_set_local);

					if(pcon_bool){ // reference state within same particle-number sector(s)

						// use check_state to determine if state is a representative (same routine as in make-general_basis)
						double norm_r = B->check_state(r);

						if(!check_nan(norm_r) && norm_r > 0){ // ref_state is a representative

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
			else{
				#pragma omp critical
				err = local_err;
			}
		}

		
	}
	return err;
}


}

#endif
