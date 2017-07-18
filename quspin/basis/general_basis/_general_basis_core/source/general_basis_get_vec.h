#ifndef _GENERAL_BASIS_GET_VEC_H
#define _GENERAL_BASIS_GET_VEC_H



#include "general_basis_core.h"
#include "numpy/ndarraytypes.h"


template<class T>
bool inline update_out_dense(std::complex<double> c, npy_intp n_vec,const std::complex<T> *in, std::complex<T> *out){
	for(npy_intp i=0;i<n_vec;i++){
		out[i] += std::complex<T>(c) * in[i];
	}
	return true;
}

template<class T>
bool inline update_out_dense(std::complex<double> c, npy_intp n_vec,const T *in, T *out){
	if(std::abs(c.imag())>1.1e-15){
		return false;
	}
	else{
		T re = c.real();
		for(npy_intp i=0;i<n_vec;i++){
			out[i] += re * in[i];
		}
		return true;
	}
}


template<class I,class J,class T>
bool get_vec_general_dense(general_basis_core<I> *B,
										 	   I basis[],
										 const J n[],
										 const npy_intp n_vec,
										 const npy_intp Ns,
										 const npy_intp Ns_full,
										 const T in[],
										 std::complex<double> c,
											   int nnt,
										 	   T out[])
{
	const int nt = B->get_nt();
	bool err = true;
	int per = B->pers[nt-nnt];
	double q = (2.0*M_PI*B->qs[nt-nnt])/B->pers[nt-nnt];
	std::complex<double> cc = std::exp(std::complex<double>(0,q));



	if(nnt > 1){
		for(int j=0;j<per && err;j++){
			err = get_vec_general_dense(B,basis,n,n_vec,Ns,Ns_full,in,c,nnt-1,out);
			c *= cc;
			B->map_state(basis,Ns,nt-nnt);			
		}
		return err;
	}
	else{
		for(int j=0;j<per && err;j++){
			npy_intp symm = 0;
			for(npy_intp k=0;k<Ns && err;k++){
				const npy_intp full = (Ns_full - basis[k]- 1)*n_vec;
				const double norm = std::sqrt(double(n[k]));

				err = update_out_dense(c/norm,n_vec,&in[symm],&out[full]);

				symm += n_vec;
			}
			c *= cc;
			B->map_state(basis,Ns,nt-nnt);
		}

		return err;
	}



}


#endif
